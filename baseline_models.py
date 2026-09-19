"""
baseline_models.py
-------------------
Naive imputation baselines for the multi-node bay dataset, evaluated under
the same artificial-masking protocol used to train/validate the GNN so
results are directly comparable.

Baselines
---------
  locf          - last observation carried forward (back-filled for
                  leading gaps)
  linear_interp - per-node/per-feature linear interpolation in time
  climatology   - per-node/per-feature hour-of-day mean, fit on the
                  training period only (no validation leakage)
  spatial_mean  - mean of all other nodes' observed value for that
                  feature at the same timestep

Protocol
--------
  1. Split chronologically the same way as train.py (80% train / 20% val).
  2. Within the validation period, corrupt MASK_RATIO of the genuinely
     observed values (fixed seed) - these held-out values are the
     evaluation targets, never seen by any method.
  3. Each baseline (and, if a checkpoint exists, the GNN) predicts the
     held-out positions using only the remaining information.
  4. RMSE / MAE are computed at the held-out positions only, in raw
     physical units, per feature (all nodes) and per node/feature.

Run:
    python baseline_models.py [--checkpoint checkpoints/best.pt] [--skip-gnn]

Output:
    analysis/baseline_comparison.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import build_dataset

ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("checkpoints")

MASK_RATIO = 0.30
MASK_SEED = 1234


# ---------------------------------------------------------------------------
# Evaluation-mask construction
# ---------------------------------------------------------------------------

def make_eval_mask(true_mask: np.ndarray, mask_ratio: float, seed: int) -> np.ndarray:
    """True where a genuinely observed value is held out for evaluation."""
    rng = np.random.default_rng(seed)
    corruption = rng.random(true_mask.shape) < mask_ratio
    return true_mask.astype(bool) & corruption


def corrupted_series(x_val: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    """x_val with NaN at both genuinely-missing and held-out positions."""
    corrupted = x_val.copy()
    corrupted[eval_mask] = np.nan
    return corrupted


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_locf(corrupted: np.ndarray) -> np.ndarray:
    _, n_nodes, _ = corrupted.shape
    out = corrupted.copy()
    for node in range(n_nodes):
        out[:, node, :] = pd.DataFrame(corrupted[:, node, :]).ffill().bfill().to_numpy()
    return out


def baseline_linear_interp(corrupted: np.ndarray) -> np.ndarray:
    _, n_nodes, _ = corrupted.shape
    out = corrupted.copy()
    for node in range(n_nodes):
        out[:, node, :] = (
            pd.DataFrame(corrupted[:, node, :])
            .interpolate(method="linear", limit_direction="both")
            .to_numpy()
        )
    return out


def baseline_climatology(corrupted, train_x, train_mask, val_hours, train_hours) -> np.ndarray:
    """Per-node/per-feature hour-of-day mean, fit on the training period only."""
    _, n_nodes, n_features = corrupted.shape
    out = corrupted.copy()
    for node in range(n_nodes):
        for feat in range(n_features):
            col_train = train_x[:, node, feat]
            observed = train_mask[:, node, feat].astype(bool)
            hourly_mean = np.full(24, np.nan)
            for hour in range(24):
                sel = observed & (train_hours == hour)
                if sel.any():
                    hourly_mean[hour] = col_train[sel].mean()
            fallback = np.nanmean(col_train[observed]) if observed.any() else 0.0
            hourly_mean[np.isnan(hourly_mean)] = fallback
            nan_pos = np.isnan(out[:, node, feat])
            out[nan_pos, node, feat] = hourly_mean[val_hours[nan_pos]]
    return out


def baseline_spatial_mean(corrupted: np.ndarray) -> np.ndarray:
    """Mean across all other nodes' observed value for the same feature/timestep."""
    _, n_nodes, n_features = corrupted.shape
    out = corrupted.copy()
    observed = ~np.isnan(corrupted)
    for feat in range(n_features):
        vals = corrupted[:, :, feat]
        mask = observed[:, :, feat]
        sums = np.nansum(np.where(mask, vals, 0.0), axis=1)
        counts = mask.sum(axis=1)
        for node in range(n_nodes):
            other_sum = sums - np.where(mask[:, node], vals[:, node], 0.0)
            other_count = counts - mask[:, node]
            with np.errstate(invalid="ignore", divide="ignore"):
                estimate = np.where(other_count > 0, other_sum / np.maximum(other_count, 1), np.nan)
            nan_pos = np.isnan(out[:, node, feat])
            out[nan_pos, node, feat] = estimate[nan_pos]
    return out


# ---------------------------------------------------------------------------
# GNN evaluation (optional, requires a trained checkpoint)
# ---------------------------------------------------------------------------

def evaluate_gnn(checkpoint_path, x_val, true_mask_val, eval_mask, forcing_val, ts_val,
                  edge_index, edge_weight, feature_names):
    """Run the trained GNN under the identical held-out-value protocol."""
    import torch
    from model import BayImputationGNN, Normaliser
    from train import WIN_LEN, STRIDE, D_MODEL, N_GAT, N_GRU, DEVICE
    from preprocess import clip_to_physical_bounds

    with open(CKPT_DIR / "normaliser.pkl", "rb") as fh:
        norm: Normaliser = pickle.load(fh)

    n_steps, n_nodes, n_features = x_val.shape
    # Hide the evaluation targets from the model, exactly like real missing data.
    input_mask = true_mask_val.astype(bool) & ~eval_mask
    x_for_model = x_val.copy()
    x_for_model[~input_mask] = np.nan
    x_norm = norm.transform(np.nan_to_num(x_for_model, nan=0.0))
    input_mask_f = input_mask.astype(np.float32)

    model = BayImputationGNN(n_features=n_features, n_nodes=n_nodes, n_forcing=3,
                              d_model=D_MODEL, n_gat_layers=N_GAT,
                              n_gru_layers=N_GRU).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(DEVICE)
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32).to(DEVICE)

    pred_sum = np.zeros((n_steps, n_nodes, n_features), dtype=np.float64)
    pred_count = np.zeros(n_steps, dtype=np.float64)

    windows = list(range(0, n_steps - WIN_LEN + 1, STRIDE))
    if not windows:
        windows = [0]
    elif windows[-1] + WIN_LEN < n_steps:
        windows.append(n_steps - WIN_LEN)

    with torch.no_grad():
        for s in windows:
            e = s + WIN_LEN
            x_win = torch.tensor(x_norm[s:e], dtype=torch.float32).to(DEVICE)
            msk_win = torch.tensor(input_mask_f[s:e], dtype=torch.float32).to(DEVICE)
            forcing_win = torch.tensor(forcing_val[s:e], dtype=torch.float32).to(DEVICE)
            ts_win = torch.tensor(ts_val[s:e], dtype=torch.float32).to(DEVICE)

            _, pred, _ = model(x_win, msk_win, forcing_win, edge_index_t, edge_weight_t, ts_win)
            pred_sum[s:e] += pred.cpu().numpy()
            pred_count[s:e] += 1

    pred_avg = pred_sum / np.maximum(pred_count[:, None, None], 1)
    pred_denorm = norm.inverse_transform(pred_avg.astype(np.float32))
    return clip_to_physical_bounds(pred_denorm, feature_names)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred, truth, eval_mask, feature_names, node_names) -> list[dict]:
    rows = []
    for feat_idx, feat in enumerate(feature_names):
        mask = eval_mask[:, :, feat_idx] & ~np.isnan(pred[:, :, feat_idx])
        if mask.sum() == 0:
            continue
        err = pred[:, :, feat_idx][mask] - truth[:, :, feat_idx][mask]
        rows.append({
            "feature": feat, "node": "ALL", "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
        })
    for node_idx, node in enumerate(node_names):
        for feat_idx, feat in enumerate(feature_names):
            mask = eval_mask[:, node_idx, feat_idx] & ~np.isnan(pred[:, node_idx, feat_idx])
            if mask.sum() == 0:
                continue
            err = pred[:, node_idx, feat_idx][mask] - truth[:, node_idx, feat_idx][mask]
            rows.append({
                "feature": feat, "node": node, "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Naive baselines vs GNN, matched masking protocol")
    parser.add_argument("--checkpoint", default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--skip-gnn", action="store_true")
    args = parser.parse_args()

    print("Loading dataset...")
    data = build_dataset()
    x_all = data["X"]
    forcing_all = np.stack([data["rain"], data["temp_min"], data["temp_max"]], axis=-1)
    ts_all = np.array([t.timestamp() for t in data["time_index"]], dtype=np.float64)
    feature_names = data["feature_names"]
    node_names = data["node_names"]
    n_steps = x_all.shape[0]

    split = int(0.8 * n_steps)
    x_train, x_val = x_all[:split], x_all[split:]
    forcing_val = forcing_all[split:]
    ts_val = ts_all[split:]
    train_mask = (~np.isnan(x_train)).astype(np.float32)
    val_true_mask = (~np.isnan(x_val)).astype(np.float32)

    train_hours = pd.DatetimeIndex(data["time_index"][:split]).hour.to_numpy()
    val_hours = pd.DatetimeIndex(data["time_index"][split:]).hour.to_numpy()

    print("Building evaluation mask (held-out genuinely-observed values)...")
    eval_mask = make_eval_mask(val_true_mask, MASK_RATIO, MASK_SEED)
    n_obs = int(val_true_mask.sum())
    print(f"  held out {eval_mask.sum():,} / {n_obs:,} observed values "
          f"({eval_mask.sum() / max(n_obs, 1) * 100:.1f}%)")

    corrupted = corrupted_series(x_val, eval_mask)

    results = {}
    print("Running LOCF...")
    results["locf"] = baseline_locf(corrupted)
    print("Running linear interpolation...")
    results["linear_interp"] = baseline_linear_interp(corrupted)
    print("Running climatology (hour-of-day, train period only)...")
    results["climatology"] = baseline_climatology(corrupted, x_train, train_mask, val_hours, train_hours)
    print("Running spatial-neighbour mean...")
    results["spatial_mean"] = baseline_spatial_mean(corrupted)

    checkpoint_path = Path(args.checkpoint)
    if not args.skip_gnn and checkpoint_path.exists() and (CKPT_DIR / "normaliser.pkl").exists():
        print(f"Running GNN ({checkpoint_path})...")
        results["gnn"] = evaluate_gnn(
            checkpoint_path, x_val, val_true_mask, eval_mask, forcing_val, ts_val,
            data["edge_index"], data["edge_weight"], feature_names,
        )
    else:
        print("  [SKIP] GNN comparison omitted (--skip-gnn or no checkpoint found)")

    all_rows = []
    for method, pred in results.items():
        for row in compute_metrics(pred, x_val, eval_mask, feature_names, node_names):
            row["method"] = method
            all_rows.append(row)

    report = pd.DataFrame(all_rows)[["method", "feature", "node", "n", "rmse", "mae"]]
    out_path = ANALYSIS_DIR / "baseline_comparison.csv"
    report.to_csv(out_path, index=False, float_format="%.6g")
    print(f"\nSaved -> {out_path}")

    print("\nPer-feature RMSE (all nodes combined), lower is better:")
    summary = report[report["node"] == "ALL"].pivot(index="feature", columns="method", values="rmse")
    print(summary.round(4).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
