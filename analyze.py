"""
analyze.py
----------
Feature and spatial importance analysis for the trained BayImputationGNN.

Run after training:
    python analyze.py --checkpoint checkpoints/best.pt

Produces four analyses printed to the console and saved as CSVs in analysis/:

  1. channel_importance.csv
     For each of the 8 sensor features: how much does permanently masking it
     hurt reconstruction of ALL other features?  (permutation importance)

  2. cross_feature_matrix.csv
     (F x F) matrix.  Entry [src, tgt] = extra reconstruction loss on 'tgt'
     when 'src' is also masked alongside 'tgt'. Reveals which features help
     predict which others.

  3. spatial_importance.csv
     (N x N) matrix.  Entry [i, j] = loss increase at node j when its
     incoming edge from node i is removed.  Shows which station-to-station
     links are load-bearing for imputation.

  4. attention_weights.csv
     (N x N) matrix of mean GAT attention weights averaged over time and
     all GAT layers.  Complements the spatial importance analysis.
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from preprocess import build_dataset, ALL_FEATURES
from model import BayImputationGNN, Normaliser
from train import (WIN_LEN, STRIDE, BATCH_SIZE, D_MODEL, N_GAT, N_GRU,
                   DEVICE, CKPT_DIR, WindowDataset)

OUT_DIR = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)

# Number of validation windows used for importance estimation (cap for speed)
MAX_WINDOWS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_and_data(checkpoint):
    with open(CKPT_DIR / "normaliser.pkl", "rb") as f:
        norm: Normaliser = pickle.load(f)

    data      = build_dataset()
    X_raw     = data["X"]
    rain      = data["rain"]
    temp_min  = data["temp_min"]
    temp_max  = data["temp_max"]
    forcing   = np.stack([rain, temp_min, temp_max], axis=-1)  # (T, 3)
    ei        = torch.tensor(data["edge_index"],  dtype=torch.long).to(DEVICE)
    ew        = torch.tensor(data["edge_weight"], dtype=torch.float32).to(DEVICE)
    ts        = np.array([t.timestamp() for t in data["time_index"]], dtype=np.float64)
    T, N, F   = X_raw.shape

    X_norm    = norm.transform(np.nan_to_num(X_raw, nan=0.0))
    true_mask = (~np.isnan(X_raw)).astype(np.float32)

    split     = int(0.8 * T)
    val_ds    = WindowDataset(X_norm[split:], true_mask[split:],
                              forcing[split:], ts[split:], mask_ratio=0.0)

    model = BayImputationGNN(n_features=F, n_nodes=N, n_forcing=3,
                              d_model=D_MODEL, n_gat_layers=N_GAT,
                              n_gru_layers=N_GRU).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    return model, val_ds, norm, ei, ew, data


def _compute_loss(model, val_ds, ei, ew, extra_mask_channels=None,
                  remove_edge_idx=None, ew_override=None):
    """
    Compute mean HuberLoss on validation windows.

    extra_mask_channels : list of int — feature indices to always mask to 0
    remove_edge_idx     : int         — index into edge list to zero out
    ew_override         : tensor      — replacement edge weights
    """
    criterion = nn.HuberLoss(delta=1.0)
    loader    = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    total, count = 0.0, 0

    ew_use = ew_override if ew_override is not None else ew

    with torch.no_grad():
        for x_b, msk_b, forcing_b, ts_b in loader:
            if count >= MAX_WINDOWS:
                break
            x_b   = x_b.to(DEVICE)
            msk_b = msk_b.to(DEVICE)
            if extra_mask_channels:
                msk_b = msk_b.clone()
                x_b   = x_b.clone()
                for ch in extra_mask_channels:
                    msk_b[..., ch] = 0.0
                    x_b[..., ch]   = 0.0
            forcing_b = forcing_b.to(DEVICE)
            ts_b      = ts_b.float().to(DEVICE)
            _, pred, _ = model(x_b, msk_b, forcing_b, ei, ew_use, ts_b)
            loss = criterion(pred * msk_b, x_b * msk_b)
            total += loss.item()
            count += 1

    return total / max(count, 1)


# ---------------------------------------------------------------------------
# 1. Channel permutation importance
# ---------------------------------------------------------------------------

def channel_importance(model, val_ds, ei, ew):
    print("\n=== 1. Channel permutation importance ===")
    baseline = _compute_loss(model, val_ds, ei, ew)
    print(f"Baseline loss: {baseline:.6f}")

    rows = []
    for i, feat in enumerate(ALL_FEATURES):
        masked_loss = _compute_loss(model, val_ds, ei, ew, extra_mask_channels=[i])
        delta = masked_loss - baseline
        rows.append({"feature": feat, "baseline_loss": baseline,
                     "masked_loss": masked_loss, "importance": delta})
        print(f"  {feat:20s}: masked={masked_loss:.6f}  delta={delta:+.6f}")

    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    path = OUT_DIR / "channel_importance.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved → {path}")
    return df


# ---------------------------------------------------------------------------
# 2. Cross-feature dependency matrix
# ---------------------------------------------------------------------------

def cross_feature_matrix(model, val_ds, ei, ew):
    print("\n=== 2. Cross-feature dependency matrix ===")
    F = len(ALL_FEATURES)
    matrix = np.zeros((F, F), dtype=np.float32)

    for tgt_i, tgt_feat in enumerate(ALL_FEATURES):
        base_tgt = _compute_loss(model, val_ds, ei, ew,
                                 extra_mask_channels=[tgt_i])
        for src_i, src_feat in enumerate(ALL_FEATURES):
            if src_i == tgt_i:
                continue
            both_loss = _compute_loss(model, val_ds, ei, ew,
                                      extra_mask_channels=[src_i, tgt_i])
            matrix[src_i, tgt_i] = both_loss - base_tgt

    df = pd.DataFrame(matrix, index=ALL_FEATURES, columns=ALL_FEATURES)
    path = OUT_DIR / "cross_feature_matrix.csv"
    df.to_csv(path)
    print(f"Saved → {path}")

    # Print a compact table
    print("\n  Rows=source masked, Cols=target hurt (positive = source helps target)")
    print(df.round(5).to_string())
    return df


# ---------------------------------------------------------------------------
# 3. Spatial edge importance
# ---------------------------------------------------------------------------

def spatial_importance(model, val_ds, ei, ew, node_names):
    print("\n=== 3. Spatial edge importance ===")
    E = ew.shape[0]
    N = len(node_names)
    short = [n.replace("raw-data-platformL", "L").replace("_parameters", "")
             for n in node_names]

    matrix = np.zeros((N, N), dtype=np.float32)
    src_arr = ei[0].cpu().numpy()
    dst_arr = ei[1].cpu().numpy()

    baseline = _compute_loss(model, val_ds, ei, ew)

    for k in range(E):
        ew_mod = ew.clone()
        ew_mod[k] = 0.0
        delta = _compute_loss(model, val_ds, ei, ew, ew_override=ew_mod) - baseline
        matrix[src_arr[k], dst_arr[k]] = delta

    df = pd.DataFrame(matrix, index=short, columns=short)
    path = OUT_DIR / "spatial_importance.csv"
    df.to_csv(path)
    print(f"Saved → {path}")

    # Top-10 most important edges
    rows = []
    for k in range(E):
        rows.append({"from": short[src_arr[k]], "to": short[dst_arr[k]],
                     "importance": matrix[src_arr[k], dst_arr[k]]})
    top = pd.DataFrame(rows).sort_values("importance", ascending=False).head(10)
    print("\n  Top 10 most important spatial edges:")
    print(top.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 4. Attention weight heatmap
# ---------------------------------------------------------------------------

def attention_heatmap(model, val_ds, ei, ew, node_names):
    print("\n=== 4. GAT attention weight heatmap ===")
    N     = len(node_names)
    short = [n.replace("raw-data-platformL", "L").replace("_parameters", "")
             for n in node_names]

    attn_accum = torch.zeros(N, N, device=DEVICE)
    count = 0
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    with torch.no_grad():
        for x_b, msk_b, forcing_b, ts_b in loader:
            if count >= 50:
                break
            x_b       = x_b.to(DEVICE)
            msk_b     = msk_b.to(DEVICE)
            forcing_b = forcing_b.to(DEVICE)
            ts_b      = ts_b.float().to(DEVICE)
            _, _, attn_mat = model(x_b, msk_b, forcing_b, ei, ew, ts_b,
                                    return_attention=True)
            if attn_mat is not None:
                attn_accum += attn_mat
                count += 1

    attn_avg = (attn_accum / max(count, 1)).cpu().numpy()
    df = pd.DataFrame(attn_avg, index=short, columns=short)
    path = OUT_DIR / "attention_weights.csv"
    df.to_csv(path)
    print(f"Saved → {path}")

    print("\n  Mean attention matrix (row=target node, col=source node):")
    print(df.round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Summary + recommendations
# ---------------------------------------------------------------------------

def print_summary(ch_imp, cross_df, attn_df):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nFeature ranking by imputation importance (most → least):")
    imp_abs = ch_imp["importance"].abs()
    imp_max = imp_abs.max()
    for _, row in ch_imp.iterrows():
        ratio = imp_abs.loc[row.name] / imp_max if imp_max > 0 else 0.0
        bar = "#" * int(ratio * 20 + 0.5)
        print(f"  {row['feature']:20s}  {row['importance']:+.6f}  {bar}")

    print("\nStrongest cross-feature dependencies (src → tgt):")
    flat = cross_df.stack().reset_index()
    flat.columns = ["src", "tgt", "delta"]
    flat = flat[flat["src"] != flat["tgt"]].sort_values("delta", ascending=False)
    for _, r in flat.head(8).iterrows():
        print(f"  {r['src']:20s} → {r['tgt']:20s}   {r['delta']:+.6f}")

    print("\nMost attentive station pairs (who listens to whom):")
    flat_a = attn_df.stack().reset_index()
    flat_a.columns = ["tgt", "src", "weight"]
    flat_a = flat_a[flat_a["tgt"] != flat_a["src"]].sort_values("weight", ascending=False)
    for _, r in flat_a.head(8).iterrows():
        print(f"  {r['src']:30s} → {r['tgt']:30s}   attn={r['weight']:.4f}")

    print("\nRecommendation: features with HIGH importance score are critical")
    print("  for spatial imputation — avoid masking or dropping them.")
    print("  Features with LOW importance are candidates for dimensionality")
    print("  reduction or exclusion if sensors fail.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--skip-cross", action="store_true",
                        help="Skip the slow cross-feature matrix (F*F passes)")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} ...")
    model, val_ds, norm, ei, ew, data = _load_model_and_data(args.checkpoint)
    node_names = data["node_names"]

    ch_imp   = channel_importance(model, val_ds, ei, ew)
    attn_df  = attention_heatmap(model, val_ds, ei, ew, node_names)
    sp_df    = spatial_importance(model, val_ds, ei, ew, node_names)

    if not args.skip_cross:
        cross_df = cross_feature_matrix(model, val_ds, ei, ew)
    else:
        cross_df = pd.read_csv(OUT_DIR / "cross_feature_matrix.csv", index_col=0) \
                   if (OUT_DIR / "cross_feature_matrix.csv").exists() \
                   else pd.DataFrame()

    if not cross_df.empty:
        print_summary(ch_imp, cross_df, attn_df)
