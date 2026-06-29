"""
train.py
--------
Training and inference script for BayImputationGNN.

Usage
-----
  # Train from scratch
  python train.py --train

  # Run imputation on new data (fills gaps in X, saves output CSV)
  python train.py --impute --checkpoint checkpoints/best.pt

Masking strategy
----------------
During training we randomly corrupt a fraction of known observations
(MASK_RATIO=0.3 by default) so the model is forced to reconstruct them
from spatial neighbours and temporal context.  This teaches it exactly
the behaviour we want at inference: fill real gaps intelligently.

Window-based processing
-----------------------
The full time series can be millions of rows.  We slice it into
overlapping windows of length WIN_LEN and stride STRIDE, train on
those, then stitch the imputations back together at inference.
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from preprocess import build_dataset
from model import BayImputationGNN, Normaliser


# ---------------------------------------------------------------------------
# Hyper-parameters (adjust as needed)
# ---------------------------------------------------------------------------
WIN_LEN    = 72           # 6 h at 5-min resolution
STRIDE     = 36           # 50 % overlap
MASK_RATIO = 0.30         # fraction of known values corrupted during training
BATCH_SIZE = 16
D_MODEL    = 64
N_GAT      = 3
N_GRU      = 2
LR         = 3e-4
EPOCHS     = 50
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR   = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Yields (x_win, mask_win, forcing_win, ts_win) windows.
    x_win      : (WIN_LEN, N, F)  normalised
    mask_win   : (WIN_LEN, N, F)  1=present, 0=missing
    forcing_win: (WIN_LEN, 3)     [rain, temp_min, temp_max]
    ts_win     : (WIN_LEN,)       unix timestamps (float)
    """

    def __init__(self, X_norm, X_raw_mask, forcing, timestamps, mask_ratio=MASK_RATIO):
        self.X = X_norm                   # (T, N, F)
        self.true_mask = X_raw_mask       # (T, N, F) 1=real data
        self.forcing = forcing            # (T, 3)
        self.ts = timestamps              # (T,)
        self.mask_ratio = mask_ratio
        self.windows = list(range(0, len(X_norm) - WIN_LEN + 1, STRIDE))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s = self.windows[idx]
        e = s + WIN_LEN

        x   = self.X[s:e].copy()            # (W, N, F)
        msk = self.true_mask[s:e].copy()    # (W, N, F)

        if self.mask_ratio > 0:
            rng = np.random.default_rng()
            corruption = rng.random(msk.shape) < self.mask_ratio
            extra_mask = msk.astype(bool) & corruption
            msk[extra_mask] = 0
            x[extra_mask] = 0

        return (
            torch.tensor(x,                  dtype=torch.float32),
            torch.tensor(msk,                dtype=torch.float32),
            torch.tensor(self.forcing[s:e],  dtype=torch.float32),
            torch.tensor(self.ts[s:e],       dtype=torch.float64),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print(f"Device: {DEVICE}")

    # -- Data -----------------------------------------------------------------
    print("\n=== Preprocessing ===")
    data = build_dataset()
    X_raw       = data["X"]             # (T, N, F) with NaNs
    rain        = data["rain"]          # (T,)
    temp_min    = data["temp_min"]      # (T,)
    temp_max    = data["temp_max"]      # (T,)
    forcing     = np.stack([rain, temp_min, temp_max], axis=-1)  # (T, 3)
    edge_index  = data["edge_index"]
    edge_weight = data["edge_weight"]
    ts = np.array([t.timestamp() for t in data["time_index"]], dtype=np.float64)

    T, N, F = X_raw.shape
    print(f"Data shape: {T} timesteps × {N} nodes × {F} features")

    # -- Normalise (fit only on observed values) ------------------------------
    norm = Normaliser()
    norm.fit(X_raw)
    X_norm = norm.transform(np.nan_to_num(X_raw, nan=0.0))
    true_mask = (~np.isnan(X_raw)).astype(np.float32)

    # Save normaliser alongside checkpoint
    with open(CKPT_DIR / "normaliser.pkl", "wb") as f:
        pickle.dump(norm, f)

    # -- Split: 80 % train, 20 % val (chronological) -------------------------
    split = int(0.8 * T)
    train_ds = WindowDataset(X_norm[:split],  true_mask[:split],
                             forcing[:split], ts[:split])
    val_ds   = WindowDataset(X_norm[split:],  true_mask[split:],
                             forcing[split:], ts[split:],
                             mask_ratio=0.0)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -- Model -----------------------------------------------------------------
    model = BayImputationGNN(
        n_features=F, n_nodes=N, n_forcing=3,
        d_model=D_MODEL, n_gat_layers=N_GAT, n_gru_layers=N_GRU,
    ).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Graph tensors (static across all batches)
    ei = torch.tensor(edge_index,  dtype=torch.long).to(DEVICE)
    ew = torch.tensor(edge_weight, dtype=torch.float32).to(DEVICE)

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=1.0)   # robust to occasional outliers

    best_val = float("inf")

    # -- Training loop ---------------------------------------------------------
    print("\n=== Training ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x_b, msk_b, forcing_b, ts_b in train_loader:
            x_b       = x_b.to(DEVICE)        # (B, W, N, F)
            msk_b     = msk_b.to(DEVICE)
            forcing_b = forcing_b.to(DEVICE)  # (B, W, 3)
            ts_b      = ts_b.float().to(DEVICE)

            _, pred, _ = model(x_b, msk_b, forcing_b, ei, ew, ts_b)
            loss = criterion(pred * msk_b, x_b * msk_b)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # -- Validation --------------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, msk_b, forcing_b, ts_b in val_loader:
                x_b = x_b.to(DEVICE); msk_b = msk_b.to(DEVICE)
                forcing_b = forcing_b.to(DEVICE); ts_b = ts_b.float().to(DEVICE)
                _, pred, _ = model(x_b, msk_b, forcing_b, ei, ew, ts_b)
                val_loss += criterion(pred * msk_b, x_b * msk_b).item()
        val_loss /= max(len(val_loader), 1)

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "best.pt")
            flag = "  ✓ saved"

        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}{flag}")

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint: {CKPT_DIR / 'best.pt'}")


# ---------------------------------------------------------------------------
# Inference / imputation
# ---------------------------------------------------------------------------

def impute(checkpoint: str):
    print(f"\n=== Inference  (checkpoint: {checkpoint}) ===")
    import pandas as pd

    data = build_dataset()
    X_raw       = data["X"]
    rain        = data["rain"]
    temp_min    = data["temp_min"]
    temp_max    = data["temp_max"]
    forcing     = np.stack([rain, temp_min, temp_max], axis=-1)  # (T, 3)
    edge_index  = data["edge_index"]
    edge_weight = data["edge_weight"]
    ts = np.array([t.timestamp() for t in data["time_index"]], dtype=np.float64)
    T, N, F = X_raw.shape

    with open(CKPT_DIR / "normaliser.pkl", "rb") as f:
        norm: Normaliser = pickle.load(f)

    X_norm    = norm.transform(np.nan_to_num(X_raw, nan=0.0))
    true_mask = (~np.isnan(X_raw)).astype(np.float32)

    model = BayImputationGNN(n_features=F, n_nodes=N, n_forcing=3,
                              d_model=D_MODEL, n_gat_layers=N_GAT,
                              n_gru_layers=N_GRU).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    ei = torch.tensor(edge_index,  dtype=torch.long).to(DEVICE)
    ew = torch.tensor(edge_weight, dtype=torch.float32).to(DEVICE)

    # Sliding-window inference with averaging in overlapping regions
    X_imputed_sum   = np.zeros_like(X_raw, dtype=np.float64)
    X_imputed_count = np.zeros(T, dtype=np.float64)

    windows = list(range(0, T - WIN_LEN + 1, STRIDE))
    if windows[-1] + WIN_LEN < T:
        windows.append(T - WIN_LEN)   # ensure tail is covered

    with torch.no_grad():
        for s in windows:
            e = s + WIN_LEN
            x_win       = torch.tensor(X_norm[s:e],   dtype=torch.float32).to(DEVICE)
            msk_win     = torch.tensor(true_mask[s:e], dtype=torch.float32).to(DEVICE)
            forcing_win = torch.tensor(forcing[s:e],   dtype=torch.float32).to(DEVICE)
            ts_win      = torch.tensor(ts[s:e],        dtype=torch.float32).to(DEVICE)

            imputed, _, _ = model(x_win, msk_win, forcing_win, ei, ew, ts_win)
            imputed_np = imputed.cpu().numpy()           # (W, N, F)

            X_imputed_sum[s:e]   += imputed_np
            X_imputed_count[s:e] += 1

    X_imputed_avg = X_imputed_sum / np.maximum(X_imputed_count[:, None, None], 1)
    X_imputed_denorm = norm.inverse_transform(X_imputed_avg.astype(np.float32))

    # Build output: use real observations where available, model fill for gaps
    X_out = np.where(true_mask.astype(bool), X_raw, X_imputed_denorm)

    # Save one CSV per node
    out_dir = Path("imputed_output")
    out_dir.mkdir(exist_ok=True)
    for i, name in enumerate(data["node_names"]):
        df = pd.DataFrame(
            X_out[:, i, :],
            index=data["time_index"],
            columns=data["feature_names"],
        )
        df.index.name = "datetime"
        # Annotate which values were imputed vs observed
        mask_df = pd.DataFrame(
            true_mask[:, i, :].astype(int),
            index=data["time_index"],
            columns=[c + "_observed" for c in data["feature_names"]],
        )
        out = pd.concat([df, mask_df], axis=1)
        path = out_dir / f"{name}_imputed.csv"
        out.to_csv(path)
        n_imputed = (true_mask[:, i, :] == 0).sum()
        print(f"  {name}: {n_imputed} values imputed → {path}")

    print(f"\nDone. Results in {out_dir}/")
    return X_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bay ST-GNN imputation")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--impute",     action="store_true")
    parser.add_argument("--checkpoint", default=str(CKPT_DIR / "best.pt"))
    args = parser.parse_args()

    if args.train:
        train()
    if args.impute:
        impute(args.checkpoint)
    if not args.train and not args.impute:
        print("Specify --train and/or --impute")
        parser.print_help()
