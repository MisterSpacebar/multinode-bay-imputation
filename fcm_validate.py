"""
fcm_validate.py
---------------
Validation of the Nutrient FCM against observed monthly data.

Two analyses:

  1. One-step-ahead hindcast  (in-sample validation)
     For every month t, plug the observed normalised activations A(t) into
     the FCM to predict A(t+1).  Compare the FCM prediction against the
     actual observed A(t+1).

     Metrics per concept:
       R²   — fraction of variance explained
       RMSE — root-mean-square error in normalised space [0,1]
       MAE  — mean absolute error in normalised space

     Plots:
       21_fcm_hindcast_scatter.png  — predicted vs actual scatter, one panel
                                      per endogenous concept (4×3 grid)
       22_fcm_hindcast_timeseries.png — observed vs predicted time series for
                                       the 6 most informative concepts

  2. Event backcasting
     Seed the FCM with observed conditions 3 months before each die-off event,
     let it roll forward 6 months, and overlay the simulated trajectory on top
     of the actual observed values.

     Plots:
       23_fcm_backcast_sept2021.png
       24_fcm_backcast_oct2022.png

Run:
    python fcm_validate.py

Outputs
-------
  analysis/fcm_hindcast_metrics.csv
  visualizations/21_fcm_hindcast_scatter.png
  visualizations/22_fcm_hindcast_timeseries.png
  visualizations/23_fcm_backcast_sept2021.png
  visualizations/24_fcm_backcast_oct2022.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Re-use loaders and constants from fcm.py
from fcm import (
    load_nutrient_concept_timeseries,
    learn_fcm_weights,
    fcm_simulate,
    NUTR_CONCEPTS, N_FORCING, N_NUTR, CONCEPT_LABELS,
    ANALYSIS_DIR, VIZ_DIR,
)

METRICS_CSV = ANALYSIS_DIR / "fcm_hindcast_metrics.csv"

# ---------------------------------------------------------------------------
# Concept display helpers
# ---------------------------------------------------------------------------
LABELS = [CONCEPT_LABELS.get(c, c) for c in NUTR_CONCEPTS]

# Endogenous (non-forcing) concept indices
ENDO_IDXS  = list(range(N_FORCING, N_NUTR))
ENDO_LABELS = [LABELS[i] for i in ENDO_IDXS]

# Concepts to highlight in timeseries plot
KEY_CONCEPTS = ["Water Temp", "Salinity", "ODO (mg/L)", "DO (%Sat)",
                "DIN", "Chlorophyll-a"]

# Die-off events: (label, seed_month, event_month, n_steps, color)
EVENTS = [
    {
        "label":       "Sept 2021 — Fish & Seagrass Die-off",
        "short":       "Sept 2021",
        "seed_month":  "2021-09",   # first available month (grab samples start here)
        "event_month": "2021-09",   # event is the seed month itself
        "n_steps":     6,
        "color":       "#d62728",
    },
    {
        "label":       "Oct 2022 — Seagrass Die-off",
        "short":       "Oct 2022",
        "seed_month":  "2022-08",
        "event_month": "2022-10",
        "n_steps":     6,
        "color":       "#ff7f0e",
    },
]

# Backcast key concepts (ones most relevant to die-offs)
BACKCAST_CONCEPTS = ["DO (%Sat)", "ODO (mg/L)", "DIN", "Water Temp",
                     "Salinity", "Chlorophyll-a"]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                             pd.PeriodIndex, pd.DataFrame, np.ndarray]:
    """
    Returns:
      W          — (16, 16) FCM weight matrix
      arr_norm   — (T, 16) normalised series with NaN preserved for true obs
      col_min    — (16,)  per-concept min
      col_range  — (16,)  per-concept range
      index      — (T,)   monthly PeriodIndex
      ts_clean   — original DataFrame (NaN = genuinely missing)
      obs_mask   — (T, 16) bool — True where the original grab/sensor data
                   was actually observed (not NaN-filled)
    """
    ts = load_nutrient_concept_timeseries()
    W, _, col_min, col_range = learn_fcm_weights(
        ts, NUTR_CONCEPTS, N_FORCING, alpha=1.5, lag=1, tag="Validation"
    )
    # Drop rows where ALL endogenous concepts are missing
    endo_cols = [NUTR_CONCEPTS[i] for i in ENDO_IDXS]
    ts_clean  = ts[NUTR_CONCEPTS].dropna(subset=endo_cols, how="all")
    index     = ts_clean.index.to_period("M")

    # True observation mask BEFORE NaN-filling
    obs_mask = ~ts_clean.isna().values   # (T, 16) bool

    # Normalised array — keep NaN where data was absent
    arr      = ts_clean.values.astype(np.float64)
    arr_norm = (arr - col_min) / col_range   # NaN preserved

    return W, arr_norm, col_min, col_range, index, ts_clean, obs_mask


# ---------------------------------------------------------------------------
# 1. One-step-ahead hindcast
# ---------------------------------------------------------------------------

def one_step_hindcast(
    W:        np.ndarray,
    arr_norm: np.ndarray,
    obs_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predicted[t] = sigmoid(W^T @ filled_A[t])  where A[t] is the
    observation at t with NaN replaced by column means (so the FCM
    can still run), but we only EVALUATE the prediction at t+1
    positions where the data was genuinely observed.

    Returns:
      pred      — (T-1, 16)  FCM one-step predictions
      obs       — (T-1, 16)  actual observed values (NaN where not measured)
      eval_mask — (T-1, 16)  bool — True where BOTH A(t) and A(t+1) were
                              genuinely observed (valid evaluation pairs)
    """
    T = arr_norm.shape[0]

    # Column means for NaN-filling inputs (so the matrix multiply works)
    col_means = np.nanmean(arr_norm, axis=0)
    col_means[np.isnan(col_means)] = 0.5

    pred = np.full((T - 1, N_NUTR), np.nan)
    obs  = arr_norm[1:].copy()   # NaN preserved where genuinely missing

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    for t in range(T - 1):
        A_t = arr_norm[t].copy()
        A_t[np.isnan(A_t)] = col_means[np.isnan(A_t)]   # fill NaN for input
        A_pred = _sig(W.T @ A_t)
        A_pred[:N_FORCING] = arr_norm[t + 1, :N_FORCING]  # forcing clamped
        pred[t] = A_pred

    # Valid evaluation: both t and t+1 must have been genuinely observed
    eval_mask = obs_mask[:-1] & obs_mask[1:]   # (T-1, 16)

    return pred, obs, eval_mask


def compute_metrics(
    pred:      np.ndarray,
    obs:       np.ndarray,
    eval_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Per-concept R², RMSE, MAE using only genuinely observed evaluation pairs.
    R² is clipped to [-2, 1] so a few bad predictions don't blow up the table.
    """
    rows = []
    for i in ENDO_IDXS:
        p    = pred[:, i]
        o    = obs[:,  i]
        mask = eval_mask[:, i] & ~(np.isnan(p) | np.isnan(o))
        n    = int(mask.sum())
        if n < 2:
            rows.append({"concept": LABELS[i], "R2": np.nan,
                         "RMSE": np.nan, "MAE": np.nan, "n": n})
            continue
        pm, om = p[mask], o[mask]
        ss_res = float(np.sum((om - pm) ** 2))
        ss_tot = float(np.sum((om - om.mean()) ** 2))
        # Guard against ss_tot ≈ 0 (concept barely varies in evaluation window)
        if ss_tot < 1e-10:
            r2 = np.nan
        else:
            r2 = float(np.clip(1 - ss_res / ss_tot, -2.0, 1.0))
        rmse = float(np.sqrt(np.mean((om - pm) ** 2)))
        mae  = float(np.mean(np.abs(om - pm)))
        rows.append({"concept": LABELS[i], "R2": r2,
                     "RMSE": rmse, "MAE": mae, "n": n})
    return pd.DataFrame(rows).set_index("concept")


def plot_hindcast_scatter(
    pred:      np.ndarray,
    obs:       np.ndarray,
    eval_mask: np.ndarray,
    metrics:   pd.DataFrame,
) -> None:
    n_cols = 4
    n_rows = int(np.ceil(len(ENDO_IDXS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, (ci, clabel) in enumerate(zip(ENDO_IDXS, ENDO_LABELS)):
        ax  = axes_flat[ax_i]
        p    = pred[:, ci]
        o    = obs[:,  ci]
        mask = eval_mask[:, ci] & ~(np.isnan(p) | np.isnan(o))
        pm, om = p[mask], o[mask]

        r2   = metrics.loc[clabel, "R2"]   if clabel in metrics.index else np.nan
        rmse = metrics.loc[clabel, "RMSE"] if clabel in metrics.index else np.nan

        # Scatter — colour by concept group, annotate n
        color = "#27ae60" if ci >= N_NUTR - 5 else "#2055c8"
        if len(pm) == 0:
            ax.text(0.5, 0.5, "no valid\nobservation pairs",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(clabel, fontsize=10, fontweight="bold")
            continue
        ax.scatter(om, pm, s=40, alpha=0.8, edgecolors="k", linewidths=0.4,
                   color=color)

        # Perfect prediction line
        lo  = min(om.min(), pm.min()) - 0.05
        hi  = max(om.max(), pm.max()) + 0.05
        lim = [lo, hi]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.text(lim[0] + 0.02, lim[1] - 0.05,
                f"n={len(pm)}", fontsize=8, color="#555")

        r2_str   = f"{r2:.3f}"   if not np.isnan(r2)   else "n/a"
        rmse_str = f"{rmse:.3f}" if not np.isnan(rmse) else "n/a"
        ax.set_title(clabel, fontsize=10, fontweight="bold")
        ax.set_xlabel("Observed A(t+1)", fontsize=8)
        ax.set_ylabel("FCM predicted A(t+1)", fontsize=8)
        ax.text(0.05, 0.93, f"R²={r2_str}  RMSE={rmse_str}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.grid(True, alpha=0.25)

    # Hide unused panels
    for ax in axes_flat[len(ENDO_IDXS):]:
        ax.axis("off")

    fig.suptitle(
        "FCM One-step-ahead Hindcast — Nutrient FCM (monthly, 2021–2026)\n"
        "Each panel: observed A(t+1) vs FCM-predicted A(t+1)  |  "
        "Values normalised to [0,1]  |  Dashed = perfect prediction",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = VIZ_DIR / "21_fcm_hindcast_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


def plot_hindcast_timeseries(
    pred:      np.ndarray,
    obs:       np.ndarray,
    eval_mask: np.ndarray,
    index:     pd.PeriodIndex,
    col_min:   np.ndarray,
    col_range: np.ndarray,
) -> None:
    """Observed vs predicted time series for key concepts, back-transformed.
    Predicted points are only plotted where eval_mask is True."""
    key_idxs = [i for i, l in enumerate(LABELS)
                if l in KEY_CONCEPTS and i in ENDO_IDXS][:6]
    key_labels = [LABELS[i] for i in key_idxs]

    # Dates aligned to pred (which starts at t=1)
    dates = index[1:].to_timestamp()

    n_cols = 2
    n_rows = int(np.ceil(len(key_idxs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13, 3.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, (ci, clabel) in enumerate(zip(key_idxs, key_labels)):
        ax = axes_flat[ax_i]

        # Back-transform to real units
        obs_real  = obs[:, ci]  * col_range[ci] + col_min[ci]
        pred_real = pred[:, ci] * col_range[ci] + col_min[ci]

        # Observed: plot wherever the data was genuinely measured
        obs_plot_mask  = ~np.isnan(obs_real)
        # Predicted: only plot at genuinely observed t+1 positions
        pred_plot_mask = eval_mask[:, ci] & ~np.isnan(pred_real)

        ax.plot(dates[obs_plot_mask], obs_real[obs_plot_mask], "o-", lw=2, ms=5,
                color="#1f4e79", label="Observed", alpha=0.9)
        ax.scatter(dates[pred_plot_mask], pred_real[pred_plot_mask], s=60,
                   marker="s", color="#c0392b", label="FCM predicted",
                   alpha=0.85, zorder=5, edgecolors="k", linewidths=0.5)

        # Shade die-off event months
        event_colors = {"2021-09": "#d62728", "2022-10": "#ff7f0e"}
        for month_str, ec in event_colors.items():
            ep = pd.Period(month_str, "M").to_timestamp()
            ax.axvspan(ep, ep + pd.offsets.MonthEnd(1),
                       color=ec, alpha=0.18, zorder=0)
            ax.text(ep + pd.Timedelta(days=5), ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                    "★", color=ec, fontsize=11, va="top")

        # Compute R² in normalised space for annotation (eval pairs only)
        valid = eval_mask[:, ci] & ~(np.isnan(obs[:, ci]) | np.isnan(pred[:, ci]))
        p_n = pred[:, ci][valid]
        o_n = obs[:, ci][valid]
        ss_res = np.sum((o_n - p_n) ** 2)
        ss_tot = np.sum((o_n - o_n.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2_str = f"R²={r2:.3f}" if not np.isnan(r2) else ""

        ax.set_title(f"{clabel}  {r2_str}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Value (real units)", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.autofmt_xdate(rotation=30, ha="right")

    for ax in axes_flat[len(key_idxs):]:
        ax.axis("off")

    # Legend for event markers
    ev_patches = [
        mpatches.Patch(color="#d62728", alpha=0.4, label="Sept 2021 die-off ★"),
        mpatches.Patch(color="#ff7f0e", alpha=0.4, label="Oct 2022 die-off ★"),
    ]
    fig.legend(handles=ev_patches, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "FCM Hindcast Time Series — Observed vs 1-step FCM Prediction\n"
        "Coloured bands = die-off event months  |  ★ = event month",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = VIZ_DIR / "22_fcm_hindcast_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# 2. Event backcasting
# ---------------------------------------------------------------------------

def _month_to_row(index: pd.PeriodIndex, month_str: str) -> int | None:
    """Return the integer row position of a given 'YYYY-MM' period."""
    target = pd.Period(month_str, "M")
    matches = np.where(index == target)[0]
    return int(matches[0]) if len(matches) else None


def plot_backcast(
    ev:        dict,
    W:         np.ndarray,
    arr_norm:  np.ndarray,
    col_min:   np.ndarray,
    col_range: np.ndarray,
    index:     pd.PeriodIndex,
    out_path:  Path,
) -> None:
    """
    Seed the FCM at seed_month, run forward n_steps months.
    Overlay simulated trajectory against observed values.
    Show 6 key concepts.
    """
    seed_row = _month_to_row(index, ev["seed_month"])
    if seed_row is None:
        print(f"  [SKIP] seed month {ev['seed_month']} not in data")
        return

    A0 = arr_norm[seed_row].copy()
    # Replace NaN with 0.5 (mid-range) for simulation stability
    A0[np.isnan(A0)] = 0.5

    # Forcing: hold at seed-month observed values throughout
    forcing_vals = A0[:N_FORCING].copy()
    n_steps      = ev["n_steps"]

    # Simulate
    traj = fcm_simulate(W, A0, list(range(N_FORCING)), forcing_vals, n_steps)
    # traj shape: (n_steps+1, 16)

    # Observed window: seed_row → seed_row + n_steps
    obs_window   = arr_norm[seed_row : seed_row + n_steps + 1]
    obs_dates    = index[seed_row : seed_row + n_steps + 1].to_timestamp()
    step_x       = np.arange(n_steps + 1)

    key_idxs  = [i for i, l in enumerate(LABELS) if l in BACKCAST_CONCEPTS][:6]
    key_labels = [LABELS[i] for i in key_idxs]

    n_cols = 2
    n_rows = int(np.ceil(len(key_idxs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    event_row = _month_to_row(index, ev["event_month"])
    event_step = (event_row - seed_row) if event_row is not None else None

    for ax_i, (ci, clabel) in enumerate(zip(key_idxs, key_labels)):
        ax = axes_flat[ax_i]

        # Back-transform to real units
        sim_real = traj[:, ci]       * col_range[ci] + col_min[ci]
        obs_real = obs_window[:, ci] * col_range[ci] + col_min[ci]
        obs_mask = ~np.isnan(obs_real)

        # Month labels on x-axis
        x_labels = [d.strftime("%b %Y") for d in obs_dates]

        ax.plot(step_x, sim_real, "s--", lw=2, ms=7,
                color=ev["color"], label="FCM simulation", alpha=0.9, zorder=5)
        ax.plot(step_x[obs_mask], obs_real[obs_mask], "o-", lw=2, ms=7,
                color="#1f4e79", label="Observed", alpha=0.9, zorder=4)

        # Mark event month
        if event_step is not None and 0 <= event_step <= n_steps:
            ax.axvspan(event_step - 0.4, event_step + 0.4,
                       color=ev["color"], alpha=0.2, zorder=0)
            ax.axvline(event_step, color=ev["color"], lw=2, ls="--", alpha=0.7)
            ax.text(event_step, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                    "★ die-off", color=ev["color"], fontsize=8.5,
                    ha="center", va="top", fontweight="bold")

        ax.set_xticks(step_x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(clabel, fontsize=10, fontweight="bold")
        ax.set_ylabel("Value (real units)", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    for ax in axes_flat[len(key_idxs):]:
        ax.axis("off")

    # RMSE summary in figure subtitle
    rmse_parts = []
    for ci, clabel in zip(key_idxs, key_labels):
        obs_n = obs_window[:, ci]
        sim_n = traj[:, ci]
        mask  = ~np.isnan(obs_n)
        if mask.sum() > 1:
            rmse = np.sqrt(np.mean((obs_n[mask] - sim_n[mask]) ** 2))
            rmse_parts.append(f"{clabel}: {rmse:.3f}")
    rmse_str = "  |  ".join(rmse_parts[:3])

    fig.suptitle(
        f"FCM Backcast — {ev['label']}\n"
        f"Seeded at {pd.Period(ev['seed_month'],'M').strftime('%b %Y')}  |  "
        f"★ = die-off event month  |  RMSE (norm.): {rmse_str}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Summary metrics table
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("ONE-STEP-AHEAD HINDCAST METRICS  (normalised [0,1] space)")
    print("=" * 60)
    print(f"  {'Concept':<22}  {'R²':>7}  {'RMSE':>7}  {'MAE':>7}  {'n':>4}")
    print("  " + "-" * 56)
    for concept, row in metrics.sort_values("R2", ascending=False).iterrows():
        r2_str   = f"{row['R2']:.3f}"   if not np.isnan(row["R2"])   else " n/a "
        rmse_str = f"{row['RMSE']:.3f}" if not np.isnan(row["RMSE"]) else " n/a "
        mae_str  = f"{row['MAE']:.3f}"  if not np.isnan(row["MAE"])  else " n/a "
        bar      = "█" * int(max(0, row["R2"]) * 20) if not np.isnan(row["R2"]) else ""
        print(f"  {concept:<22}  {r2_str:>7}  {rmse_str:>7}  {mae_str:>7}  "
              f"{int(row['n']):>4}  {bar}")
    print()
    med_r2 = metrics["R2"].median()
    print(f"  Median R² across all endogenous concepts: {med_r2:.3f}")
    print("  (R² > 0.5 = FCM has real predictive skill for that concept)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data and fitting FCM weights...")
    W, arr_norm, col_min, col_range, index, ts_clean, obs_mask = prepare_data()

    # ── 1. One-step-ahead hindcast ─────────────────────────────────────────
    print("\n=== One-step-ahead hindcast ===")
    pred, obs, eval_mask = one_step_hindcast(W, arr_norm, obs_mask)
    metrics = compute_metrics(pred, obs, eval_mask)

    metrics.to_csv(METRICS_CSV, float_format="%.4f")
    print(f"  Metrics saved → {METRICS_CSV}")
    print_metrics_table(metrics)

    print("\n[Plot 1] Scatter grid...")
    plot_hindcast_scatter(pred, obs, eval_mask, metrics)

    print("[Plot 2] Time series...")
    plot_hindcast_timeseries(pred, obs, eval_mask, index, col_min, col_range)

    # ── 2. Event backcasting ───────────────────────────────────────────────
    print("\n=== Event backcasting ===")
    out_paths = [
        VIZ_DIR / "23_fcm_backcast_sept2021.png",
        VIZ_DIR / "24_fcm_backcast_oct2022.png",
    ]
    for ev, out_path in zip(EVENTS, out_paths):
        print(f"[Backcast] {ev['short']} — seeded at {ev['seed_month']}...")
        plot_backcast(ev, W, arr_norm, col_min, col_range, index, out_path)
        if not out_path.exists():
            print(f"  [SKIP] {out_path.name} not generated")

    print("\nDone. All outputs:")
    print(f"  {METRICS_CSV}")
    for p in [VIZ_DIR / f"{n}_fcm_{s}.png"
              for n, s in [("21","hindcast_scatter"), ("22","hindcast_timeseries"),
                           ("23","backcast_sept2021"), ("24","backcast_oct2022")]]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
