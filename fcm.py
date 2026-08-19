"""
fcm.py
------
Fuzzy Cognitive Map (FCM) for Biscayne Bay water quality.

Two FCMs are built and compared:

  Physical FCM  (daily resolution, 2025-03 → 2026-07)
  ─────────────────────────────────────────────────────
  11 concepts = 3 forcing + 8 continuous sensor features
  Data source: imputed 5-min sensor timeseries, resampled to daily means

  Nutrient FCM  (monthly resolution, 2021-09 → 2026-07)
  ───────────────────────────────────────────────────────
  16 concepts = 3 forcing + 8 sensor features + 5 nutrient features
    pH, Chl-a (µg/L), Secchi depth (m), NO2+NO3 (µmol/L), DIN (µmol/L)
  Data source: 2021-2024 bi-monthly field grab samples (spatially averaged
    across all Biscayne Bay Surface samples) + 2025-2026 imputed sensor
    data aggregated to monthly means

Weight learning
───────────────
Ridge regression with 1-step lag:
    A_j(t+1) = Σ_i  W[i,j] * A_i(t)

Weights normalised to [-1, 1].  Forcing columns are exogenous (no incoming
edges).  Columns absent for a given FCM are dropped before fitting.

FCM simulation
───────────────
A(t+1)[free]    = sigmoid( W^T · A(t) )
A(t+1)[forcing] = clamped value

Scenarios:  Baseline | Heavy rain | Heat wave | Cold & dry

Run:
    python fcm.py

Outputs
───────
  analysis/fcm_weights_physical.csv   — 11×11 physical FCM weights
  analysis/fcm_weights_nutrient.csv   — 16×16 nutrient FCM weights
  visualizations/08_fcm_heatmap.png        — physical FCM heatmap
  visualizations/09_fcm_graph.png          — physical FCM causal network
  visualizations/10_fcm_scenarios.png      — physical FCM scenarios
  visualizations/11_fcm_influence_ranking.png
  visualizations/12_fcm_nutrient_heatmap.png   — nutrient FCM heatmap
  visualizations/13_fcm_nutrient_graph.png     — nutrient FCM causal network
  visualizations/14_fcm_nutrient_scenarios.png — nutrient FCM scenarios
  visualizations/15_fcm_comparison.png         — side-by-side FCM comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from preprocess import (build_dataset, ALL_FEATURES, load_historical_grab_samples,
                         build_net_water_forcing)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path("analysis")
VIZ_DIR      = Path("visualizations")
IMPUTED_DIR  = Path("imputed_output")
ANALYSIS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Physical FCM — 11 concepts  (daily, 2025-2026)
# ---------------------------------------------------------------------------
FORCING_CONCEPTS  = ["net_water", "temp_min", "temp_max"]
SENSOR_CONCEPTS   = list(ALL_FEATURES)                  # 8 features
PHYS_CONCEPTS     = FORCING_CONCEPTS + SENSOR_CONCEPTS  # 11

N_PHYS    = len(PHYS_CONCEPTS)
N_FORCING = len(FORCING_CONCEPTS)
N_SENSOR  = len(SENSOR_CONCEPTS)

# ---------------------------------------------------------------------------
# Nutrient FCM — 16 concepts  (monthly, 2021-2026)
# ---------------------------------------------------------------------------
NUTRIENT_EXTRA   = ["ph", "chl_a_ugL", "secchi_m", "no2no3_umolL", "din_umolL"]
NUTR_CONCEPTS    = PHYS_CONCEPTS + NUTRIENT_EXTRA       # 16
N_NUTR           = len(NUTR_CONCEPTS)

# ---------------------------------------------------------------------------
# Human-readable labels for both FCMs
# ---------------------------------------------------------------------------
CONCEPT_LABELS = {
    "net_water":      "Net Water (Rain−PET)",
    "rain":           "Rainfall",
    "temp_min":       "Air Temp Min",
    "temp_max":       "Air Temp Max",
    "temp_c":         "Water Temp",
    "sal_ppt":        "Salinity",
    "odo_mgL":        "ODO (mg/L)",
    "depth_m":        "Depth",
    "pressure_psia":  "Pressure",
    "odo_pct":        "ODO (%Sat)",
    "spec_cond_uScm": "Spec. Cond.",
    "turbidity_fnu":  "Turbidity",
    # Nutrient extras
    "ph":             "pH",
    "chl_a_ugL":      "Chlorophyll-a",
    "secchi_m":       "Secchi Depth",
    "no2no3_umolL":   "NO₂+NO₃",
    "din_umolL":      "DIN",
}

IMPUTED_FILES = {
    "L0":                  "raw-data-platformL0_parameters_imputed.csv",
    "L1":                  "raw-data-platformL1_parameters_imputed.csv",
    "L2":                  "raw-data-platformL2_parameters_imputed.csv",
    "L6":                  "raw-data-platformL6_parameters_imputed.csv",
    "L7":                  "raw-data-platformL7_parameters_imputed.csv",
    "biscayne_bay":        "biscayne_bay_imputed.csv",
    "consolidated_crest5": "consolidated_crest5_imputed.csv",
}

# ---------------------------------------------------------------------------
# 1. Data loading — Physical FCM (daily, 2025-2026)
# ---------------------------------------------------------------------------

def load_physical_concept_timeseries() -> pd.DataFrame:
    """
    Daily DataFrame with columns = PHYS_CONCEPTS (11).
    Sensor features are spatially averaged across all imputed nodes,
    weighting observed > imputed.
    """
    print("Loading imputed node data...")
    node_dfs: dict[str, pd.DataFrame] = {}
    for name, fname in IMPUTED_FILES.items():
        path = IMPUTED_DIR / fname
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = df.index.tz_localize(None) if df.index.tzinfo is not None else df.index
        node_dfs[name] = df

    ref_index = node_dfs[list(node_dfs.keys())[0]].index

    sensor_arrays: dict[str, np.ndarray] = {}
    for feat in SENSOR_CONCEPTS:
        obs_col = f"{feat}_observed"
        stacked_vals, stacked_obs = [], []
        for df in node_dfs.values():
            if feat in df.columns:
                stacked_vals.append(df[feat].values.astype(np.float64))
                w = df[obs_col].values.astype(np.float64) if obs_col in df.columns \
                    else np.ones(len(df), dtype=np.float64)
                stacked_obs.append(w)
        if not stacked_vals:
            sensor_arrays[feat] = np.full(len(ref_index), np.nan)
            continue
        vals    = np.stack(stacked_vals, axis=1)
        weights = np.where(np.stack(stacked_obs, axis=1) > 0.5, 1.0, 0.5)
        denom   = weights.sum(axis=1)
        denom[denom == 0] = 1.0
        sensor_arrays[feat] = (vals * weights).sum(axis=1) / denom

    sensor_df = pd.DataFrame(sensor_arrays, index=ref_index)

    print("Loading forcing signals...")
    data  = build_dataset()
    t_idx = pd.DatetimeIndex(data["time_index"])
    if t_idx.tzinfo is not None:
        t_idx = t_idx.tz_localize(None)
    _, net_w = build_net_water_forcing(t_idx)
    forcing_df = pd.DataFrame({
        "net_water": net_w,
        "temp_min":  data["temp_min"],
        "temp_max":  data["temp_max"],
    }, index=t_idx)

    combined = pd.concat([forcing_df, sensor_df], axis=1)[PHYS_CONCEPTS]
    daily    = combined.resample("1D").mean().dropna(how="all")
    daily    = daily.interpolate(method="time", limit=14).ffill().bfill()

    print(f"  Physical series: {len(daily)} days × {N_PHYS} concepts")
    print(f"  Date range: {daily.index[0].date()} → {daily.index[-1].date()}")
    return daily


# ---------------------------------------------------------------------------
# 1b. Data loading — Nutrient FCM (monthly, 2021-2026)
# ---------------------------------------------------------------------------

def load_nutrient_concept_timeseries() -> pd.DataFrame:
    """
    Monthly DataFrame with columns = NUTR_CONCEPTS (16).

    2021-09 → 2024-12: from grab-sample CSV (bi-monthly field surveys).
      Sensor features from BB surface samples (Biscayne Bay sites only).
      Nutrient features (pH, Chl-a, Secchi, NO2+NO3, DIN) from same source.
      Forcing signals: loaded from rainfall CSVs if 2021-2024 records exist
      (fetched via fetch_historical_weather.py); NaN otherwise.

    2025-03 → 2026-07: from imputed continuous sensor data (resampled
      to monthly means) + weather forcing.
      Nutrient columns: NaN (continuous sensors don't measure nutrients).
    """
    # --- 2021-2024 grab samples ---
    print("Loading 2021-2024 grab samples (Biscayne Bay Surface)...")
    grab = load_historical_grab_samples(
        site_types=["Biscayne Bay"],
        sample_type="Surface",
    )
    # Grab sensor overlap + nutrient extras
    grab_sensor_cols  = [c for c in SENSOR_CONCEPTS if c in grab.columns]
    grab_nutr_cols    = [c for c in NUTRIENT_EXTRA  if c in grab.columns]
    grab_keep         = grab_sensor_cols + grab_nutr_cols
    grab_monthly      = grab[grab_keep].copy()
    # Build monthly weather forcing for the grab-sample period (uses
    # historical rainfall CSVs if available, otherwise stays NaN)
    raw_idx = grab_monthly.index
    if hasattr(raw_idx, "to_timestamp"):
        grab_idx_dt = pd.DatetimeIndex(raw_idx.to_timestamp()).tz_localize("UTC")
    else:
        grab_idx_dt = pd.DatetimeIndex(raw_idx).tz_localize("UTC") \
                      if raw_idx.tzinfo is None else pd.DatetimeIndex(raw_idx)
    try:
        _, net_w_grab = build_net_water_forcing(grab_idx_dt)
        _, tmn_grab, tmx_grab = __import__("preprocess").build_forcing_for_index(grab_idx_dt)
        grab_monthly["net_water"] = net_w_grab
        grab_monthly["temp_min"]  = tmn_grab
        grab_monthly["temp_max"]  = tmx_grab
    except Exception:
        for fc in FORCING_CONCEPTS:
            grab_monthly[fc] = np.nan
    # Reindex to full concept list, filling absent columns with NaN
    grab_monthly = grab_monthly.reindex(columns=NUTR_CONCEPTS)

    # --- 2025-2026 continuous (monthly-resampled) ---
    print("Loading 2025-2026 continuous data (monthly)...")
    phys_daily = load_physical_concept_timeseries()
    phys_monthly = phys_daily.resample("MS").mean()
    # Add NaN nutrient columns for 2025-2026
    for nc in NUTRIENT_EXTRA:
        phys_monthly[nc] = np.nan
    phys_monthly = phys_monthly[NUTR_CONCEPTS]

    # --- Combine (grab first, then continuous; overlapping months: prefer continuous) ---
    combined = pd.concat([grab_monthly, phys_monthly])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined = combined.dropna(how="all")

    # Light interpolation within each column (limit=2 months to avoid long gaps)
    combined = combined.interpolate(method="time", limit=2).ffill(limit=1).bfill(limit=1)

    print(f"  Nutrient series: {len(combined)} months × {N_NUTR} concepts")
    print(f"  Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"  Nutrient NaN rates:")
    for nc in NUTRIENT_EXTRA:
        pct = combined[nc].isna().mean() * 100
        print(f"    {nc:20s}: {pct:.0f}% NaN")
    return combined


# ---------------------------------------------------------------------------
# 2. Ridge regression (no sklearn dependency)
# ---------------------------------------------------------------------------

def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge: w = (X'X + αI)^{-1} X'y  (no intercept)."""
    XtX = X.T @ X
    return np.linalg.solve(XtX + alpha * np.eye(X.shape[1]), X.T @ y)


# ---------------------------------------------------------------------------
# 3a. Optimal-lag selection via held-out validation MSE
# ---------------------------------------------------------------------------

def select_optimal_lag(
    ts:             pd.DataFrame,
    concept_list:   list[str],
    n_forcing:      int,
    candidate_lags: list[int],
    alpha:          float = 2.0,
    val_frac:       float = 0.20,
    tag:            str   = "",
) -> int:
    """
    For each candidate lag fit Ridge on the first (1-val_frac) of the series,
    evaluate MSE on the held-out tail, return the lag with lowest validation MSE.
    """
    C         = len(concept_list)
    endo_cols = concept_list[n_forcing:]
    arr_full  = ts[concept_list].dropna(subset=endo_cols, how="all")
    arr       = arr_full.values.astype(np.float64)

    col_min   = np.nanmin(arr,  axis=0)
    col_range = np.nanmax(arr,  axis=0) - col_min
    col_range[col_range == 0] = 1.0
    arr_norm  = (arr - col_min) / col_range

    col_means = np.nanmean(arr_norm, axis=0)
    col_means[np.isnan(col_means)] = 0.5
    nan_mask   = np.isnan(arr_norm)
    arr_filled = arr_norm.copy()
    arr_filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    T         = len(arr_filled)
    val_start = max(int(T * (1 - val_frac)), min(candidate_lags) + 2)
    endo_idx  = list(range(n_forcing, C))

    scores: dict[int, float] = {}
    for lag in candidate_lags:
        if val_start - lag < 2 or T - val_start < 1:
            continue
        X_tr = arr_filled[:val_start - lag]
        Y_tr = arr_filled[lag:val_start]
        X_vl = arr_filled[val_start - lag: T - lag]
        Y_vl = arr_filled[val_start:]

        W = np.zeros((C, C))
        for j in range(n_forcing, C):
            if nan_mask[lag:val_start, j].mean() > 0.60:
                continue
            W[:, j] = _ridge(X_tr, Y_tr[:, j], alpha=alpha)
        W[:, :n_forcing] = 0.0

        Y_pred   = X_vl @ W
        obs_mask = ~nan_mask[val_start:, :][:, endo_idx]
        if obs_mask.sum() == 0:
            continue
        scores[lag] = float(np.mean(
            (Y_pred[:, endo_idx][obs_mask] - Y_vl[:, endo_idx][obs_mask]) ** 2
        ))

    if not scores:
        best = candidate_lags[0]
    else:
        best = min(scores, key=scores.get)

    score_str = "  ".join(f"lag={k}: {v:.4f}" for k, v in sorted(scores.items()))
    print(f"  [{tag}] Lag selection — {score_str}")
    print(f"  [{tag}] → best lag = {best}")
    return best


# ---------------------------------------------------------------------------
# 3. FCM weight learning  (concept-list agnostic)
# ---------------------------------------------------------------------------

def learn_fcm_weights(
    ts:           pd.DataFrame,
    concept_list: list[str],
    n_forcing:    int,
    alpha:        float = 2.0,
    lag:          int   = 1,
    tag:          str   = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      W          — (C, C) float; W[i,j] = influence of concept i on j
      arr_norm   — (T, C) min-max normalised values  [0, 1]
      col_min    — (C,) per-concept minimum
      col_range  — (C,) per-concept range

    Rows/columns with >50% NaN in the time series are excluded from the
    regression (nutrient columns that only exist in one period).
    """
    C      = len(concept_list)
    labels = [CONCEPT_LABELS.get(c, c) for c in concept_list]

    # Drop rows where ALL endogenous concepts are NaN
    endo_cols = concept_list[n_forcing:]
    arr_full  = ts[concept_list].copy()
    arr_full  = arr_full.dropna(subset=endo_cols, how="all")

    arr = arr_full.values.astype(np.float64)

    # Per-concept min-max normalisation to [0, 1]  (NaN-safe)
    col_min   = np.nanmin(arr,  axis=0)
    col_max   = np.nanmax(arr,  axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    arr_norm  = (arr - col_min) / col_range

    # Replace NaN with per-column mean (so regression still works for
    # columns that exist in only part of the time window)
    col_means = np.nanmean(arr_norm, axis=0)
    col_means[np.isnan(col_means)] = 0.5
    nan_mask  = np.isnan(arr_norm)
    arr_filled = arr_norm.copy()
    arr_filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    X_lag = arr_filled[:-lag]   # (T-lag, C)
    Y_lag = arr_filled[lag:]    # (T-lag, C)

    W = np.zeros((C, C), dtype=np.float64)
    for j in range(n_forcing, C):
        # Skip columns that were mostly filled-in (> 60% were NaN originally)
        nan_frac_j = nan_mask[lag:, j].mean()
        if nan_frac_j > 0.60:
            continue
        W[:, j] = _ridge(X_lag, Y_lag[:, j], alpha=alpha)

    # Forcing concepts are exogenous — zero their incoming edges
    W[:, :n_forcing] = 0.0

    # Normalise to [-1, 1]
    abs_max = np.abs(W).max()
    if abs_max > 0:
        W = W / abs_max

    header = f"\n=== FCM weights ({tag}) — lag={lag} ===\n"
    print(header + f"  {'SOURCE':<22} → {'TARGET':<22}  WEIGHT")
    print("  " + "-" * 56)
    edges = sorted(
        [(abs(W[i, j]), W[i, j], labels[i], labels[j])
         for i in range(C) for j in range(C)
         if i != j and abs(W[i, j]) > 0.05],
        reverse=True,
    )
    for _, w, src, tgt in edges[:18]:
        bar  = "█" * int(abs(w) * 20)
        sign = "+" if w > 0 else "-"
        print(f"  {src:<22} → {tgt:<22}  {sign}{abs(w):.3f}  {bar}")

    return W, arr_norm, col_min, col_range


# ---------------------------------------------------------------------------
# 4. FCM simulation  (concept-list agnostic)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fcm_simulate(
    W:               np.ndarray,
    A0:              np.ndarray,
    clamped_indices: list[int],
    clamped_values:  np.ndarray,
    n_steps:         int = 40,
) -> np.ndarray:
    """
    A(t+1)[free]    = sigmoid( W^T · A(t) )
    A(t+1)[clamped] = fixed value
    Returns (n_steps+1, C) trajectory.
    """
    A    = A0.copy()
    traj = [A.copy()]
    for _ in range(n_steps):
        A_new = _sigmoid(W.T @ A)
        A_new[clamped_indices] = clamped_values
        A     = A_new
        traj.append(A.copy())
    return np.stack(traj)


# ---------------------------------------------------------------------------
# 5a. Visualisation — heatmap  (generic)
# ---------------------------------------------------------------------------

def plot_heatmap(
    W:            np.ndarray,
    concept_list: list[str],
    n_forcing:    int,
    out_path:     Path,
    title:        str = "FCM — Causal Influence Matrix",
) -> None:
    C      = len(concept_list)
    labels = [CONCEPT_LABELS.get(c, c) for c in concept_list]
    n_sensor_local = C - n_forcing

    display = W.copy()
    np.fill_diagonal(display, np.nan)
    vmax = max(0.01, float(np.nanmax(np.abs(display))))

    fig, ax = plt.subplots(figsize=(max(9, C * 0.75), max(7, C * 0.65)))
    im = ax.imshow(display.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Source concept  (what causes)", fontsize=10)
    ax.set_ylabel("Target concept  (what is affected)", fontsize=10)
    ax.set_title(title + "\nW[i→j] > 0 : i promotes j  |  W[i→j] < 0 : i suppresses j",
                 fontsize=11, pad=10)

    div = n_forcing - 0.5
    ax.axvline(div, color="k", lw=1.5, ls="--", alpha=0.5)
    ax.axhline(div, color="k", lw=1.5, ls="--", alpha=0.5)

    # Block separators for nutrient FCM
    if C > N_PHYS:
        div2 = N_PHYS - 0.5
        ax.axvline(div2, color="gray", lw=1, ls=":", alpha=0.6)
        ax.axhline(div2, color="gray", lw=1, ls=":", alpha=0.6)

    for i in range(C):
        for j in range(C):
            v = display[i, j]
            if np.isnan(v) or abs(v) < 0.07:
                continue
            txt_col = "white" if abs(v) > 0.5 else "black"
            ax.text(i, j, f"{v:+.2f}", ha="center", va="center",
                    fontsize=5.5, color=txt_col)

    plt.colorbar(im, ax=ax, label="Causal weight  W[src → tgt]",
                 fraction=0.04, pad=0.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# 5b. Visualisation — circular causal graph  (generic)
# ---------------------------------------------------------------------------

def plot_causal_graph(
    W:            np.ndarray,
    concept_list: list[str],
    n_forcing:    int,
    out_path:     Path,
    threshold:    float = 0.07,
    title:        str   = "FCM — Causal Network",
) -> None:
    C        = len(concept_list)
    labels   = [CONCEPT_LABELS.get(c, c) for c in concept_list]
    n_sensor_local = C - n_forcing
    n_extra  = C - N_PHYS if C > N_PHYS else 0

    # Colour scheme: orange=forcing, blue=sensor, green=nutrient
    node_colors = (
        ["#e07b00"] * n_forcing
        + ["#2055c8"] * n_sensor_local
    )
    if n_extra > 0:
        node_colors = (
            ["#e07b00"] * n_forcing
            + ["#2055c8"] * N_SENSOR
            + ["#27ae60"] * n_extra
        )

    angles = np.linspace(0, 2 * np.pi, C, endpoint=False) - np.pi / 2
    xs     = np.cos(angles)
    ys     = np.sin(angles)

    out_strength = np.abs(W).sum(axis=1)
    node_sizes   = 800 + out_strength * 2200

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)

    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) < threshold:
                continue
            color = "#c0392b" if w > 0 else "#2980b9"
            lw    = 0.8 + abs(w) * 3.5
            alpha = float(np.clip(0.35 + abs(w) * 0.55, 0, 0.95))
            ax.annotate("",
                xy=(xs[j], ys[j]), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha, shrinkA=13, shrinkB=13,
                                connectionstyle="arc3,rad=0.18"))
            mx = (xs[i] + xs[j]) / 2 * 1.12
            my = (ys[i] + ys[j]) / 2 * 1.12
            ax.text(mx, my, f"{w:+.2f}", fontsize=5.5, ha="center", va="center",
                    color=color, alpha=float(np.clip(alpha + 0.1, 0, 1)))

    for i in range(C):
        ax.scatter(xs[i], ys[i], s=node_sizes[i], c=node_colors[i],
                   zorder=5, edgecolors="white", linewidths=2.5)

    for i in range(C):
        ang = angles[i]
        off = 1.35
        ha  = "left"  if np.cos(ang) > 0.1 else ("right" if np.cos(ang) < -0.1 else "center")
        va  = "bottom" if np.sin(ang) > 0.1 else ("top"  if np.sin(ang) < -0.1 else "center")
        ax.text(off * np.cos(ang), off * np.sin(ang), labels[i],
                ha=ha, va=va, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    patches = [
        mpatches.Patch(color="#e07b00", label="Exogenous forcing"),
        mpatches.Patch(color="#2055c8", label="Sensor / water quality"),
    ]
    if n_extra > 0:
        patches.append(mpatches.Patch(color="#27ae60", label="Nutrients / ecology"))
    patches += [
        mpatches.Patch(color="#c0392b", label="Positive influence (+)"),
        mpatches.Patch(color="#2980b9", label="Negative influence (−)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=9,
              framealpha=0.9, edgecolor="#ccc")
    ax.set_title(f"{title}\nEdges |W| ≥ {threshold:.2f}  |  Node size ∝ total outgoing",
                 fontsize=12, pad=14)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# 5c. Visualisation — scenario simulations  (generic)
# ---------------------------------------------------------------------------

def plot_scenarios(
    W:            np.ndarray,
    concept_list: list[str],
    n_forcing:    int,
    baseline_A:   np.ndarray,
    out_path:     Path,
    n_steps:      int = 40,
    step_label:   str = "days",
    title_suffix: str = "",
) -> None:
    mean_forcing = baseline_A[:n_forcing].copy()
    forcing_idx  = list(range(n_forcing))

    scenarios = [
        ("Baseline (mean)",           "#555555", mean_forcing.copy()),
        ("Heavy rain  (rain=0.9)",    "#1f77b4", np.array([0.9, mean_forcing[1], mean_forcing[2]])),
        ("Heat wave  (temp_max=0.9)", "#d62728", np.array([0.05, mean_forcing[1], 0.9])),
        ("Cold & dry  (temp_max=0.1)","#2ca02c", np.array([0.05, mean_forcing[1], 0.1])),
    ]

    # 4 response concepts: water temp, salinity, ODO, + most interesting available
    candidates = ["temp_c", "sal_ppt", "odo_mgL", "turbidity_fnu",
                  "chl_a_ugL", "ph", "din_umolL"]
    plot_keys  = [c for c in candidates if c in concept_list][:4]
    plot_idxs  = [concept_list.index(c) for c in plot_keys]
    plot_labels= [CONCEPT_LABELS.get(c, c) for c in plot_keys]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    steps = np.arange(n_steps + 1)

    for ax_i, (si, slabel) in enumerate(zip(plot_idxs, plot_labels)):
        ax = axes[ax_i]
        for sc_label, color, forced_vals in scenarios:
            traj = fcm_simulate(W, baseline_A.copy(), forcing_idx, forced_vals, n_steps)
            ls   = "--" if sc_label.startswith("Baseline") else "-"
            ax.plot(steps, traj[:, si], color=color, lw=2, ls=ls,
                    label=sc_label, alpha=0.9)
        ax.set_title(slabel, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Simulation step ({step_label})", fontsize=9)
        ax.set_ylabel("Normalised activation  [0–1]", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"FCM Scenario Simulations{title_suffix}\n"
                 "Forcing held fixed; sensor concepts evolve under FCM dynamics",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# 5d. Visualisation — influence ranking  (generic)
# ---------------------------------------------------------------------------

def plot_influence_ranking(
    W:            np.ndarray,
    concept_list: list[str],
    n_forcing:    int,
    out_path:     Path,
    title:        str = "FCM Influence Rankings",
) -> None:
    C          = len(concept_list)
    labels     = [CONCEPT_LABELS.get(c, c) for c in concept_list]
    n_extra    = C - N_PHYS if C > N_PHYS else 0
    node_colors = (["#e07b00"] * n_forcing
                   + ["#2055c8"] * N_SENSOR
                   + ["#27ae60"] * n_extra) if C > N_PHYS else \
                  (["#e07b00"] * n_forcing + ["#2055c8"] * (C - n_forcing))

    pos_out = np.clip(W, 0, None).sum(axis=1)
    neg_out = np.clip(W, None, 0).sum(axis=1)
    in_deg  = np.abs(W).sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, C * 0.38)))

    ax = axes[0]
    y  = np.arange(C)
    ax.barh(y, pos_out, color="#c0392b", alpha=0.8, label="Positive outgoing")
    ax.barh(y, neg_out, color="#2980b9", alpha=0.8, label="Negative outgoing")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Sum of outgoing weights", fontsize=10)
    ax.set_title("Outgoing influence\n(what drives other concepts)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    for i, c in enumerate(node_colors):
        ax.get_yticklabels()[i].set_color(c)
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax = axes[1]
    sorted_idx = np.argsort(in_deg)[::-1]
    ax.bar(range(C), in_deg[sorted_idx],
           color=[node_colors[i] for i in sorted_idx], alpha=0.85)
    ax.set_xticks(range(C))
    ax.set_xticklabels([labels[i] for i in sorted_idx],
                       rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Sum of incoming |W|", fontsize=10)
    ax.set_title("Incoming influence\n(what is most driven by others)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for i, idx in enumerate(sorted_idx):
        ax.get_xticklabels()[i].set_color(node_colors[idx])
        ax.get_xticklabels()[i].set_fontweight("bold")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# 5e. Visualisation — side-by-side FCM comparison  (shared sensor concepts)
# ---------------------------------------------------------------------------

def plot_fcm_comparison(
    W_phys: np.ndarray,
    W_nutr: np.ndarray,
) -> None:
    """
    Show the 11×11 shared-concept sub-matrices side by side so changes
    in causal strength between the physical FCM and the nutrient FCM
    (where months with nutrient data shift the regression) are visible.
    """
    shared = PHYS_CONCEPTS
    C      = N_PHYS
    labels = [CONCEPT_LABELS.get(c, c) for c in shared]

    # Extract the shared sub-matrix from the nutrient FCM
    W_nutr_sub = W_nutr[:C, :C].copy()

    vmax = max(0.01, float(np.nanmax(np.abs([W_phys, W_nutr_sub]))))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    titles    = ["Physical FCM (daily, 2025–2026)", "Nutrient FCM (monthly, 2021–2026)"]
    mats      = [W_phys, W_nutr_sub]

    for ax, mat, ttl in zip(axes, mats, titles):
        disp = mat.copy()
        np.fill_diagonal(disp, np.nan)
        im = ax.imshow(disp.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        div = N_FORCING - 0.5
        ax.axvline(div, color="k", lw=1.5, ls="--", alpha=0.5)
        ax.axhline(div, color="k", lw=1.5, ls="--", alpha=0.5)
        for i in range(C):
            for j in range(C):
                v = disp[i, j]
                if np.isnan(v) or abs(v) < 0.07:
                    continue
                ax.text(i, j, f"{v:+.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Causal weight W[src→tgt]")

    fig.suptitle("FCM Comparison — Shared Physical Concepts\n"
                 "Left: daily resolution  |  Right: extended monthly (2021–2026) with nutrients",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = VIZ_DIR / "15_fcm_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_causal_summary(W, concept_list, n_forcing, tag=""):
    C      = len(concept_list)
    labels = [CONCEPT_LABELS.get(c, c) for c in concept_list]
    cats   = [
        ("Forcing → Sensor/Nutrient", range(n_forcing), range(n_forcing, C)),
        ("Sensor  → Sensor",          range(n_forcing, N_PHYS), range(n_forcing, N_PHYS)),
    ]
    if C > N_PHYS:
        cats += [
            ("Sensor  → Nutrient",  range(n_forcing, N_PHYS), range(N_PHYS, C)),
            ("Nutrient → Sensor",   range(N_PHYS, C),          range(n_forcing, N_PHYS)),
            ("Nutrient → Nutrient", range(N_PHYS, C),          range(N_PHYS, C)),
        ]
    print(f"\n{'='*60}\nCAUSAL SUMMARY  [{tag}]\n{'='*60}")
    for cat_name, src_r, tgt_r in cats:
        edges = [(W[i, j], labels[i], labels[j])
                 for i in src_r for j in tgt_r
                 if i != j and abs(W[i, j]) > 0.05]
        if not edges:
            continue
        edges.sort(key=lambda x: abs(x[0]), reverse=True)
        print(f"\n{cat_name}:")
        for w, src, tgt in edges[:6]:
            bar    = "█" * int(abs(w) * 20)
            sign   = "+" if w > 0 else "−"
            effect = "promotes" if w > 0 else "suppresses"
            print(f"  {src:<22} {effect:<11} {tgt:<22} ({sign}{abs(w):.3f})  {bar}")


def main() -> None:
    # =====================================================================
    # A. Physical FCM  (daily, 2025-2026, 11 concepts)
    # =====================================================================
    print("\n" + "="*60)
    print("PHYSICAL FCM  (daily, 2025–2026, 11 concepts)")
    print("="*60)

    phys_daily = load_physical_concept_timeseries()
    best_lag_phys = select_optimal_lag(
        phys_daily, PHYS_CONCEPTS, N_FORCING,
        candidate_lags=[1, 2, 3, 7], alpha=2.0, tag="Physical",
    )
    W_phys, arr_phys, _, _ = learn_fcm_weights(
        phys_daily, PHYS_CONCEPTS, N_FORCING, alpha=2.0, lag=best_lag_phys,
        tag="Physical daily",
    )

    labels_phys = [CONCEPT_LABELS.get(c, c) for c in PHYS_CONCEPTS]
    pd.DataFrame(W_phys, index=labels_phys, columns=labels_phys).to_csv(
        ANALYSIS_DIR / "fcm_weights_physical.csv", float_format="%.4f")
    print(f"  Saved → analysis/fcm_weights_physical.csv")

    baseline_phys = arr_phys.mean(axis=0)
    _print_causal_summary(W_phys, PHYS_CONCEPTS, N_FORCING, "Physical")

    traj_p = fcm_simulate(W_phys, baseline_phys.copy(), list(range(N_FORCING)),
                          baseline_phys[:N_FORCING], n_steps=60)
    delta_p = np.abs(traj_p[-1] - traj_p[-2]).max()
    print(f"\nConvergence: max |ΔA| at step 60 = {delta_p:.6f}",
          "✓" if delta_p < 0.001 else "(increase n_steps)")

    print("\n[Physical FCM] Generating visualisations...")
    plot_heatmap(W_phys, PHYS_CONCEPTS, N_FORCING,
                 VIZ_DIR / "08_fcm_heatmap.png",
                 title="Physical FCM — Causal Influence (daily 2025–2026)")
    plot_causal_graph(W_phys, PHYS_CONCEPTS, N_FORCING,
                      VIZ_DIR / "09_fcm_graph.png",
                      title="Physical FCM — Biscayne Bay Water Quality")
    plot_scenarios(W_phys, PHYS_CONCEPTS, N_FORCING, baseline_phys,
                   VIZ_DIR / "10_fcm_scenarios.png",
                   step_label="days", title_suffix=" — Physical FCM (2025–2026)")
    plot_influence_ranking(W_phys, PHYS_CONCEPTS, N_FORCING,
                           VIZ_DIR / "11_fcm_influence_ranking.png",
                           title="Physical FCM Influence Rankings — Biscayne Bay")

    # =====================================================================
    # B. Nutrient FCM  (monthly, 2021-2026, 16 concepts)
    # =====================================================================
    print("\n" + "="*60)
    print("NUTRIENT FCM  (monthly, 2021–2026, 16 concepts)")
    print("="*60)

    nutr_monthly = load_nutrient_concept_timeseries()
    best_lag_nutr = select_optimal_lag(
        nutr_monthly, NUTR_CONCEPTS, N_FORCING,
        candidate_lags=[1, 2, 3], alpha=1.5, tag="Nutrient",
    )
    W_nutr, arr_nutr, _, _ = learn_fcm_weights(
        nutr_monthly, NUTR_CONCEPTS, N_FORCING, alpha=1.5, lag=best_lag_nutr,
        tag="Nutrient monthly",
    )

    labels_nutr = [CONCEPT_LABELS.get(c, c) for c in NUTR_CONCEPTS]
    pd.DataFrame(W_nutr, index=labels_nutr, columns=labels_nutr).to_csv(
        ANALYSIS_DIR / "fcm_weights_nutrient.csv", float_format="%.4f")
    print(f"  Saved → analysis/fcm_weights_nutrient.csv")

    baseline_nutr = arr_nutr.mean(axis=0)
    _print_causal_summary(W_nutr, NUTR_CONCEPTS, N_FORCING, "Nutrient")

    traj_n = fcm_simulate(W_nutr, baseline_nutr.copy(), list(range(N_FORCING)),
                          baseline_nutr[:N_FORCING], n_steps=60)
    delta_n = np.abs(traj_n[-1] - traj_n[-2]).max()
    print(f"\nConvergence: max |ΔA| at step 60 = {delta_n:.6f}",
          "✓" if delta_n < 0.001 else "(increase n_steps)")

    print("\n[Nutrient FCM] Generating visualisations...")
    plot_heatmap(W_nutr, NUTR_CONCEPTS, N_FORCING,
                 VIZ_DIR / "12_fcm_nutrient_heatmap.png",
                 title="Nutrient FCM — Causal Influence (monthly 2021–2026)")
    plot_causal_graph(W_nutr, NUTR_CONCEPTS, N_FORCING,
                      VIZ_DIR / "13_fcm_nutrient_graph.png",
                      title="Nutrient FCM — Biscayne Bay 2021–2026")
    plot_scenarios(W_nutr, NUTR_CONCEPTS, N_FORCING, baseline_nutr,
                   VIZ_DIR / "14_fcm_nutrient_scenarios.png",
                   step_label="months", title_suffix=" — Nutrient FCM (2021–2026)")
    plot_influence_ranking(W_nutr, NUTR_CONCEPTS, N_FORCING,
                           VIZ_DIR / "15_fcm_nutrient_ranking.png",
                           title="Nutrient FCM Influence Rankings — 2021–2026")

    # =====================================================================
    # C. Side-by-side comparison of shared concepts
    # =====================================================================
    print("\n[Comparison] Generating side-by-side comparison...")
    plot_fcm_comparison(W_phys, W_nutr)

    print("\nDone.  Outputs in analysis/ and visualizations/")
    print("  Physical FCM : analysis/fcm_weights_physical.csv")
    print("  Nutrient FCM : analysis/fcm_weights_nutrient.csv")
    print("  Figures      : visualizations/08_fcm_*.png through 15_fcm_*.png")


if __name__ == "__main__":
    main()
