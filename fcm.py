"""
fcm.py
------
Fuzzy Cognitive Map (FCM) for Biscayne Bay water quality.

Learns directed causal weights between 11 concepts:

  Forcing  (exogenous — no incoming edges):
    rain, temp_min, temp_max

  Sensor features (endogenous):
    temp_c, sal_ppt, odo_mgL, depth_m,
    pressure_psia, odo_pct, spec_cond_uScm, turbidity_fnu

Weight learning
---------------
Imputed time series is resampled to daily and min-max normalised to [0,1].
For each sensor concept j, a lagged Ridge regression is fitted:

    A_j(t+1) = Σ_i  W[i,j] * A_i(t)

using all 11 concepts as predictors and 1-day lag.  This gives a directed
weight matrix: W[i,j] > 0 means "i promotes j tomorrow",
                         W[i,j] < 0 means "i suppresses j tomorrow".

Weights are then normalised to [-1, 1].  Forcing columns (rain, temp_min,
temp_max) are zeroed out as targets (they are exogenous inputs only).

FCM simulation
--------------
A(t+1) = sigmoid( W^T · A(t) )    for free (sensor) nodes
A(t+1)[forcing] = clamped value    for exogenous forcing nodes

Scenarios compare system equilibrium under:
  1. Baseline           — all concepts at their historical mean
  2. Heavy rain event   — rain = 0.9
  3. Heat wave          — temp_max = 0.9
  4. Cold & dry         — temp_max = 0.1, rain = 0.05

Run:
    python fcm.py

Outputs
-------
  analysis/fcm_weights.csv            — 11×11 causal weight matrix
  visualizations/08_fcm_heatmap.png   — colour-coded weight heatmap
  visualizations/09_fcm_graph.png     — circular causal network diagram
  visualizations/10_fcm_scenarios.png — scenario simulation trajectories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

from preprocess import build_dataset, ALL_FEATURES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path("analysis")
VIZ_DIR      = Path("visualizations")
IMPUTED_DIR  = Path("imputed_output")
ANALYSIS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Concept definitions
# ---------------------------------------------------------------------------
FORCING_CONCEPTS = ["rain", "temp_min", "temp_max"]
SENSOR_CONCEPTS  = list(ALL_FEATURES)      # 8 features
ALL_CONCEPTS     = FORCING_CONCEPTS + SENSOR_CONCEPTS   # 11 total

N_CONCEPTS = len(ALL_CONCEPTS)
N_FORCING  = len(FORCING_CONCEPTS)
N_SENSOR   = len(SENSOR_CONCEPTS)

CONCEPT_LABELS = {
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
}

IMPUTED_FILES = {
    "L0":           "raw-data-platformL0_parameters_imputed.csv",
    "L1":           "raw-data-platformL1_parameters_imputed.csv",
    "L2":           "raw-data-platformL2_parameters_imputed.csv",
    "L6":           "raw-data-platformL6_parameters_imputed.csv",
    "L7":           "raw-data-platformL7_parameters_imputed.csv",
    "biscayne_bay": "biscayne_bay_imputed.csv",
}

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_concept_timeseries() -> pd.DataFrame:
    """
    Returns a daily DataFrame with columns = ALL_CONCEPTS.

    Sensor features are spatially averaged across all 6 nodes, weighting
    original observed values (w=1.0) over imputed values (w=0.5) so that
    stations with real data dominate when available.

    Forcing signals (rain, temp_min, temp_max) come directly from
    preprocess.build_dataset().
    """
    print("Loading imputed node data...")
    node_dfs: dict[str, pd.DataFrame] = {}
    for name, fname in IMPUTED_FILES.items():
        path = IMPUTED_DIR / fname
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Drop timezone so we can concat with forcing later
        df.index = df.index.tz_localize(None) if df.index.tzinfo is not None else df.index
        node_dfs[name] = df

    ref_index = node_dfs[list(node_dfs.keys())[0]].index

    # Weighted spatial average per sensor feature
    sensor_arrays: dict[str, np.ndarray] = {}
    for feat in SENSOR_CONCEPTS:
        obs_col = f"{feat}_observed"
        stacked_vals, stacked_obs = [], []
        for df in node_dfs.values():
            if feat in df.columns:
                stacked_vals.append(df[feat].values.astype(np.float64))
                if obs_col in df.columns:
                    stacked_obs.append(df[obs_col].values.astype(np.float64))
                else:
                    stacked_obs.append(np.ones(len(df), dtype=np.float64))
        if not stacked_vals:
            sensor_arrays[feat] = np.full(len(ref_index), np.nan)
            continue
        vals    = np.stack(stacked_vals, axis=1)   # (T, n_nodes)
        weights = np.where(np.stack(stacked_obs, axis=1) > 0.5, 1.0, 0.5)
        denom   = weights.sum(axis=1)
        denom[denom == 0] = 1.0
        sensor_arrays[feat] = (vals * weights).sum(axis=1) / denom

    sensor_df = pd.DataFrame(sensor_arrays, index=ref_index)

    # Forcing signals
    print("Loading forcing signals from preprocess...")
    data = build_dataset()
    t_idx = pd.DatetimeIndex(data["time_index"])
    if t_idx.tzinfo is not None:
        t_idx = t_idx.tz_localize(None)
    forcing_df = pd.DataFrame({
        "rain":     data["rain"],
        "temp_min": data["temp_min"],
        "temp_max": data["temp_max"],
    }, index=t_idx)

    # Concatenate and resample to daily mean
    combined = pd.concat([forcing_df, sensor_df], axis=1)[ALL_CONCEPTS]
    daily = combined.resample("1D").mean()
    daily = daily.dropna(how="all")

    # Fill remaining gaps with time-interpolation + edge fill
    daily = daily.interpolate(method="time", limit=14).ffill().bfill()

    print(f"  Daily concept series: {len(daily)} days × {N_CONCEPTS} concepts")
    print(f"  Date range: {daily.index[0].date()} → {daily.index[-1].date()}")
    return daily


# ---------------------------------------------------------------------------
# 2. Ridge regression (no sklearn dependency)
# ---------------------------------------------------------------------------

def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge: w = (X'X + αI)^{-1} X'y  (no intercept)."""
    XtX = X.T @ X
    return np.linalg.solve(XtX + alpha * np.eye(X.shape[1]), X.T @ y)


# ---------------------------------------------------------------------------
# 3. FCM weight learning
# ---------------------------------------------------------------------------

def learn_fcm_weights(
    daily: pd.DataFrame,
    alpha: float = 2.0,
    lag: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      W          — (N_CONCEPTS, N_CONCEPTS) float; W[i,j] = i→j
      arr_norm   — (T, N_CONCEPTS) min-max normalised daily values
      col_min    — (N_CONCEPTS,) per-concept minimum (for inverse transform)
      col_range  — (N_CONCEPTS,) per-concept range
    """
    print("\nLearning FCM weights via lagged Ridge regression (lag={} day)...".format(lag))

    arr = daily.values.astype(np.float64)

    # Per-concept min-max normalisation → [0, 1]
    col_min   = arr.min(axis=0)
    col_max   = arr.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    arr_norm  = (arr - col_min) / col_range

    X = arr_norm[:-lag]   # predictors at time t        (T-lag, 11)
    Y = arr_norm[lag:]    # targets   at time t+lag     (T-lag, 11)

    W = np.zeros((N_CONCEPTS, N_CONCEPTS), dtype=np.float64)

    # Fit a separate regression for each sensor concept
    for j in range(N_FORCING, N_CONCEPTS):
        w_j = _ridge(X, Y[:, j], alpha=alpha)   # (11,)
        W[:, j] = w_j

    # Forcing concepts are exogenous — zero out their columns (no incoming edges)
    W[:, :N_FORCING] = 0.0

    # Normalise entire weight matrix to [-1, 1]
    abs_max = np.abs(W).max()
    if abs_max > 0:
        W = W / abs_max

    # Print summary
    labels = [CONCEPT_LABELS[c] for c in ALL_CONCEPTS]
    edges = [
        (abs(W[i, j]), W[i, j], labels[i], labels[j])
        for i in range(N_CONCEPTS)
        for j in range(N_CONCEPTS)
        if i != j and abs(W[i, j]) > 0.05
    ]
    edges.sort(reverse=True)

    print(f"\n  {'SOURCE':<22} → {'TARGET':<22}  WEIGHT")
    print("  " + "-" * 56)
    for _, w, src, tgt in edges[:15]:
        bar = "█" * int(abs(w) * 20)
        sign = "+" if w > 0 else "-"
        print(f"  {src:<22} → {tgt:<22}  {sign}{abs(w):.3f}  {bar}")

    return W, arr_norm, col_min, col_range


# ---------------------------------------------------------------------------
# 4. FCM simulation
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
    Iterates the FCM update rule:
        A(t+1)[free]    = sigmoid( W^T · A(t) )
        A(t+1)[clamped] = fixed value

    Returns trajectory of shape (n_steps+1, N_CONCEPTS).
    """
    A = A0.copy()
    traj = [A.copy()]
    for _ in range(n_steps):
        A_new = _sigmoid(W.T @ A)
        A_new[clamped_indices] = clamped_values
        A = A_new
        traj.append(A.copy())
    return np.stack(traj)   # (n_steps+1, N_CONCEPTS)


# ---------------------------------------------------------------------------
# 5a. Visualisation — heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(W: np.ndarray) -> None:
    labels = [CONCEPT_LABELS[c] for c in ALL_CONCEPTS]

    display = W.copy()
    np.fill_diagonal(display, np.nan)   # hide self-loops

    vmax = max(0.01, float(np.nanmax(np.abs(display))))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        display.T,          # rows=target, cols=source
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(range(N_CONCEPTS))
    ax.set_yticks(range(N_CONCEPTS))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Source concept  (what causes)", fontsize=11)
    ax.set_ylabel("Target concept  (what is affected)", fontsize=11)
    ax.set_title(
        "Fuzzy Cognitive Map — Causal Influence Matrix\n"
        "W[i→j] > 0 : i promotes j  |  W[i→j] < 0 : i suppresses j",
        fontsize=12, pad=12,
    )

    # Divider between forcing / sensor blocks
    div = N_FORCING - 0.5
    for kw in (dict(x=div, ymin=0, ymax=1), ):
        ax.axvline(div, color="k", lw=1.5, ls="--", alpha=0.6)
        ax.axhline(div, color="k", lw=1.5, ls="--", alpha=0.6)

    # Block labels
    ax.text(N_FORCING / 2 - 0.5, -1.5, "Forcing", ha="center",
            fontsize=8, style="italic", color="#555")
    ax.text(N_FORCING + N_SENSOR / 2 - 0.5, -1.5, "Sensor features", ha="center",
            fontsize=8, style="italic", color="#555")

    # Annotate cells with significant weights
    for i in range(N_CONCEPTS):
        for j in range(N_CONCEPTS):
            v = display[i, j]
            if np.isnan(v) or abs(v) < 0.07:
                continue
            txt_col = "white" if abs(v) > 0.5 else "black"
            ax.text(i, j, f"{v:+.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_col)

    plt.colorbar(im, ax=ax, label="Causal weight  W[src → tgt]",
                 fraction=0.046, pad=0.04)
    plt.tight_layout()

    out = VIZ_DIR / "08_fcm_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# 5b. Visualisation — circular causal graph
# ---------------------------------------------------------------------------

def plot_causal_graph(W: np.ndarray, threshold: float = 0.07) -> None:
    labels = [CONCEPT_LABELS[c] for c in ALL_CONCEPTS]

    # Circular positions — forcing nodes at top, sensors evenly spaced below
    angles = np.linspace(0, 2 * np.pi, N_CONCEPTS, endpoint=False) - np.pi / 2
    R  = 1.0
    xs = R * np.cos(angles)
    ys = R * np.sin(angles)

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)

    # Node colours and sizes
    node_colors = ["#e07b00"] * N_FORCING + ["#2055c8"] * N_SENSOR
    out_strength = np.abs(W).sum(axis=1)   # total outgoing influence
    node_sizes   = 900 + out_strength * 2500

    # --- edges first (drawn behind nodes) ---
    for i in range(N_CONCEPTS):
        for j in range(N_CONCEPTS):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) < threshold:
                continue
            color  = "#c0392b" if w > 0 else "#2980b9"
            lw     = 0.8 + abs(w) * 3.5
            alpha  = 0.35 + abs(w) * 0.55
            alpha  = float(np.clip(alpha, 0, 0.95))
            # Shorten arrow to node edge so it doesn't overlap the circle
            shrink = 14
            ax.annotate(
                "",
                xy=(xs[j], ys[j]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    shrinkA=shrink, shrinkB=shrink,
                    connectionstyle="arc3,rad=0.18",
                ),
            )
            # Weight label near midpoint of arc
            mx = (xs[i] + xs[j]) / 2 * 1.12
            my = (ys[i] + ys[j]) / 2 * 1.12
            ax.text(mx, my, f"{w:+.2f}", fontsize=6, ha="center", va="center",
                    color=color, alpha=float(np.clip(alpha + 0.1, 0, 1)))

    # --- nodes ---
    for i in range(N_CONCEPTS):
        ax.scatter(xs[i], ys[i], s=node_sizes[i],
                   c=node_colors[i], zorder=5,
                   edgecolors="white", linewidths=2.5)

    # --- labels ---
    for i in range(N_CONCEPTS):
        ang  = angles[i]
        off  = 1.32
        ha   = "left"  if np.cos(ang) > 0.1 else ("right" if np.cos(ang) < -0.1 else "center")
        va   = "bottom" if np.sin(ang) > 0.1 else ("top"  if np.sin(ang) < -0.1 else "center")
        ax.text(
            off * np.cos(ang), off * np.sin(ang),
            labels[i],
            ha=ha, va=va, fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )

    # Legend
    patches = [
        mpatches.Patch(color="#e07b00", label="Exogenous forcing"),
        mpatches.Patch(color="#2055c8", label="Sensor / water quality"),
        mpatches.Patch(color="#c0392b", label="Positive influence (+)"),
        mpatches.Patch(color="#2980b9", label="Negative influence (−)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=9,
              framealpha=0.9, edgecolor="#ccc")

    ax.set_title(
        "Fuzzy Cognitive Map — Biscayne Bay Water Quality\n"
        f"Edges shown for |W| ≥ {threshold:.2f}   |   "
        "Node size ∝ total outgoing influence",
        fontsize=12, pad=16,
    )

    out = VIZ_DIR / "09_fcm_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# 5c. Visualisation — scenario simulations
# ---------------------------------------------------------------------------

def plot_scenarios(
    W:          np.ndarray,
    baseline_A: np.ndarray,
    n_steps:    int = 40,
) -> None:
    """
    Runs 4 scenarios, plots trajectories of 4 key sensor concepts:
      Water Temp, Salinity, ODO (mg/L), Turbidity
    """
    forcing_idx  = list(range(N_FORCING))   # [0, 1, 2]
    mean_forcing = baseline_A[:N_FORCING].copy()

    scenarios = [
        ("Baseline (mean conditions)",     "#555555",
         mean_forcing.copy()),
        ("Heavy rain  (rain=0.9)",         "#1f77b4",
         np.array([0.9, mean_forcing[1], mean_forcing[2]])),
        ("Heat wave  (temp_max=0.9)",      "#d62728",
         np.array([0.05, mean_forcing[1], 0.9])),
        ("Cold & dry  (temp_max=0.1)",     "#2ca02c",
         np.array([0.05, mean_forcing[1], 0.1])),
    ]

    plot_sensors = ["temp_c", "sal_ppt", "odo_mgL", "turbidity_fnu"]
    sensor_idxs  = [ALL_CONCEPTS.index(s) for s in plot_sensors]
    sensor_labels = [CONCEPT_LABELS[s] for s in plot_sensors]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    steps = np.arange(n_steps + 1)

    for ax_i, (si, slabel) in enumerate(zip(sensor_idxs, sensor_labels)):
        ax = axes[ax_i]
        for sc_label, color, forced_vals in scenarios:
            traj = fcm_simulate(W, baseline_A.copy(), forcing_idx, forced_vals, n_steps)
            ls = "--" if sc_label.startswith("Baseline") else "-"
            ax.plot(steps, traj[:, si], color=color, lw=2, ls=ls,
                    label=sc_label, alpha=0.9)
        ax.set_title(slabel, fontsize=11, fontweight="bold")
        ax.set_xlabel("Simulation step (days)", fontsize=9)
        ax.set_ylabel("Normalised activation  [0–1]", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "FCM Scenario Simulations — System response to environmental forcing\n"
        "Forcing held fixed; sensor concepts evolve under FCM dynamics",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    out = VIZ_DIR / "10_fcm_scenarios.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# 5d. Visualisation — influence ranking bar chart
# ---------------------------------------------------------------------------

def plot_influence_ranking(W: np.ndarray) -> None:
    """
    For each concept, compute total positive and negative outgoing influence.
    Shows a diverging stacked bar chart.
    """
    labels     = [CONCEPT_LABELS[c] for c in ALL_CONCEPTS]
    pos_out    = np.clip(W, 0, None).sum(axis=1)   # (N,) sum of positive outgoing
    neg_out    = np.clip(W, None, 0).sum(axis=1)   # (N,) sum of negative outgoing
    in_deg     = np.abs(W).sum(axis=0)             # (N,) total incoming influence

    node_colors = ["#e07b00"] * N_FORCING + ["#2055c8"] * N_SENSOR

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: outgoing influence (who is most influential)
    ax = axes[0]
    y  = np.arange(N_CONCEPTS)
    ax.barh(y, pos_out, color="#c0392b", alpha=0.8, label="Positive outgoing")
    ax.barh(y, neg_out, color="#2980b9", alpha=0.8, label="Negative outgoing")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Sum of outgoing weights", fontsize=10)
    ax.set_title("Outgoing influence\n(what drives other concepts)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    for i, c in enumerate(node_colors):
        ax.get_yticklabels()[i].set_color(c)
        ax.get_yticklabels()[i].set_fontweight("bold")

    # Right: incoming influence (who is most driven)
    ax = axes[1]
    colors = [node_colors[i] for i in np.argsort(in_deg)[::-1]]
    sorted_idx = np.argsort(in_deg)[::-1]
    ax.bar(range(N_CONCEPTS), in_deg[sorted_idx],
           color=[node_colors[i] for i in sorted_idx], alpha=0.85)
    ax.set_xticks(range(N_CONCEPTS))
    ax.set_xticklabels([labels[i] for i in sorted_idx], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Sum of incoming weights  |W|", fontsize=10)
    ax.set_title("Incoming influence\n(what is most driven by others)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for i, idx in enumerate(sorted_idx):
        ax.get_xticklabels()[i].set_color(node_colors[idx])
        ax.get_xticklabels()[i].set_fontweight("bold")

    fig.suptitle("FCM Influence Rankings — Biscayne Bay", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = VIZ_DIR / "11_fcm_influence_ranking.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load concept time series (daily, spatially averaged)
    daily = load_concept_timeseries()

    # 2. Learn FCM weights
    W, arr_norm, col_min, col_range = learn_fcm_weights(daily, alpha=2.0, lag=1)

    # 3. Save weight CSV
    labels = [CONCEPT_LABELS[c] for c in ALL_CONCEPTS]
    W_df   = pd.DataFrame(W, index=labels, columns=labels)
    out_csv = ANALYSIS_DIR / "fcm_weights.csv"
    W_df.to_csv(out_csv, float_format="%.4f")
    print(f"\nSaved weight matrix → {out_csv}")

    # 4. Baseline activation = mean of normalised daily series
    baseline_A = arr_norm.mean(axis=0)

    # 5. Print dominant edges grouped by category
    print("\n" + "=" * 60)
    print("CAUSAL SUMMARY")
    print("=" * 60)
    categories = [
        ("Forcing → Sensor",  range(N_FORCING), range(N_FORCING, N_CONCEPTS)),
        ("Sensor  → Sensor",  range(N_FORCING, N_CONCEPTS), range(N_FORCING, N_CONCEPTS)),
    ]
    for cat_name, src_range, tgt_range in categories:
        edges = [
            (W[i, j], labels[i], labels[j])
            for i in src_range for j in tgt_range
            if i != j and abs(W[i, j]) > 0.05
        ]
        if not edges:
            continue
        edges.sort(key=lambda x: abs(x[0]), reverse=True)
        print(f"\n{cat_name}:")
        for w, src, tgt in edges[:8]:
            bar  = "█" * int(abs(w) * 20)
            sign = "+" if w > 0 else "−"
            effect = "promotes" if w > 0 else "suppresses"
            print(f"  {src:<22} {effect:<11} {tgt:<22} ({sign}{abs(w):.3f})  {bar}")

    # 6. Convergence check
    test_traj = fcm_simulate(W, baseline_A.copy(), list(range(N_FORCING)),
                              baseline_A[:N_FORCING], n_steps=60)
    delta = np.abs(test_traj[-1] - test_traj[-2]).max()
    print(f"\nFCM convergence check (baseline): max |ΔA| at step 60 = {delta:.6f}")
    print("  (< 0.001 = converged)" if delta < 0.001 else "  (not fully converged — increase n_steps)")

    # 7. Visualisations
    print("\nGenerating visualisations...")
    plot_heatmap(W)
    plot_causal_graph(W, threshold=0.07)
    plot_scenarios(W, baseline_A)
    plot_influence_ranking(W)

    print("\nDone. Outputs in analysis/ and visualizations/")


if __name__ == "__main__":
    main()
