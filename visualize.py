"""
visualize.py
------------
Generates plots from the imputed data and analysis results.
All figures are saved to visualizations/.

Run:
    python visualize.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from pathlib import Path

OUT_DIR      = Path("visualizations")
IMPUTED_DIR  = Path("imputed_output")
ANALYSIS_DIR = Path("analysis")

OUT_DIR.mkdir(exist_ok=True)

FEATURE_LABELS = {
    "temp_c":          "Temperature (°C)",
    "sal_ppt":         "Salinity (PPT)",
    "odo_mgL":         "ODO (mg/L)",
    "depth_m":         "Depth (m)",
    "pressure_psia":   "Pressure (psia)",
    "odo_pct":         "ODO (% Sat)",
    "spec_cond_uScm":  "Specific Conductance (µS/cm)",
    "turbidity_fnu":   "Turbidity (FNU)",
}

SENSOR_COLS = list(FEATURE_LABELS.keys())

NODE_COLORS = {
    "L0":                  "#e6194b",
    "L1":                  "#3cb44b",
    "L2":                  "#4363d8",
    "L6":                  "#f58231",
    "L7":                  "#911eb4",
    "biscayne_bay":        "#42d4f4",
    "consolidated_crest5": "#a9a9a9",
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
# Load helpers
# ---------------------------------------------------------------------------

def load_node(name):
    path = IMPUTED_DIR / IMPUTED_FILES[name]
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_all():
    return {name: load_node(name) for name in IMPUTED_FILES}


# ---------------------------------------------------------------------------
# Plot 1: Time-series overview — one panel per feature, all nodes overlaid
# ---------------------------------------------------------------------------

def plot_timeseries_overview(nodes: dict):
    """One figure with 8 subplots; each shows all stations for that feature."""
    print("  [1] Time-series overview...")
    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)
    axes = axes.flatten()

    for ax, feat in zip(axes, SENSOR_COLS):
        label = FEATURE_LABELS[feat]
        obs_col = feat + "_observed"

        for name, df in nodes.items():
            if feat not in df.columns:
                continue

            color = NODE_COLORS.get(name, "gray")

            # Downsample to hourly for readability
            hourly = df[feat].resample("1h").mean()
            ax.plot(hourly.index, hourly.values,
                    color=color, lw=0.7, alpha=0.8, label=name)

        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.grid(True, lw=0.3, alpha=0.5)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.98))
    fig.autofmt_xdate(rotation=30)
    fig.suptitle("All Stations — Imputed Sensor Time Series\n(2025-03 to 2026-06)",
                 y=1.00, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    path = OUT_DIR / "01_timeseries_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved → {path}")


# ---------------------------------------------------------------------------
# Plot 2: Observed vs imputed coverage per node (stacked bar)
# ---------------------------------------------------------------------------

def plot_coverage(nodes: dict):
    """Stacked bar showing % observed vs imputed for each feature × node."""
    print("  [2] Observed vs imputed coverage...")

    node_names = list(nodes.keys())
    n_nodes  = len(node_names)
    n_feats  = len(SENSOR_COLS)

    obs_pct  = np.zeros((n_nodes, n_feats))
    for i, name in enumerate(node_names):
        df = nodes[name]
        for j, feat in enumerate(SENSOR_COLS):
            obs_col = feat + "_observed"
            if obs_col in df.columns:
                obs_pct[i, j] = df[obs_col].mean() * 100

    imp_pct = 100 - obs_pct

    x = np.arange(n_feats)
    bar_w = 0.13
    offsets = np.linspace(-(n_nodes - 1) / 2, (n_nodes - 1) / 2, n_nodes) * bar_w

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, name in enumerate(node_names):
        color = NODE_COLORS.get(name, "gray")
        ax.bar(x + offsets[i], obs_pct[i],  width=bar_w, color=color,
               alpha=0.9, label=name)
        ax.bar(x + offsets[i], imp_pct[i],  width=bar_w, color=color,
               alpha=0.3, bottom=obs_pct[i], hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in SENSOR_COLS],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of timesteps", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Data Coverage per Node & Feature\n(solid = observed, hatched = imputed)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    plt.tight_layout()

    path = OUT_DIR / "02_coverage_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved → {path}")


# ---------------------------------------------------------------------------
# Plot 3: Channel importance bar chart
# ---------------------------------------------------------------------------

def plot_channel_importance():
    print("  [3] Channel importance...")
    df = pd.read_csv(ANALYSIS_DIR / "channel_importance.csv")
    df = df.sort_values("importance", ascending=True)

    colors = ["#d73027" if v >= 0 else "#4575b4" for v in df["importance"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh([FEATURE_LABELS.get(f, f) for f in df["feature"]],
                   df["importance"], color=colors, edgecolor="white", lw=0.5)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Δ Loss when feature is masked\n(positive = feature helps imputation)", fontsize=9)
    ax.set_title("Feature Importance for Imputation\n(permutation method)", fontsize=11)

    for bar, val in zip(bars, df["importance"]):
        ax.text(val + np.sign(val) * 2e-6, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2e}", va="center", ha="left" if val >= 0 else "right",
                fontsize=8)

    ax.grid(axis="x", lw=0.3, alpha=0.5)
    plt.tight_layout()

    path = OUT_DIR / "03_channel_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved → {path}")


# ---------------------------------------------------------------------------
# Plot 4: Spatial graph — nodes on map with attention-weighted edges
# ---------------------------------------------------------------------------

def plot_spatial_graph():
    print("  [4] Spatial graph with attention weights...")
    attn = pd.read_csv(ANALYSIS_DIR / "attention_weights.csv", index_col=0)
    node_names = attn.columns.tolist()

    # Approximate coordinates for the 7 nodes (from preprocess output)
    COORDS = {
        "L0":                    (25.911432, -80.137273),
        "L1":                    (25.874909, -80.183208),
        "L2":                    (25.853710, -80.159411),
        "L6":                    (25.844242, -80.148913),
        "L7":                    (25.779099, -80.208921),
        "biscayne_bay":          (25.870369, -80.164941),
        "consolidated_crest5":   (25.726950, -80.269660),
    }

    fig, ax = plt.subplots(figsize=(9, 11))

    # Draw edges (source → target), thickness and alpha from attention weight
    max_w = attn.values.max()
    for tgt in node_names:
        for src in node_names:
            if src == tgt:
                continue
            w = attn.loc[tgt, src]
            if w < 0.05:
                continue
            lat_s, lon_s = COORDS[src]
            lat_t, lon_t = COORDS[tgt]
            ax.annotate("",
                xy=(lon_t, lat_t), xytext=(lon_s, lat_s),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.15,head_length=0.08",
                    color="steelblue",
                    alpha=float(w / max_w) * 0.85 + 0.10,
                    lw=float(w / max_w) * 3.0 + 0.3,
                    connectionstyle="arc3,rad=0.08",
                ))

    # Draw nodes
    for name in node_names:
        lat, lon = COORDS[name]
        color = NODE_COLORS.get(name, "gray")
        ax.scatter(lon, lat, s=220, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(lon + 0.003, lat + 0.002, name, fontsize=9, fontweight="bold",
                color=color, zorder=6)

    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title("Station Graph — GAT Attention Weights\n"
                 "(arrow thickness = attention strength, tgt ← src)", fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.4)
    plt.tight_layout()

    path = OUT_DIR / "04_spatial_graph.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved → {path}")


# ---------------------------------------------------------------------------
# Plot 5: Attention weight heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap():
    print("  [5] Attention heatmap...")
    attn = pd.read_csv(ANALYSIS_DIR / "attention_weights.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=attn.values.max())

    ax.set_xticks(range(len(attn.columns)))
    ax.set_yticks(range(len(attn.index)))
    ax.set_xticklabels(attn.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(attn.index, fontsize=9)
    ax.set_xlabel("Source node (listened to)", fontsize=9)
    ax.set_ylabel("Target node (listener)", fontsize=9)
    ax.set_title("Mean GAT Attention Weights\n(how much each node attends to its neighbours)",
                 fontsize=10)

    for i in range(len(attn.index)):
        for j in range(len(attn.columns)):
            val = attn.values[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if val > attn.values.max() * 0.6 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Attention weight")
    plt.tight_layout()

    path = OUT_DIR / "05_attention_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved → {path}")


# ---------------------------------------------------------------------------
# Plot 6: Per-node observed vs imputed time series for top 3 features
# ---------------------------------------------------------------------------

def plot_node_detail(nodes: dict):
    """For each node: one figure with 3 panels showing raw/imputed splits."""
    print("  [6] Per-node detail plots...")
    TOP_FEATS = ["temp_c", "sal_ppt", "odo_mgL"]

    for name, df in nodes.items():
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        for ax, feat in zip(axes, TOP_FEATS):
            obs_col = feat + "_observed"
            hourly_val = df[feat].resample("1h").mean()
            hourly_obs = df[obs_col].resample("1h").max() if obs_col in df.columns else None

            ax.plot(hourly_val.index, hourly_val.values,
                    color="steelblue", lw=0.8, alpha=0.9, label="imputed / observed")

            if hourly_obs is not None:
                # Shade imputed regions
                is_imp = (hourly_obs == 0)
                ax.fill_between(hourly_val.index, hourly_val.min(), hourly_val.max(),
                                where=is_imp, color="orange", alpha=0.15,
                                label="imputed region")

            ax.set_ylabel(FEATURE_LABELS[feat], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)
            ax.legend(fontsize=7, loc="upper right")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        fig.autofmt_xdate(rotation=30)
        fig.suptitle(f"Station: {name} — Observed (blue) with Imputed Regions (orange shading)",
                     fontsize=11)
        plt.tight_layout()

        path = OUT_DIR / f"06_node_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"     saved → {path}")


# ---------------------------------------------------------------------------
# Plot 7: Monthly salinity comparison across all nodes
# ---------------------------------------------------------------------------

def plot_monthly_salinity(nodes: dict):
    print("  [7] Monthly salinity boxplot...")
    records = []
    for name, df in nodes.items():
        monthly = df["sal_ppt"].resample("ME").mean().dropna()
        for dt, val in monthly.items():
            records.append({"node": name, "month": dt.strftime("%Y-%m"), "sal_ppt": val})

    df_long = pd.DataFrame(records)
    if df_long.empty:
        return

    months = sorted(df_long["month"].unique())
    node_names = list(nodes.keys())
    x = np.arange(len(months))
    bar_w = 0.13
    offsets = np.linspace(-(len(node_names)-1)/2, (len(node_names)-1)/2, len(node_names)) * bar_w

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, name in enumerate(node_names):
        sub = df_long[df_long["node"] == name].set_index("month")
        vals = [sub.loc[m, "sal_ppt"] if m in sub.index else np.nan for m in months]
        ax.bar(x + offsets[i], vals, width=bar_w,
               color=NODE_COLORS.get(name, "gray"), label=name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Salinity (PPT)", fontsize=10)
    ax.set_title("Monthly Mean Salinity by Station (2025-03 to 2026-06)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    plt.tight_layout()

    path = OUT_DIR / "07_monthly_salinity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading imputed data...")
    nodes = load_all()

    print("Generating plots...")
    plot_timeseries_overview(nodes)
    plot_coverage(nodes)
    plot_channel_importance()
    plot_spatial_graph()
    plot_attention_heatmap()
    plot_node_detail(nodes)
    plot_monthly_salinity(nodes)

    print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} figures in {OUT_DIR}/")
