"""
multinode_visuals.py
---------------------
Presentation-ready graphics for the "multi-node ST-GNN" slide deck.
These are distinct from visualizations/ (which are raw analysis dumps) -
each figure here is purpose-built to support one slide of the rundown.

Outputs (saved to multinode_visuals/):
  01_architecture_diagram.png       - model pipeline schematic (slide 3)
  02_coverage_heatmap.png           - node x feature coverage matrix (slide 1/7)
  03_attention_network_annotated.png- spatial graph with hub/pair callouts (slide 2/5)
  04_baseline_comparison_grouped.png- GNN vs naive baselines, per feature (slide 8)
  05_node_reliability_tiers.png     - nodes ranked into confidence tiers (slide 7)
  06_dieoff_event_comparison.png    - Sept'21 vs Oct'22 anomaly signature (slide 9)
  07_findings_summary.png           - env-science vs data-science takeaways (slide 10)
  08_training_pipeline.png          - chronological split + masking/window scheme (slide 4)
  09_attention_ranking.png          - hub-ness / concentration bar ranking (slide 5)
  10_before_after_imputation.png    - raw record vs. imputed record, real gap example (slide 3)
  11_before_after_small_gap.png     - same, but a ~2h gap at 5-min resolution (slide 3)
  12_channel_importance_clean.png   - tiered, normalised channel importance (slide 6)
  13_gnn_concept.png                - generic graph concept -> our stations (slide 3, new)
  14_dieoff_overview.png            - anomaly bars + causal pathway diagram (slide 10)
  15_baseline_overview.png          - bars + "why this test doesn't show the GNN's edge" (slide 9)
  16_wetseason_overview.png         - wet-season rainfall bars + pathway, 2021 vs 2022
  17_limitations_and_next.png       - GNN shortcomings + bridge to next FCM presentation

Run:
    python multinode_visuals.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import contextily as ctx
from pyproj import Transformer
from pathlib import Path

OUT_DIR = Path("multinode_visuals")
OUT_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR = Path("analysis")
IMPUTED_DIR = Path("imputed_output")

_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def merc(lon, lat): return _to_merc.transform(lon, lat)

NODE_COLORS = {
    "L0": "#e6194b", "L1": "#3cb44b", "L2": "#4363d8", "L6": "#f58231",
    "L7": "#911eb4", "biscayne_bay": "#42d4f4", "consolidated_crest5": "#a9a9a9",
}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})


# ---------------------------------------------------------------------------
# 1. Architecture diagram
# ---------------------------------------------------------------------------

def plot_architecture():
    print("  [1] Architecture diagram...")
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    # Technical name stays the heading; plain-English line added under it for clarity
    stages = [
        ("Node Encoder",
         "values | mask | rain/temp\nsin/cos time \u2192 Linear+LN+ReLU",
         "Turns each station's raw\nreadings into one combined signal",
         "#4363d8"),
        ("Spatial GAT\n(\u00d73 layers)",
         "attend to k-nearest\nneighbours, weighted\nby distance prior",
         "Lets each station borrow\ninfo from its nearest neighbours",
         "#3cb44b"),
        ("Temporal GRU\n(\u00d72 layers)",
         "propagate fused\nrepresentation\nforward in time",
         "Carries recent trends\nforward between readings",
         "#f58231"),
        ("Decoder",
         "Linear \u2192 8-channel\nreconstructed values",
         "Turns the pattern back\ninto real sensor values",
         "#e6194b"),
    ]

    box_w, box_h = 2.75, 3.3
    xs = [0.6, 3.9, 7.2, 10.5]
    y0 = 2.3

    for (title, tech_desc, plain_desc, color), x in zip(stages, xs):
        box = FancyBboxPatch((x, y0), box_w, box_h,
                              boxstyle="round,pad=0.06,rounding_size=0.14",
                              linewidth=2, edgecolor=color, facecolor=color + "1a")
        ax.add_patch(box)
        ax.text(x + box_w / 2, y0 + box_h - 0.32, title, ha="center", va="top",
                fontsize=15, fontweight="bold", color=color)
        ax.text(x + box_w / 2, y0 + box_h - 1.15, tech_desc, ha="center", va="top",
                fontsize=10.5, color="#333333")
        ax.plot([x + 0.3, x + box_w - 0.3], [y0 + 1.55, y0 + 1.55], color=color, lw=1, alpha=0.5)
        ax.text(x + box_w / 2, y0 + 1.40, plain_desc, ha="center", va="top",
                fontsize=10.8, color="#444444")

    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x1 + box_w, y0 + box_h / 2), (x2, y0 + box_h / 2),
                                      arrowstyle="-|>", mutation_scale=24,
                                      linewidth=2.2, color="#333333"))

    # --- Input side: a small cluster of station dots feeding in ---
    dot_colors = list(NODE_COLORS.values())
    cx, cy = 0.6 + box_w / 2, 1.55
    positions = [(-0.4, 0.14), (0.0, 0.32), (0.4, 0.14), (-0.2, -0.18), (0.2, -0.18)]
    for (dx, dy), c in zip(positions, dot_colors):
        ax.scatter(cx + dx, cy + dy, s=150, color=c, edgecolor="white", linewidths=1, zorder=5)
    ax.annotate("", xy=(cx, y0 - 0.05), xytext=(cx, cy + 0.4),
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.5))
    ax.text(cx, cy - 0.75, "All 7 stations,\nsame moment in time", ha="center", fontsize=10, color="#555555")

    # --- Output side: mini before/after sparkline showing a gap being filled ---
    ox, oy = 10.5 + box_w / 2, 1.65
    t = np.linspace(-0.5, 0.5, 40)
    y_true = 0.14 * np.sin(t * 8)
    gap = (t > -0.1) & (t < 0.15)
    ax.plot(ox + t, oy + 0.4 + y_true, color="#999999", lw=1.6)
    ax.plot(ox + t[~gap], oy + y_true[~gap], color="#2e7d32", lw=2)
    ax.plot(ox + t[gap], oy + y_true[gap], color="#e6194b", lw=2, ls="--")
    ax.annotate("", xy=(ox, y0 - 0.05), xytext=(ox, oy + 0.6),
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.5))
    ax.text(ox, oy - 0.65,
            "Real readings kept as-is (green);\nmissing readings replaced (red dashed)",
            ha="center", fontsize=10, color="#555555")

    # Loop-back note for GRU hidden state
    ax.annotate("hidden state carried\nwindow-to-window",
                xy=(7.2 + box_w / 2, y0 + box_h), xytext=(7.2 + box_w / 2, 6.95),
                ha="center", fontsize=10, color="#f58231",
                arrowprops=dict(arrowstyle="-|>", color="#f58231", lw=1.3, connectionstyle="arc3,rad=-0.3"))

    ax.set_title("BayImputationGNN: Per-Timestep Forward Pass", fontsize=17, fontweight="bold", pad=12)
    plt.tight_layout()
    path = OUT_DIR / "01_architecture_diagram.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 1a. What is a GNN - generic graph concept, then mapped to our stations
# ---------------------------------------------------------------------------

def plot_gnn_concept():
    print("  [1a] GNN concept diagram...")
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6))

    # --- Left: a generic graph ---
    ax_l.set_xlim(0, 10)
    ax_l.set_ylim(0, 10)
    ax_l.axis("off")
    generic_pos = {
        "A": (2, 7.5), "B": (5, 8.5), "C": (7.5, 6.5),
        "D": (3, 3.5), "E": (6.5, 3),
    }
    generic_edges = [("A", "B"), ("B", "C"), ("A", "D"), ("D", "E"), ("E", "C"), ("B", "D")]
    for u, v in generic_edges:
        ax_l.plot([generic_pos[u][0], generic_pos[v][0]], [generic_pos[u][1], generic_pos[v][1]],
                  color="#888888", lw=1.6, zorder=1)
    for name, (x, y) in generic_pos.items():
        ax_l.scatter(x, y, s=650, color="#4363d8", edgecolor="white", linewidths=2, zorder=3)
        ax_l.text(x, y, name, ha="center", va="center", color="white", fontsize=13,
                  fontweight="bold", zorder=4)
    ax_l.annotate("Node = an entity", xy=generic_pos["C"], xytext=(8.7, 8.2),
                  fontsize=10.5, color="#333333",
                  arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
    mid_x = (generic_pos["A"][0] + generic_pos["D"][0]) / 2
    mid_y = (generic_pos["A"][1] + generic_pos["D"][1]) / 2
    ax_l.annotate("Edge = a relationship\nbetween two entities", xy=(mid_x, mid_y), xytext=(0.2, 1.2),
                  fontsize=10.5, color="#333333",
                  arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
    ax_l.set_title("A Graph, in General", fontsize=15, fontweight="bold")

    # --- Right: the same idea, mapped onto our 7 stations ---
    ax_r.axis("off")
    # simple k=2 nearest-neighbour lines for a clean, uncluttered look
    names = list(COORDS.keys())
    latlon = {n: COORDS[n] for n in names}

    def dist(a, b):
        return ((latlon[a][0] - latlon[b][0]) ** 2 + (latlon[a][1] - latlon[b][1]) ** 2) ** 0.5

    for n in names:
        nearest = sorted((m for m in names if m != n), key=lambda m: dist(n, m))[:2]
        for m in nearest:
            x1, y1 = latlon[n][1], latlon[n][0]
            x2, y2 = latlon[m][1], latlon[m][0]
            ax_r.plot([x1, x2], [y1, y2], color="#888888", lw=1.4, zorder=1)

    # Manual per-node label offsets (lon, lat degrees) to dodge the tight northern cluster
    right_label_offset = {
        "L0": (0.007, 0.003), "L1": (-0.014, 0.001), "biscayne_bay": (0.013, -0.007),
        "L2": (-0.003, -0.013), "L6": (0.013, 0.0), "L7": (0.0, -0.013),
        "consolidated_crest5": (0.0, 0.012),
    }
    for name in names:
        lat, lon = latlon[name]
        color = NODE_COLORS.get(name, "gray")
        ax_r.scatter(lon, lat, s=420, color=color, edgecolor="white", linewidths=2, zorder=3)
        dlon, dlat = right_label_offset.get(name, (0.0, -0.013))
        va = "bottom" if dlat > 0 else "top"
        ax_r.text(lon + dlon, lat + dlat, name, ha="center", va=va, fontsize=10,
                  fontweight="bold", color=color, zorder=4)

    ax_r.set_title("Our Graph: Bay Sensor Stations", fontsize=15, fontweight="bold")
    ax_r.text(0.5, -0.1, "Node = a station  |  Edge = \"these two stations are close enough\n"
                         "to share information\" (weighted by real-world distance)",
              transform=ax_r.transAxes, ha="center", fontsize=10, color="#555555")

    fig.suptitle("What Is a Graph Neural Network (GNN)?", fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "13_gnn_concept.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 1b. Before / after imputation - a real gap, raw vs. filled
# ---------------------------------------------------------------------------

def plot_before_after():
    print("  [1b] Before/after imputation example...")
    df = pd.read_csv(IMPUTED_DIR / "raw-data-platformL1_parameters_imputed.csv",
                      index_col=0, parse_dates=True)
    feat, obs_col = "sal_ppt", "sal_ppt_observed"
    gap_start = pd.Timestamp("2025-09-10 15:40:00", tz="UTC")
    gap_end = pd.Timestamp("2026-01-01 04:55:00", tz="UTC")

    hourly_val = df[feat].resample("1h").mean()
    hourly_obs = df[obs_col].resample("1h").max()
    raw_only = hourly_val.where(hourly_obs > 0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    ax_top.plot(raw_only.index, raw_only.values, color="#2e7d32", lw=1)
    ax_top.axvspan(gap_start, gap_end, color="#cfcfcf", alpha=0.5)
    ax_top.text(gap_start + (gap_end - gap_start) / 2, ax_top.get_ylim()[1] * 0.9,
                "112-day gap - no sensor data", ha="center", fontsize=10, color="#555555")
    ax_top.set_ylabel("Salinity (PPT)")
    ax_top.set_title("BEFORE: Raw Sensor Record (station L1)", fontsize=12.5, fontweight="bold")
    ax_top.grid(True, lw=0.3, alpha=0.5)

    is_gap = (hourly_val.index >= gap_start) & (hourly_val.index <= gap_end)
    ax_bot.plot(hourly_val.index[~is_gap], hourly_val.values[~is_gap], color="#2e7d32", lw=1,
                label="Observed")
    ax_bot.plot(hourly_val.index[is_gap], hourly_val.values[is_gap], color="#e6194b", lw=1.2,
                label="Model-filled (imputed)")
    ax_bot.axvspan(gap_start, gap_end, color="#fde3e3", alpha=0.5)
    ax_bot.set_ylabel("Salinity (PPT)")
    ax_bot.set_title("AFTER: Imputed Record - same gap now filled from neighbouring stations",
                      fontsize=12.5, fontweight="bold")
    ax_bot.legend(loc="upper right", fontsize=9)
    ax_bot.grid(True, lw=0.3, alpha=0.5)
    ax_bot.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate(rotation=20)

    fig.suptitle("Before vs. After Imputation: A Real 112-Day Gap at Station L1",
                 fontsize=14.5, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT_DIR / "10_before_after_imputation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 1c. Before / after imputation - a small, couple-hour gap at full 5-min resolution
# ---------------------------------------------------------------------------

def plot_before_after_small_gap():
    print("  [1c] Before/after imputation - small gap example...")
    df = pd.read_csv(IMPUTED_DIR / "raw-data-platformL1_parameters_imputed.csv",
                      index_col=0, parse_dates=True)
    feat, obs_col = "odo_mgL", "odo_mgL_observed"
    gap_start = pd.Timestamp("2025-05-13 21:00:00", tz="UTC")
    gap_end = pd.Timestamp("2025-05-13 22:00:00", tz="UTC")
    window_start = gap_start - pd.Timedelta(hours=6)
    window_end = gap_end + pd.Timedelta(hours=6)

    sub = df.loc[window_start:window_end, [feat, obs_col]].copy()
    is_gap = (sub.index >= gap_start) & (sub.index <= gap_end)
    raw_only = sub[feat].where(sub[obs_col] > 0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    ax_top.plot(sub.index, raw_only.values, color="#2e7d32", lw=1.4, marker="o", ms=3)
    ax_top.axvspan(gap_start, gap_end, color="#cfcfcf", alpha=0.5)
    ax_top.text(gap_start + (gap_end - gap_start) / 2, 0.94, "1h gap - no sensor data",
                transform=ax_top.get_xaxis_transform(), ha="center", va="top",
                fontsize=10, color="#555555")
    ax_top.set_ylabel("Dissolved Oxygen (mg/L)")
    ax_top.set_title("BEFORE: Raw Sensor Record (station L1, 5-min readings)",
                      fontsize=12.5, fontweight="bold")
    ax_top.grid(True, lw=0.3, alpha=0.5)

    ax_bot.plot(sub.index[~is_gap], sub[feat].values[~is_gap], color="#2e7d32", lw=1.4,
                marker="o", ms=3, label="Observed")
    ax_bot.plot(sub.index[is_gap], sub[feat].values[is_gap], color="#e6194b", lw=1.6,
                marker="o", ms=3, label="Model-filled (imputed)")
    ax_bot.axvspan(gap_start, gap_end, color="#fde3e3", alpha=0.5)
    ax_bot.set_ylabel("Dissolved Oxygen (mg/L)")
    ax_bot.set_title("AFTER: Imputed Record - gap filled using recent trend + neighbouring stations",
                      fontsize=12.5, fontweight="bold")
    ax_bot.legend(loc="upper right", fontsize=9)
    ax_bot.grid(True, lw=0.3, alpha=0.5)
    ax_bot.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=20)

    fig.suptitle("Before vs. After Imputation: A Short ~1h Gap at Station L1 (May 13, 2025)",
                 fontsize=14.5, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT_DIR / "11_before_after_small_gap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 1d. Channel importance - cleaner, tiered, normalised version of slide 6's graphic
# ---------------------------------------------------------------------------

def plot_channel_importance_clean():
    print("  [1d] Channel importance (clean)...")
    df = pd.read_csv(ANALYSIS_DIR / "channel_importance.csv")
    df["magnitude"] = -df["importance"]
    df = df.sort_values("magnitude", ascending=True)
    max_mag = df["magnitude"].max()
    df["pct"] = df["magnitude"] / max_mag * 100

    labels = {
        "temp_c": "Temperature", "sal_ppt": "Salinity", "turbidity_fnu": "Turbidity",
        "odo_mgL": "Dissolved Oxygen (mg/L)", "odo_pct": "Dissolved Oxygen (% Sat)",
        "spec_cond_uScm": "Specific Conductance", "depth_m": "Depth", "pressure_psia": "Pressure",
    }

    def tier(pct):
        if pct >= 50:
            return "#c62828", "Critical - shared across stations"
        if pct >= 5:
            return "#f9a825", "Moderate - some cross-station signal"
        return "#9e9e9e", "Negligible - purely local, safe to cut"

    colors, _tier_labels = zip(*(tier(p) for p in df["pct"]))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [labels.get(f, f) for f in df["feature"]]
    bars = ax.barh(names, df["pct"], color=colors, edgecolor="white")
    for bar, pct in zip(bars, df["pct"]):
        label = f"{pct:.0f}%" if pct >= 1 else "~0%"
        ax.text(max(pct, 1.5) + 1.5, bar.get_y() + bar.get_height() / 2, label,
                va="center", fontsize=10)

    ax.set_xlabel("Importance for cross-station imputation (% of Temperature, the top driver)")
    ax.set_xlim(0, 112)
    ax.set_title("Which Sensors Actually Help Fill In Other Stations' Gaps?",
                 fontsize=13.5, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#c62828", label="Critical - shared across stations"),
        mpatches.Patch(color="#f9a825", label="Moderate - some cross-station signal"),
        mpatches.Patch(color="#9e9e9e", label="Negligible - purely local, safe to cut"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    plt.tight_layout()
    path = OUT_DIR / "12_channel_importance_clean.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 2. Coverage heatmap (node x feature), source: report/coverage_chart.txt
# ---------------------------------------------------------------------------

COVERAGE = {
    # node                 : {Temp, Sal, DO_mg, Depth, Press, DO_pct, SpecCond, Turb}
    "L0":                  [35.9, 35.9, 35.9, 23.9, 23.9, 35.9, 23.9, 35.9],
    "L1":                  [67.7, 67.7, 67.7, 31.7, 31.7, 42.0, 31.7, 42.0],
    "L2":                  [68.7, 69.5, 69.5, 50.8, 50.8, 61.6, 50.8, 61.6],
    "L6":                  [13.1, 13.1, 13.1,  9.1,  9.1,  9.1,  9.1,  9.1],
    "L7":                  [19.5, 19.5, 19.5,  7.5,  7.5, 19.5,  7.5, 19.5],
    "biscayne_bay":        [32.4, 32.4, 32.4,  0.0,  0.0, 12.0,  0.0, 12.0],
    "consolidated_crest5": [12.0, 12.0, 12.0,  0.0,  0.0, 12.0,  0.0, 12.0],
}
COVERAGE_COLS = ["Temp", "Sal", "DO(mg/L)", "Depth", "Pressure", "DO(%)", "SpecCond", "Turbidity"]


def plot_coverage_heatmap():
    print("  [2] Coverage heatmap...")
    nodes = list(COVERAGE.keys())
    mat = np.array([COVERAGE[n] for n in nodes])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(COVERAGE_COLS)))
    ax.set_xticklabels(COVERAGE_COLS, rotation=35, ha="right")
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 30 or v > 75 else "black")

    plt.colorbar(im, ax=ax, shrink=0.85, label="% timesteps directly observed")
    ax.set_title("Sensor Coverage by Node × Feature\n(darker green = more reliable, red = mostly model-imputed)",
                 fontsize=11)
    plt.tight_layout()
    path = OUT_DIR / "02_coverage_heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 3. Annotated attention network - highlights hub + fragile pair
# ---------------------------------------------------------------------------

COORDS = {
    "L0": (25.911432, -80.137273), "L1": (25.874909, -80.183208),
    "L2": (25.853710, -80.159411), "L6": (25.844242, -80.148913),
    "L7": (25.779099, -80.208921), "biscayne_bay": (25.870369, -80.164941),
    "consolidated_crest5": (25.726950, -80.269660),
}


def plot_attention_network_annotated():
    print("  [3] Annotated attention network (basemap)...")
    attn = pd.read_csv(ANALYSIS_DIR / "attention_weights.csv", index_col=0)
    node_names = attn.columns.tolist()
    max_w = attn.values.max()

    # Project station coordinates to Web Mercator so they can sit on real map tiles
    xy = {name: merc(lon, lat) for name, (lat, lon) in COORDS.items()}
    CLUSTER = ["L0", "L1", "L2", "biscayne_bay", "L6"]   # nodes too close together at full extent

    def draw_edges(axis, subset=None):
        for tgt in node_names:
            for src in node_names:
                if src == tgt:
                    continue
                if subset is not None and (src not in subset or tgt not in subset):
                    continue
                w = attn.loc[tgt, src]
                if w < 0.05:
                    continue
                x_s, y_s = xy[src]
                x_t, y_t = xy[tgt]
                is_pair = {src, tgt} == {"L7", "consolidated_crest5"}
                axis.annotate("", xy=(x_t, y_t), xytext=(x_s, y_s),
                              arrowprops=dict(
                                  arrowstyle="->,head_width=0.18,head_length=0.10",
                                  color="crimson" if is_pair else "steelblue",
                                  alpha=0.95 if is_pair else float(w / max_w) * 0.75 + 0.15,
                                  lw=4.5 if is_pair else float(w / max_w) * 4.0 + 0.6,
                                  connectionstyle="arc3,rad=0.08", zorder=4))

    def draw_nodes(axis, subset=None, fontsize=12, label_dy=-650):
        for name in node_names:
            if subset is not None and name not in subset:
                continue
            x, y = xy[name]
            color = NODE_COLORS.get(name, "gray")
            is_hub = name == "L2"
            axis.scatter(x, y, s=560 if is_hub else 300, color=color, zorder=6,
                         edgecolors="gold" if is_hub else "white",
                         linewidths=3.5 if is_hub else 1.5)
            axis.text(x, y + label_dy, name, fontsize=fontsize, fontweight="bold",
                      ha="center", va="top", color=color, zorder=7,
                      path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    fig, ax = plt.subplots(figsize=(11, 9.5))

    # Only draw the L7<->crest5 pair and the (few) edges touching them on the main map;
    # the cluster's internal edges are shown in the zoomed inset instead, to avoid clutter.
    draw_edges(ax)
    draw_nodes(ax, subset=["L7", "consolidated_crest5"], fontsize=13, label_dy=-1200)

    # Fragile-pair badge, with a real arrowhead pointing at the connecting edge
    pair_mid_x = (xy["L7"][0] + xy["consolidated_crest5"][0]) / 2
    pair_mid_y = (xy["L7"][1] + xy["consolidated_crest5"][1]) / 2
    badge2_x, badge2_y = pair_mid_x + 3200, pair_mid_y - 200
    ax.annotate("", xy=(pair_mid_x + 300, pair_mid_y), xytext=(badge2_x, badge2_y),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.6), zorder=8)
    ax.scatter(badge2_x, badge2_y, s=620, color="crimson", edgecolors="black",
               linewidths=1.3, zorder=9, marker="o")
    ax.text(badge2_x, badge2_y, "2", ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", zorder=10)

    # Legend box (top-left, clear of all stations)
    legend_text = (
        "①  HUB (L2): merged L2+L3+L5 platforms (same\n"
        "     site - Little River), best-covered node\n"
        "     (60.4%), primary source for L0/L6/Bay\n"
        "     - see zoomed inset\n\n"
        "②  FRAGILE PAIR: L7 \u2194 crest5 attend\n"
        "     ONLY to each other (w=1.0) - single\n"
        "     point of failure if both go offline"
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10.5,
            va="top", ha="left", zorder=10,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#555555", alpha=0.92))

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Pad bounds around stations before adding basemap tiles (extra room south for crest5's label)
    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    pad_x = (max(xs) - min(xs)) * 0.20 + 1500
    pad_y_top = (max(ys) - min(ys)) * 0.15 + 1500
    pad_y_bot = (max(ys) - min(ys)) * 0.15 + 3500
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y_bot, max(ys) + pad_y_top)
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldGrayCanvas,
                    zoom=12, attribution=False)

    # --- Zoomed inset: the 5 stations that are too close together at full extent ---
    cxs = [xy[n][0] for n in CLUSTER]
    cys = [xy[n][1] for n in CLUSTER]
    cpad = 900
    x0c, x1c = min(cxs) - cpad, max(cxs) + cpad
    y0c, y1c = min(cys) - cpad, max(cys) + cpad

    axins = ax.inset_axes([0.60, 0.30, 0.39, 0.50])
    axins.set_xlim(x0c, x1c)
    axins.set_ylim(y0c, y1c)
    draw_edges(axins, subset=CLUSTER)
    draw_nodes(axins, subset=CLUSTER, fontsize=11, label_dy=-350)
    hub_x, hub_y = xy["L2"]
    axins.annotate("HUB\n(L2+L3+L5)", xy=(hub_x, hub_y), xytext=(hub_x + 900, hub_y + 900),
                   fontsize=10.5, fontweight="bold", color="#7a5c00", zorder=8,
                   arrowprops=dict(arrowstyle="-|>", color="goldenrod", lw=1.6),
                   bbox=dict(boxstyle="round,pad=0.25", fc="#fff8dc", ec="goldenrod"))
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.4)
    ctx.add_basemap(axins, crs="EPSG:3857", source=ctx.providers.Esri.WorldGrayCanvas,
                    zoom=15, attribution=False)
    axins.set_title("Zoom: northern-bay cluster", fontsize=9.5, fontweight="bold")

    # Connector lines + rectangle showing where the inset comes from on the main map
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.2, alpha=0.9)

    ax.set_title("Learned Spatial Attention Network\narrow thickness = attention strength (target \u2190 source)",
                 fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    path = OUT_DIR / "03_attention_network_annotated.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 4. Baseline comparison - grouped bars per feature
# ---------------------------------------------------------------------------

def plot_baseline_comparison():
    print("  [4] Baseline comparison...")
    df = pd.read_csv(ANALYSIS_DIR / "baseline_comparison.csv")
    df = df[df["node"] == "ALL"]

    features = ["temp_c", "sal_ppt", "odo_mgL"]
    feature_labels = {"temp_c": "Temperature (\u00b0C)", "sal_ppt": "Salinity (PPT)", "odo_mgL": "DO (mg/L)"}
    methods = ["climatology", "spatial_mean", "locf", "linear_interp", "gnn"]
    method_labels = {"climatology": "Climatology", "spatial_mean": "Spatial\nMean",
                      "locf": "LOCF", "linear_interp": "Linear\nInterp", "gnn": "GNN\n(Ours)"}
    method_colors = {"climatology": "#a9a9a9", "spatial_mean": "#f58231",
                      "locf": "#4363d8", "linear_interp": "#3cb44b", "gnn": "#e6194b"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, feat in zip(axes, features):
        sub = df[df["feature"] == feat].set_index("method").reindex(methods)
        colors = [method_colors[m] for m in methods]
        bars = ax.bar(range(len(methods)), sub["rmse"], color=colors, edgecolor="white", width=0.7)
        best_idx = sub["rmse"].values.argmin()
        max_val = sub["rmse"].max()
        for i, (bar, m) in enumerate(zip(bars, methods)):
            if m == "gnn":
                bar.set_edgecolor("black")
                bar.set_linewidth(2.2)
            if i == best_idx:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.14,
                        "\u2193 BEST", ha="center", fontsize=9.5, fontweight="bold", color="#2e7d32")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([method_labels[m] for m in methods], fontsize=10.5)
        ax.set_title(feature_labels[feat], fontsize=15, fontweight="bold", pad=10)
        ax.set_ylabel("RMSE (lower is better)", fontsize=10.5)
        ax.set_ylim(0, max_val * 1.32)
        for bar, val in zip(bars, sub["rmse"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max_val * 0.02, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.grid(axis="y", lw=0.3, alpha=0.4)

    fig.suptitle("GNN vs. Naive Baselines: 30%-Random-Gap Test", fontsize=17, fontweight="bold", y=1.02)
    fig.text(0.5, 0.965,
             "GNN beats climatology & spatial averaging everywhere - but its real edge (long, structural gaps) isn't tested here",
             ha="center", fontsize=11, color="#555555")
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    path = OUT_DIR / "04_baseline_comparison_grouped.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 4b. Baseline overview - bars + "why" explainer (what each protocol actually tests)
# ---------------------------------------------------------------------------

def plot_baseline_overview():
    print("  [4b] Baseline overview (bars + why explainer)...")
    df = pd.read_csv(ANALYSIS_DIR / "baseline_comparison.csv")
    df = df[df["node"] == "ALL"]

    features = ["temp_c", "sal_ppt", "odo_mgL"]
    feature_labels = {"temp_c": "Temperature", "sal_ppt": "Salinity", "odo_mgL": "DO (mg/L)"}
    methods = ["locf", "linear_interp", "gnn"]
    method_labels = {"locf": "LOCF", "linear_interp": "Linear\nInterp", "gnn": "GNN\n(Ours)"}
    method_colors = {"locf": "#4363d8", "linear_interp": "#3cb44b", "gnn": "#e6194b"}

    fig = plt.figure(figsize=(12.5, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.35)

    for j, feat in enumerate(features):
        ax = fig.add_subplot(gs[0, j])
        sub = df[df["feature"] == feat].set_index("method").reindex(methods)
        colors = [method_colors[m] for m in methods]
        bars = ax.bar(range(len(methods)), sub["rmse"], color=colors, edgecolor="white", width=0.65)
        for bar, m in zip(bars, methods):
            if m == "gnn":
                bar.set_edgecolor("black")
                bar.set_linewidth(2)
        max_val = sub["rmse"].max()
        for bar, val in zip(bars, sub["rmse"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max_val * 0.04, f"{val:.2f}",
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([method_labels[m] for m in methods], fontsize=9.5)
        ax.set_title(feature_labels[feat], fontsize=13, fontweight="bold")
        ax.set_ylim(0, max_val * 1.3)
        if j == 0:
            ax.set_ylabel("RMSE (lower = better)", fontsize=9.5)
        ax.grid(axis="y", lw=0.3, alpha=0.4)

    # --- "Why" explainer: what the test measures vs. what deployment looks like ---
    ax_l = fig.add_subplot(gs[1, 0:1])
    ax_r = fig.add_subplot(gs[1, 1:3])

    def draw_strip(ax, pattern, title, note):
        n = len(pattern)
        for i, is_gap in enumerate(pattern):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                         color="#cfcfcf" if is_gap else "#2e7d32"))
        ax.set_xlim(0, n)
        ax.set_ylim(-0.9, 1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.text(n / 2, -0.5, note, ha="center", va="top", fontsize=9.3, color="#555555")

    rng = np.random.default_rng(3)
    scattered = rng.random(60) < 0.30
    draw_strip(ax_l, scattered, "This Test: Scattered Random Gaps",
               "Real values are always minutes away\n\u2192 LOCF / Linear Interp win easily")

    structural = np.zeros(60, dtype=bool)
    structural[8:56] = True
    draw_strip(ax_r, structural, "Real Deployment: Structural Gaps (e.g., crest5, 92.5% missing)",
               "No real value nearby for weeks/months \u2192 LOCF flatlines, Linear Interp draws one\n"
               "straight line across the whole gap \u2192 only the GNN can use OTHER stations to fill it")

    fig.suptitle("Baseline Comparison: What This Test Shows, and What It Doesn't",
                 fontsize=16.5, fontweight="bold", y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT_DIR / "15_baseline_overview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 5. Node reliability tiers
# ---------------------------------------------------------------------------

def plot_reliability_tiers():
    print("  [5] Node reliability tiers...")
    overall = {n: np.mean(v) for n, v in COVERAGE.items()}
    order = sorted(overall, key=overall.get, reverse=True)
    vals = [overall[n] for n in order]

    def tier_color(v):
        if v >= 50: return "#2e7d32"     # reliable
        if v >= 20: return "#f9a825"     # moderate
        return "#c62828"                 # low-confidence

    colors = [tier_color(v) for v in vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(order, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9)

    ax.axhline(50, color="#2e7d32", lw=1, ls="--", alpha=0.6)
    ax.axhline(20, color="#c62828", lw=1, ls="--", alpha=0.6)
    ax.text(len(order) - 0.4, 51, "Reliable", color="#2e7d32", fontsize=8, ha="right")
    ax.text(len(order) - 0.4, 21, "Low-confidence threshold", color="#c62828", fontsize=8, ha="right")

    ax.set_ylabel("Mean observed coverage (%)")
    ax.set_title("Node Reliability Tiers\n(green = trust single-timestamp values, red = seasonal aggregates only)",
                 fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = OUT_DIR / "05_node_reliability_tiers.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 6. Die-off event comparison
# ---------------------------------------------------------------------------

def plot_dieoff_comparison():
    print("  [6] Die-off event comparison...")
    df = pd.read_csv(ANALYSIS_DIR / "dieoff_anomalies.csv")
    df = df.set_index("period")

    cols = ["temp", "sal", "do_per", "do_mgL", "ph", "nh4", "din"]
    labels = ["Temp", "Salinity", "DO (%)", "DO (mg/L)", "pH", "NH4", "DIN"]
    events = {"2021-09": "Sept 2021\n(fish + seagrass)", "2022-10": "Oct 2022\n(seagrass only)"}

    x = np.arange(len(cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (period, label) in enumerate(events.items()):
        vals = df.loc[period, cols].astype(float).values
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label)
        for bx, v in zip(bars, vals):
            ax.text(bx.get_x() + bx.get_width() / 2, v + (0.05 if v >= 0 else -0.12),
                    f"{v:.2f}", ha="center", fontsize=7.5)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Z-score vs 2021-2024 baseline")
    ax.set_title("Die-off Event Signatures: Shared Hypoxia, Different Mechanism\n"
                 "2021 = freshwater dilution stress | 2022 = saline nutrient-loading stress", fontsize=11)
    ax.legend()
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    plt.tight_layout()
    path = OUT_DIR / "06_dieoff_event_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 6b. Die-off overview - anomaly bars + causal pathway diagram for each event
# ---------------------------------------------------------------------------

def plot_dieoff_overview():
    print("  [6b] Die-off overview (bars + pathways)...")
    df = pd.read_csv(ANALYSIS_DIR / "dieoff_anomalies.csv").set_index("period")

    cols = ["sal", "do_mgL", "ph", "nh4"]
    labels = ["Salinity", "DO (mg/L)", "pH", "NH4"]
    events = {"2021-09": ("Sept 2021 (fish + seagrass)", "#1f77b4"),
              "2022-10": ("Oct 2022 (seagrass only)", "#ff7f0e")}

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.38)
    ax_bar = fig.add_subplot(gs[0])
    ax_flow = fig.add_subplot(gs[1])

    x = np.arange(len(cols))
    width = 0.35
    for i, (period, (label, color)) in enumerate(events.items()):
        vals = df.loc[period, cols].astype(float).values
        offset = (i - 0.5) * width
        bars = ax_bar.bar(x + offset, vals, width, label=label, color=color)
        for bx, v in zip(bars, vals):
            ax_bar.text(bx.get_x() + bx.get_width() / 2, v + (0.05 if v >= 0 else -0.14),
                        f"{v:.2f}", ha="center", fontsize=9)
    ax_bar.axhline(0, color="black", lw=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=11)
    ax_bar.set_ylabel("Z-score vs 2021-2024 baseline", fontsize=10)
    ax_bar.set_title("The Key Divergence: Same Hypoxia, Different Chemistry",
                      fontsize=13.5, fontweight="bold")
    ax_bar.legend(fontsize=9.5, loc="lower left")
    ax_bar.grid(axis="y", lw=0.3, alpha=0.5)

    # --- Causal pathway diagrams ---
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 4.6)
    ax_flow.axis("off")

    def draw_pathway(y_center, steps, color, label):
        box_w, box_h = 2.15, 1.15
        xs = [0.2, 2.65, 5.1, 7.55]
        for (text, x) in zip(steps, xs):
            box = FancyBboxPatch((x, y_center - box_h / 2), box_w, box_h,
                                  boxstyle="round,pad=0.06,rounding_size=0.12",
                                  linewidth=1.6, edgecolor=color, facecolor=color + "22")
            ax_flow.add_patch(box)
            ax_flow.text(x + box_w / 2, y_center, text, ha="center", va="center",
                         fontsize=8.7, color="#222222")
        for x1, x2 in zip(xs[:-1], xs[1:]):
            ax_flow.add_patch(FancyArrowPatch((x1 + box_w, y_center), (x2, y_center),
                                               arrowstyle="-|>", mutation_scale=16,
                                               linewidth=1.6, color=color))
        ax_flow.text(-0.1, y_center, label, ha="right", va="center", fontsize=10.5,
                     fontweight="bold", color=color, rotation=90)

    draw_pathway(3.5,
                 ["Canal freshwater\ndischarge", "Salinity + pH drop\n(dilution stress)",
                  "Hypoxia\n(DO \u2193\u2193)", "Fish + seagrass\ndie-off"],
                 "#1f77b4", "2021")
    draw_pathway(1.1,
                 ["Canal nutrient\nloading (NH4, DIN \u2191)", "Microbial respiration\nconsumes oxygen",
                  "Hypoxia\n(DO \u2193\u2193)", "Seagrass die-off only\n(fish relocated)"],
                 "#ff7f0e", "2022")

    ax_flow.set_title("Why the Same Symptom Had Two Different Causes", fontsize=13.5, fontweight="bold", pad=8)

    fig.suptitle("Die-off Events: Full Picture", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=(0.03, 0, 1, 0.95))
    path = OUT_DIR / "14_dieoff_overview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 6c. Wet-season rainfall comparison - bars + pathway (2021 vs 2022)
# ---------------------------------------------------------------------------

def plot_wetseason_overview():
    print("  [6c] Wet-season rainfall overview (bars + pathways)...")
    rain = pd.read_csv("water_data/rainfall/miami_airport_daily_2021_2024.csv")
    rain["date"] = pd.to_datetime(rain[["year", "month", "day"]])

    months = [6, 7, 8, 9, 10]
    month_labels = ["Jun", "Jul", "Aug", "Sep", "Oct"]
    colors = {2021: "#1f77b4", 2022: "#ff7f0e"}

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.38)
    ax_bar = fig.add_subplot(gs[0])
    ax_flow = fig.add_subplot(gs[1])

    x = np.arange(len(months))
    width = 0.35
    totals = {}
    for i, yr in enumerate([2021, 2022]):
        sub = rain[(rain["year"] == yr) & (rain["month"].isin(months))]
        monthly = sub.groupby("month")["rain_in"].sum().reindex(months, fill_value=0)
        totals[yr] = monthly.sum()
        offset = (i - 0.5) * width
        bars = ax_bar.bar(x + offset, monthly.values, width,
                           label=f"{yr} (total {monthly.sum():.1f} in)", color=colors[yr])
        for bx, v in zip(bars, monthly.values):
            ax_bar.text(bx.get_x() + bx.get_width() / 2, v + 0.3, f"{v:.1f}",
                        ha="center", fontsize=9)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(month_labels, fontsize=11)
    ax_bar.set_ylabel("Monthly rainfall (in)", fontsize=10)
    ax_bar.set_title("Wet Season Rainfall: 2021 vs. 2022 (Miami Airport)",
                      fontsize=13.5, fontweight="bold")
    ax_bar.set_ylim(0, 19)
    ax_bar.legend(fontsize=9.5, loc="upper right")
    ax_bar.grid(axis="y", lw=0.3, alpha=0.5)

    # --- Rainfall "character" pathway diagrams ---
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 4.6)
    ax_flow.axis("off")

    def draw_pathway(y_center, steps, color, label):
        box_w, box_h = 2.15, 1.15
        xs = [0.2, 2.65, 5.1, 7.55]
        for (text, x) in zip(steps, xs):
            box = FancyBboxPatch((x, y_center - box_h / 2), box_w, box_h,
                                  boxstyle="round,pad=0.06,rounding_size=0.12",
                                  linewidth=1.6, edgecolor=color, facecolor=color + "22")
            ax_flow.add_patch(box)
            ax_flow.text(x + box_w / 2, y_center, text, ha="center", va="center",
                         fontsize=8.5, color="#222222")
        for x1, x2 in zip(xs[:-1], xs[1:]):
            ax_flow.add_patch(FancyArrowPatch((x1 + box_w, y_center), (x2, y_center),
                                               arrowstyle="-|>", mutation_scale=16,
                                               linewidth=1.6, color=color))
        ax_flow.text(-0.1, y_center, label, ha="right", va="center", fontsize=10.5,
                     fontweight="bold", color=color, rotation=90)

    draw_pathway(3.5,
                 ["Sustained wet season\n(35.4 in total)", "Distributed moderate\nrain days (peak 1.3in)",
                  "Gradual canal\ndischarge buildup", "Sept 2021:\ndilution die-off"],
                 colors[2021], "2021")
    draw_pathway(1.1,
                 ["Wetter overall\n(47.9 in total)", "Two named storms:\npre-Alex (Jun 3-4) +\nHurricane Ian (Sep 27-28)",
                  "Sudden nutrient/\norganic pulse", "Oct 2022:\nnutrient-loading die-off"],
                 colors[2022], "2022")

    ax_flow.set_title("Same Season, Different Rainfall Character",
                       fontsize=13.5, fontweight="bold", pad=8)
    ax_flow.text(5, 0.05,
                 "Pre-Alex: NHC Tropical Cyclone Report AL012022 (Brown & Delgado, 2022)  |  "
                 "Ian: NHC Tropical Cyclone Report AL092022 (Bucci et al., 2023)",
                 ha="center", va="bottom", fontsize=7.5, style="italic", color="#777777")

    fig.suptitle("Wet-Season Rainfall Patterns: 2021 vs. 2022", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=(0.03, 0, 1, 0.95))
    path = OUT_DIR / "16_wetseason_overview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 7. Findings summary infographic
# ---------------------------------------------------------------------------

ENV_FINDINGS = [
    "Temp & salinity are the bay's dominant\nspatially-coherent 'carrier' signals",
    "Depth/pressure/spec.cond. are locally-driven\n(redundant for network-wide monitoring)",
    "Same hypoxia, different mechanism:\n2021 = freshwater dilution,\n2022 = saline nutrient loading",
    "Canal-mouth stations show stress\n~2 months before open-bay impact",
]

DS_FINDINGS = [
    "Graph attention is interpretable:\nrecovered a sensible hub + fragile pair\nfrom data alone",
    "Permutation/cross-feature importance\ncan justify sensor deployment decisions",
    "A model can lose on the metric you\nreport (random-gap RMSE) yet be the\nonly option for the case you need\n(structural missingness)",
    "Attention concentration (w=1.0 pair)\nis itself an operational risk signal",
]


def plot_findings_summary():
    print("  [7] Findings summary infographic...")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.add_patch(FancyBboxPatch((0.2, 8.6), 4.5, 0.9, boxstyle="round,pad=0.1",
                                 fc="#e8f5e9", ec="#2e7d32", lw=1.5))
    ax.text(2.45, 9.05, "Environmental Science", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#2e7d32")

    ax.add_patch(FancyBboxPatch((5.3, 8.6), 4.5, 0.9, boxstyle="round,pad=0.1",
                                 fc="#e3f2fd", ec="#1565c0", lw=1.5))
    ax.text(7.55, 9.05, "Data Science / Analytics", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1565c0")

    y = 7.7
    for item in ENV_FINDINGS:
        ax.add_patch(FancyBboxPatch((0.2, y - 1.15), 4.5, 1.3, boxstyle="round,pad=0.08",
                                     fc="#f1f8f2", ec="#a5d6a7", lw=1))
        ax.text(0.5, y - 0.5, item, ha="left", va="center", fontsize=8.7, color="#1b3d1e")
        y -= 1.75

    y = 7.7
    for item in DS_FINDINGS:
        ax.add_patch(FancyBboxPatch((5.3, y - 1.15), 4.5, 1.3, boxstyle="round,pad=0.08",
                                     fc="#eef5fc", ec="#90caf9", lw=1))
        ax.text(5.6, y - 0.5, item, ha="left", va="center", fontsize=8.7, color="#0d2f52")
        y -= 1.75

    ax.set_title("Key Findings by Audience", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    path = OUT_DIR / "07_findings_summary.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 7b. Limitations - GNN shortcomings + bridge to next presentation (FCM)
# ---------------------------------------------------------------------------

GNN_LIMITATIONS = [
    "Fills a 5-minute gap and a\n5-month gap the same way -\nno sense of its own confidence",
    "Only tested on short gaps -\nlong gaps have never been\nformally scored",
    "A couple of stations lean on\njust one other station,\nwith no backup",
]

FCM_PREVIEW = [
    "Shifts the question from\n\"what's missing\" to\n\"why it's changing\"",
    "Adds cause-and-effect\nreasoning between variables",
    "Same dataset, new lens:\ncausal structure instead\nof point values",
]


def plot_limitations_and_next():
    print("  [7b] Limitations + next-steps diagram...")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    ax.add_patch(FancyBboxPatch((0.3, 6.5), 4.2, 1.0, boxstyle="round,pad=0.1",
                                 fc="#fdecea", ec="#c62828", lw=1.8))
    ax.text(2.4, 7.0, "GNN: Gap Limitations", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#c62828")

    ax.add_patch(FancyBboxPatch((5.5, 6.5), 4.2, 1.0, boxstyle="round,pad=0.1",
                                 fc="#ede7f6", ec="#5e35b1", lw=1.8))
    ax.text(7.6, 7.0, "Next Up: FCM", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#5e35b1")

    y = 5.6
    for item in GNN_LIMITATIONS:
        ax.add_patch(FancyBboxPatch((0.3, y - 1.35), 4.2, 1.55, boxstyle="round,pad=0.1",
                                     fc="#fef4f3", ec="#ef9a9a", lw=1.2))
        ax.text(2.4, y - 0.57, item, ha="center", va="center", fontsize=11.5, color="#5c1a1a")
        y -= 1.85

    y = 5.6
    for item in FCM_PREVIEW:
        ax.add_patch(FancyBboxPatch((5.5, y - 1.35), 4.2, 1.55, boxstyle="round,pad=0.1",
                                     fc="#f5f2fb", ec="#b39ddb", lw=1.2))
        ax.text(7.6, y - 0.57, item, ha="center", va="center", fontsize=11.5, color="#31235a")
        y -= 1.85

    ax.add_patch(FancyArrowPatch((4.5, 4.0), (5.5, 4.0), arrowstyle="-|>",
                                 mutation_scale=32, linewidth=3, color="#5e35b1"))

    ax.set_title("From Filling Gaps to Explaining Them",
                 fontsize=16, fontweight="bold", pad=10)
    plt.tight_layout()
    path = OUT_DIR / "17_limitations_and_next.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 8. Training pipeline - chronological split + masking/window scheme
# ---------------------------------------------------------------------------

def plot_training_pipeline():
    print("  [8] Training pipeline diagram...")
    t_start = pd.Timestamp("2025-03-10")
    t_end = pd.Timestamp("2026-08-01")
    t_split = t_start + 0.8 * (t_end - t_start)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6.5),
                                          gridspec_kw={"height_ratios": [1, 1.6]})

    # --- Top: chronological 80/20 split over the full record ---
    ax_top.barh(0, (t_split - t_start).days, left=0, height=0.6, color="#4363d8", label="TRAIN (80%)")
    ax_top.barh(0, (t_end - t_split).days, left=(t_split - t_start).days, height=0.6,
                color="#f58231", label="VAL (20%)")
    ax_top.axvline((t_split - t_start).days, color="black", lw=1.2, ls="--")
    ax_top.text((t_split - t_start).days / 2, 0, "TRAIN\n(80%)", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
    ax_top.text((t_split - t_start).days + (t_end - t_split).days / 2, 0, "VAL\n(20%)",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax_top.text(0, 0.55, t_start.strftime("%b %Y"), ha="left", fontsize=9)
    ax_top.text((t_split - t_start).days, 0.55, t_split.strftime("%b %Y") + " (split)",
                ha="center", fontsize=9, fontweight="bold")
    ax_top.text((t_end - t_start).days, 0.55, t_end.strftime("%b %Y"), ha="right", fontsize=9)
    ax_top.set_xlim(0, (t_end - t_start).days)
    ax_top.set_ylim(-0.5, 0.9)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_top.set_title("Chronological split (no shuffling - avoids leaking future events into training)",
                      fontsize=11, fontweight="bold")

    # Zoom-in connector from the split point down to the window detail panel
    ax_top.annotate("", xy=(0.5, -0.6), xytext=(0.15, -0.55),
                     xycoords=("axes fraction", "axes fraction"),
                     textcoords=("axes fraction", "axes fraction"),
                     arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2,
                                     connectionstyle="arc3,rad=-0.15"))

    # --- Bottom: what one 6h window looks like during training ---
    n_steps = 72  # WIN_LEN
    rng = np.random.default_rng(7)
    status = np.full(n_steps, "missing", dtype=object)
    observed_idx = rng.choice(n_steps, size=int(n_steps * 0.55), replace=False)
    status[observed_idx] = "observed"
    masked_idx = rng.choice(observed_idx, size=int(len(observed_idx) * 0.30), replace=False)
    status[masked_idx] = "masked_target"

    colors = {"observed": "#2e7d32", "masked_target": "#f9a825", "missing": "#cfcfcf"}
    for i, s in enumerate(status):
        ax_bot.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=colors[s]))

    # Second, overlapping window (50% stride) drawn below to show overlap
    off = n_steps // 2
    for i, s in enumerate(status):
        ax_bot.add_patch(plt.Rectangle((i + off, -1.4), 0.9, 1, color=colors[s], alpha=0.55))

    ax_bot.annotate("", xy=(off, 0.5), xytext=(0, 0.5),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax_bot.text(off / 2, 0.75, "STRIDE = 36 steps (50% overlap)", ha="center", fontsize=9)

    ax_bot.set_xlim(-2, n_steps + off + 2)
    ax_bot.set_ylim(-2, 2.2)
    ax_bot.set_yticks([0.5, -0.9])
    ax_bot.set_yticklabels(["Window t", "Window t+stride"], fontsize=9)
    ax_bot.set_xticks([])
    for spine in ax_bot.spines.values():
        spine.set_visible(False)
    ax_bot.set_title("One 6h window (72 steps @ 5-min) - random masking creates the reconstruction task",
                      fontsize=11, fontweight="bold")

    legend_handles = [mpatches.Patch(color=colors["observed"], label="Observed (visible to model)"),
                       mpatches.Patch(color=colors["masked_target"], label="Observed but masked (loss target)"),
                       mpatches.Patch(color=colors["missing"], label="Genuinely missing")]
    ax_bot.legend(handles=legend_handles, loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle("Training Protocol: Chronological Split + Masking Strategy", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT_DIR / "08_training_pipeline.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------
# 9. Attention ranking - hub-ness and dependency concentration per node
# ---------------------------------------------------------------------------

def plot_attention_ranking():
    print("  [9] Attention ranking bars...")
    attn = pd.read_csv(ANALYSIS_DIR / "attention_weights.csv", index_col=0)
    nodes = attn.columns.tolist()

    # Hub-ness: how much OTHER nodes rely on this one as a source (column sum)
    supplied = attn.sum(axis=0).reindex(nodes)
    # Fragility: how concentrated is this node's own attention on a single source (row max)
    concentration = attn.max(axis=1).reindex(nodes)

    order_supplied = supplied.sort_values(ascending=True).index.tolist()
    order_conc = concentration.sort_values(ascending=True).index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors1 = [NODE_COLORS.get(n, "gray") for n in order_supplied]
    bars1 = ax1.barh(order_supplied, supplied.loc[order_supplied], color=colors1, edgecolor="white")
    for bar, n in zip(bars1, order_supplied):
        if n == "L2":
            bar.set_edgecolor("gold")
            bar.set_linewidth(2.5)
    for bar, v in zip(bars1, supplied.loc[order_supplied]):
        ax1.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    ax1.set_xlabel("Total attention supplied to other nodes")
    ax1.set_title("Hub-ness\n(higher = more relied upon as a source)", fontsize=10.5)
    ax1.set_xlim(0, max(supplied) * 1.55)
    l7_val = supplied.loc["L7"]
    ax1.annotate("1.00 of this is crest5's\nexclusive dependency, not\nbroad reliance - see right panel",
                 xy=(l7_val, order_supplied.index("L7")), xytext=(l7_val + 0.30, order_supplied.index("L7")),
                 fontsize=7.8, va="center", color="#555555")

    colors2 = [NODE_COLORS.get(n, "gray") for n in order_conc]
    bars2 = ax2.barh(order_conc, concentration.loc[order_conc], color=colors2, edgecolor="white")
    for bar, n in zip(bars2, order_conc):
        if n in ("L7", "consolidated_crest5"):
            bar.set_edgecolor("crimson")
            bar.set_linewidth(2.5)
    for bar, v in zip(bars2, concentration.loc[order_conc]):
        ax2.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    ax2.set_xlabel("Max attention weight on a single source")
    ax2.set_title("Dependency Concentration\n(1.0 = relies on exactly one source - fragile)", fontsize=10.5)
    ax2.set_xlim(0, 1.15)

    fig.suptitle("Learned Attention: Who Is a Hub, Who Is Fragile", fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.02,
             "L2 (gold outline) is the hub for the well-covered northern cluster (broad reliance from L0/L1/L6/Bay).\n"
             "L7's raw total is inflated by crest5's single, fragile dependency (red outline, right panel) - not a sign of independent reliability.",
             ha="center", fontsize=8.5, color="#444444")
    plt.tight_layout(rect=(0, 0.06, 1, 0.93))
    path = OUT_DIR / "09_attention_ranking.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved -> {path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating slide graphics -> {OUT_DIR}/")
    plot_architecture()
    plot_gnn_concept()
    plot_before_after()
    plot_before_after_small_gap()
    plot_channel_importance_clean()
    plot_coverage_heatmap()
    plot_attention_network_annotated()
    plot_baseline_comparison()
    plot_baseline_overview()
    plot_reliability_tiers()
    plot_dieoff_comparison()
    plot_dieoff_overview()
    plot_wetseason_overview()
    plot_findings_summary()
    plot_limitations_and_next()
    plot_training_pipeline()
    plot_attention_ranking()
    print("Done.")
