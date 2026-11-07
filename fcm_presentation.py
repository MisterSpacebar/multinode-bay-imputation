"""Render slide-ready FCM graphics from saved results, without refitting models.

Run: python fcm_presentation.py
Exports PNG and editable SVG figures to FCM/graphics/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "FCM" / "graphics"
POSITIVE = "#00796b"
NEGATIVE = "#b94b25"
GROUPS = {
    "forcing": ("#fff1da", "#ac6c00"),
    "physical": ("#e7f0fc", "#245b96"),
    "ecology": ("#edf4e5", "#577c2b"),
}

# Fixed semantic positions preserve the same concept locations across maps.
POSITIONS = {
    "Net Water (Rain\u2212PET)": (1.6, 6.7),
    "Air Temp Min": (4.8, 7.1),
    "Air Temp Max": (8.2, 6.9),
    "Water Temp": (8.2, 4.8),
    "Salinity": (1.6, 4.2),
    "ODO (mg/L)": (8.3, 2.5),
    "Depth": (1.5, 0.8),
    "Pressure": (4.8, 0.8),
    "ODO (%Sat)": (11.7, 2.5),
    "Spec. Cond.": (1.6, 2.3),
    "Turbidity": (11.7, 6.8),
    "pH": (14.9, 6.6),
    "Chlorophyll-a": (14.7, 4.2),
    "Secchi Depth": (14.8, 1.0),
    "NO\u2082+NO\u2083": (4.8, 4.6),
    "DIN": (4.8, 2.5),
}
DISPLAY = {
    "Net Water (Rain\u2212PET)": "Net water\nRain - PET",
    "ODO (mg/L)": "Dissolved oxygen\nmg/L",
    "ODO (%Sat)": "Oxygen saturation\n%",
    "Spec. Cond.": "Specific\nconductivity",
    "NO\u2082+NO\u2083": "Nitrite + nitrate",
}


def load_weights(path: Path) -> pd.DataFrame:
    weights = pd.read_csv(path, index_col=0)
    if weights.index.has_duplicates or weights.columns.has_duplicates:
        raise ValueError(f"Duplicate concept labels in {path}")
    if list(weights.index) != list(weights.columns):
        raise ValueError(f"Source and target labels must match in {path}")
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (np.abs(values) > 1).any():
        raise ValueError(f"Expected finite weights in [-1, 1] in {path}")
    unknown = set(weights.index) - POSITIONS.keys()
    if unknown:
        raise ValueError(f"No map positions for concepts: {sorted(unknown)}")
    return weights


def select_edges(
    weights: pd.DataFrame, concepts: list[str], threshold: float,
    negative_threshold: float | None = None,
) -> list[tuple[str, str, float]]:
    """Filter edges by |W|; negative edges may use a lower cutoff so they stay visible."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if negative_threshold is None:
        negative_threshold = threshold
    elif not 0 < negative_threshold <= threshold:
        raise ValueError("negative_threshold must be in (0, threshold]")
    if len(set(concepts)) != len(concepts):
        raise ValueError("Concept selection must not contain duplicates")
    missing = set(concepts) - set(weights.index)
    if missing:
        raise ValueError(f"Concepts missing from weights: {sorted(missing)}")
    return [
        (source, target, float(weights.loc[source, target]))
        for source in concepts for target in concepts
        if source != target and (
            weights.loc[source, target] >= threshold
            or weights.loc[source, target] <= -negative_threshold)
    ]


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = OUT / f"{name}.{extension}"
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, dpi=180, facecolor="white")
        print(f"Saved {path.relative_to(ROOT)}")
    plt.close(fig)


def plot_concept_map(
    weights: pd.DataFrame, name: str, title: str,
    concepts: list[str], threshold: float,
    positions: dict[str, tuple[float, float]] | None = None,
    selected_pairs: list[tuple[str, str]] | None = None,
    negative_threshold: float | None = None,
) -> None:
    edges = select_edges(weights, concepts, threshold, negative_threshold)
    if selected_pairs is not None:
        available = {(source, target) for source, target, _ in edges}
        missing = set(selected_pairs) - available
        if missing:
            raise ValueError(f"Selected edges absent or below threshold: {sorted(missing)}")
        edges = [edge for edge in edges if edge[:2] in selected_pairs]
    layout = POSITIONS if positions is None else positions
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.18)
    ax.set(xlim=(0, 16.5), ylim=(0, 8))
    ax.axis("off")
    boxes = {}
    for concept in concepts:
        group = ("forcing" if concept in list(weights.index[:3]) else
                 "ecology" if concept in list(weights.index[11:]) else "physical")
        fill, border = GROUPS[group]
        x, y = layout[concept]
        box = FancyBboxPatch(
            (x - 1.12, y - 0.39), 2.24, 0.78,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            facecolor=fill, edgecolor=border, linewidth=1.8, zorder=4,
        )
        ax.add_patch(box)
        boxes[concept] = box
        ax.text(x, y, DISPLAY.get(concept, concept), ha="center", va="center",
                fontsize=11, weight="bold", color="#243443", zorder=5)

    edge_pairs = {(source, target) for source, target, _ in edges}
    label_rects: list[tuple[float, float, float, float]] = []
    fig.canvas.draw()
    node_rects = [box.get_window_extent().bounds for box in boxes.values()]

    def overlaps(a: tuple[float, float, float, float],
                 b: tuple[float, float, float, float]) -> bool:
        return (a[0] < b[0] + b[2] and a[0] + a[2] > b[0]
                and a[1] < b[1] + b[3] and a[1] + a[3] > b[1])

    for source, target, weight in edges:
        start, end = np.array(layout[source]), np.array(layout[target])
        reciprocal = (target, source) in edge_pairs
        rad = 0.38 if reciprocal else 0.06
        color = POSITIVE if weight > 0 else NEGATIVE
        arrow = FancyArrowPatch(
            start, end, patchA=boxes[source], patchB=boxes[target],
            shrinkA=4, shrinkB=4, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=19,
            linewidth=1.2 + abs(weight) * 2.5,
            linestyle="-" if weight > 0 else "--", color=color, zorder=2,
        )
        ax.add_patch(arrow)
        # arc3 uses a clockwise perpendicular displacement in display space.
        start_px, end_px = ax.transData.transform([start, end])
        delta = end_px - start_px
        midpoint_px = (start_px + end_px) / 2 + 0.5 * rad * np.array([delta[1], -delta[0]])
        candidates = [
            midpoint_px + np.array([dx, dy])
            for dx, dy in [(0, 0), (0, 18), (0, -18), (28, 0), (-28, 0),
                           (28, 24), (-28, -24), (0, 42), (0, -42)]
        ]
        rects = [(float(p[0] - 25), float(p[1] - 10), 50.0, 20.0) for p in candidates]
        scores = [sum(overlaps(rect, other) for other in node_rects + label_rects)
                  for rect in rects]
        chosen = min(range(len(scores)), key=lambda i: scores[i])
        midpoint_px = candidates[chosen]
        label_rects.append(rects[chosen])
        midpoint = ax.transData.inverted().transform(midpoint_px)
        ax.text(*midpoint, f"{weight:+.2f}", color=color, fontsize=10,
                ha="center", va="center", zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.6})

    total = len(select_edges(weights, list(weights.index), np.finfo(float).eps))
    fig.suptitle(title, x=0.5, y=0.97, fontsize=22, weight="bold", color="#243443")
    fig.text(
        0.5, 0.91,
        f"{len(concepts)}/{len(weights)} concepts | {len(edges)}/{total} nonzero cross-concept edges shown"
        + (f" | display filter: W >= +{threshold:.2f} or W <= \u2212{negative_threshold:.2f}"
           if negative_threshold is not None else f" | display filter: |W| >= {threshold:.2f}")
        + (" | curated pathways" if selected_pairs is not None else ""),
        ha="center", fontsize=12, color="#52606c",
    )
    legend = [Patch(facecolor=fill, edgecolor=border, label=group.title())
              for group, (fill, border) in GROUPS.items()]
    legend += [
        Line2D([0], [0], color=POSITIVE, lw=2, label="Positive (+) association"),
        Line2D([0], [0], color=NEGATIVE, lw=2, ls="--", label="Negative (-) association"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.065),
               ncol=5, frameon=False, fontsize=11)
    fig.text(
        0.5, 0.035,
        "Arrow: source at t -> target at fitted lag. Width = |W|. Self-loops hidden, retained in model.",
        ha="center", fontsize=10,
    )
    fig.text(0.5, 0.012,
             "Learned lagged associations, NOT established causal effects. Filtering changes display only.",
             ha="center", fontsize=10, color=NEGATIVE)
    save_figure(fig, name)


def plot_validation() -> None:
    metrics = pd.read_csv(ROOT / "analysis" / "fcm_hindcast_metrics.csv").dropna(subset=["R2"])
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.19, right=0.92, bottom=0.15, top=0.84)
    rows = metrics.sort_values("R2", ascending=False)
    y = np.arange(len(rows))
    ax.barh(y, rows["R2"], color=[POSITIVE if r >= 0 else NEGATIVE for r in rows["R2"]], height=0.65)
    ax.set_yticks(y, rows["concept"], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(min(0, rows["R2"].min() - 0.1), 1.0)
    ax.axvline(0, color="#243443", lw=1.5)
    ax.set_xlabel("In-sample one-step hindcast R-squared (monthly)", fontsize=13)
    ax.grid(axis="x", alpha=0.15)
    for i, row in enumerate(rows.itertuples()):
        ax.text(max(row.R2, 0) + 0.02, i, f"{row.R2:.2f}  (n={row.n})", va="center", fontsize=11)
    fig.suptitle("In-sample fit is moderate; skill beyond persistence is not shown", fontsize=24, weight="bold")
    fig.text(0.5, 0.89,
             f"{len(rows)} modelled concepts, R-squared {rows['R2'].min():.2f} to {rows['R2'].max():.2f}. "
             "Most of this is 'this month predicts next month'.",
             ha="center", fontsize=14, color=NEGATIVE)
    fig.text(0.5, 0.045,
             "In-sample, and targets can include loader-filled values. Held-out, the FCM loses to persistence "
             "on every concept tested (graphics/07).\n"
             "Depth, pressure and turbidity are not modelled (too sparse).",
             ha="center", fontsize=12)
    save_figure(fig, "05_validation_scorecard")


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    cards = [
        (2.3, 6.3, "Continuous sensors + weather", "5-minute sensor values\nGNN fills missing values\nObserved weight 1; imputed 0.5"),
        (2.3, 2.9, "Historical field samples", "Surface grab samples + weather\nNutrients and physical overlap\nDifferent sampling / site coverage"),
        (7.6, 6.3, "Physical FCM", "Daily regional averages\n11 concepts\n3 forcing + 8 sensor variables"),
        (7.6, 2.9, "Nutrient FCM", "Monthly merged series\n16 concepts\nAdds 5 nutrient / ecology variables"),
        (13.0, 4.6, "Learn, explore, evaluate", "Min-max scaling + lagged Ridge\nSigned normalized weights\nScenarios + hindcast diagnostics"),
    ]
    for x, y, title, body in cards:
        ax.add_patch(FancyBboxPatch((x - 2.05, y - 1.2), 4.1, 2.4,
                                   boxstyle="round,pad=0.1", fc="#e7f0fc", ec="#245b96"))
        ax.text(x, y + 0.65, title, ha="center", fontsize=13, weight="bold")
        ax.text(x, y - 0.25, body, ha="center", va="center", fontsize=12, linespacing=1.6)
    for start, end in [((4.5, 6.3), (5.4, 6.3)), ((4.5, 2.9), (5.4, 2.9)),
                       ((7.6, 5.0), (7.6, 4.2)), ((9.8, 6.3), (10.8, 5.2)),
                       ((9.8, 2.9), (10.8, 4.0))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=25,
                                    color="#245b96", lw=2))
    fig.suptitle("From incomplete measurements to an interpretable network", fontsize=23, weight="bold")
    fig.text(0.5, 0.07,
             "GNN reconstructs sensor values. FCM explores variable relationships. Neither step establishes causality.\n"
             "Monthly nutrient observations do not span the entire continuous-sensor period.",
             ha="center", fontsize=13, color=NEGATIVE)
    save_figure(fig, "01_data_to_fcm")


def plot_event_context() -> None:
    anomalies = pd.read_csv(ROOT / "analysis" / "dieoff_anomalies.csv", index_col=0)
    features = ["temp", "do_mgL", "do_per", "ph", "din"]
    labels = ["Water temperature", "Dissolved oxygen (mg/L)",
              "Oxygen saturation", "pH", "DIN"]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.23, right=0.94, bottom=0.18, top=0.83)
    y = np.arange(len(features))
    for offset, month, color in [(-0.18, "2021-09", "#245b96"),
                                  (0.18, "2022-10", "#ac6c00")]:
        values = anomalies.loc[month, features].to_numpy(dtype=float)
        ax.barh(y + offset, values, height=0.32, label=month, color=color)
        for i, value in enumerate(values):
            ax.text(value + (0.04 if value >= 0 else -0.04), i + offset,
                    f"{value:+.2f}", ha="left" if value >= 0 else "right",
                    va="center", fontsize=12)
    ax.set_yticks(y, labels, fontsize=13)
    ax.set_xlim(-2.5, 1.5)
    ax.axvline(0, color="#243443", lw=1)
    ax.set_xlabel("Standardized anomaly relative to pooled multi-year surface-sample baseline", fontsize=12)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(fontsize=13, frameon=False)
    fig.suptitle("Event-month context | oxygen below the pooled baseline", fontsize=23, weight="bold")
    fig.text(0.5, 0.89, "Descriptive measurements, NOT FCM predictions or causal attribution.",
             ha="center", fontsize=14, color=NEGATIVE)
    fig.text(0.5, 0.055,
             "Saved anomalies pool bay + canal surface samples (excluding inlet / reef / outfall sites).\n"
             "Not seasonally matched; pooled means can hide site-specific extremes. Anomaly is not a hypoxia diagnosis.",
             ha="center", fontsize=12)
    save_figure(fig, "06_event_context")


def main() -> None:
    physical = load_weights(ROOT / "analysis" / "fcm_weights_physical.csv")
    nutrient = load_weights(ROOT / "analysis" / "fcm_weights_nutrient.csv")
    plot_pipeline()
    plot_concept_map(physical, "02_physical_concept_map",
                     "Physical FCM | strongest cross-concept relationships",
                     list(physical.index), 0.25, negative_threshold=0.10)
    plot_concept_map(nutrient, "03_nutrient_concept_map",
                     "Nutrient FCM | strong-edge overview",
                     list(nutrient.index), 0.42, negative_threshold=0.22)
    focus_positions = {
        "Net Water (Rain\u2212PET)": (1.6, 6.5),
        "Air Temp Max": (6.0, 6.8),
        "Water Temp": (10.1, 5.8),
        "ODO (mg/L)": (14.2, 6.5),
        "NO\u2082+NO\u2083": (1.6, 3.2),
        "DIN": (6.0, 2.3),
        "Chlorophyll-a": (10.2, 3.0),
        "Secchi Depth": (14.2, 1.3),
    }
    plot_concept_map(nutrient, "04_ecology_focus_map",
                     "Ecology-focused FCM | selected pathways, signed relationships",
                     list(focus_positions), 0.20, focus_positions,
                     selected_pairs=[
                         (list(nutrient.index)[0], "NO\u2082+NO\u2083"),
                         (list(nutrient.index)[0], "Chlorophyll-a"),
                         ("Air Temp Max", "Water Temp"),
                         ("Water Temp", "ODO (mg/L)"),
                         ("Water Temp", "DIN"),
                         ("Water Temp", "Chlorophyll-a"),
                         ("NO\u2082+NO\u2083", "Chlorophyll-a"),
                         ("NO\u2082+NO\u2083", "DIN"),
                         ("DIN", "Secchi Depth"),
                         ("Chlorophyll-a", "Secchi Depth"),
                     ])
    plot_validation()
    plot_event_context()


if __name__ == "__main__":
    main()
