"""Explainer graphics: what an FCM is, and a compact workflow strip.

Run: python fcm_explainer.py
Exports PNG and editable SVG figures to FCM/graphics/.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from fcm_presentation import GROUPS, NEGATIVE, POSITIVE, ROOT, save_figure

INK = "#243443"
NAVY = "#245b96"

# (source, target) excerpts from the nutrient FCM, shown in the explainer map.
EXAMPLE_EDGES = [
    ("Air Temp Max", "Water Temp"),
    ("Water Temp", "ODO (mg/L)"),
    ("Water Temp", "DIN"),
    ("DIN", "Secchi Depth"),
]
NODES = {
    "Air Temp Max": ("Air temperature\n(max)", (1.7, 6.4), "forcing"),
    "Water Temp": ("Water\ntemperature", (5.4, 6.4), "physical"),
    "ODO (mg/L)": ("Dissolved\noxygen", (9.1, 6.4), "physical"),
    "DIN": ("Dissolved inorganic\nnitrogen (DIN)", (5.4, 3.2), "ecology"),
    "Secchi Depth": ("Water clarity\n(Secchi depth)", (9.1, 3.2), "ecology"),
}
HALF_W, HALF_H = 1.15, 0.55


def draw_node(ax, label, xy, group, fontsize=12.5):
    fill, border = GROUPS[group]
    ax.add_patch(FancyBboxPatch(
        (xy[0] - HALF_W, xy[1] - HALF_H), 2 * HALF_W, 2 * HALF_H,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        fc=fill, ec=border, lw=2, zorder=4))
    ax.text(*xy, label, ha="center", va="center", fontsize=fontsize,
            weight="bold", color=INK, zorder=5)


def badge(ax, number, xy):
    ax.add_patch(Circle(xy, 0.27, fc=INK, ec="white", lw=1.5, zorder=8))
    ax.text(*xy, str(number), ha="center", va="center", color="white",
            fontsize=13, weight="bold", zorder=9)


def edge_endpoints(source, target):
    start = np.array(NODES[source][1])
    end = np.array(NODES[target][1])
    direction = (end - start) / np.linalg.norm(end - start)
    # Stop at the box edge along the dominant axis, plus a small gap.
    horizontal = abs(direction[0]) > abs(direction[1])
    offset = (HALF_W if horizontal else HALF_H) + 0.12
    return start + direction * offset, end - direction * offset


def plot_what_is_fcm() -> None:
    weights = pd.read_csv(ROOT / "analysis" / "fcm_weights_nutrient.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")

    ax.text(8, 8.5, "What is a Fuzzy Cognitive Map (FCM)?", ha="center",
            fontsize=25, weight="bold", color=INK)
    ax.text(8, 7.95, "A map of variables where signed arrows show how a change in one "
            "tends to carry over to another", ha="center", fontsize=14, color="#52606c")

    for key, (label, xy, group) in NODES.items():
        draw_node(ax, label, xy, group)

    for source, target in EXAMPLE_EDGES:
        weight = float(weights.loc[source, target])
        color = POSITIVE if weight > 0 else NEGATIVE
        start, end = edge_endpoints(source, target)
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=26,
            lw=1.5 + abs(weight) * 4, ls="-" if weight > 0 else "--",
            color=color, zorder=2))
        mid = (start + end) / 2
        vertical = abs(end[1] - start[1]) > abs(end[0] - start[0])
        # Horizontal arrows are short, so their weight sits above the shaft.
        label_xy = mid if vertical else mid + np.array([0, 0.42])
        ax.text(*label_xy, f"{weight:+.2f}", ha="center", va="center", fontsize=14,
                weight="bold", color=color, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5})
        if (source, target) == ("Water Temp", "DIN"):
            vertical_mid = mid

    badge(ax, 1, (NODES["Air Temp Max"][1][0] - HALF_W + 0.05, NODES["Air Temp Max"][1][1] + HALF_H + 0.1))
    badge(ax, 2, (vertical_mid[0] + 0.5, vertical_mid[1] + 0.6))
    badge(ax, 3, (vertical_mid[0] + 0.7, vertical_mid[1]))

    legend_y = 4.3
    ax.text(0.7, legend_y + 0.55, "Reading the colours", fontsize=12, weight="bold", color=INK)
    for i, (color, ls, text) in enumerate([
            (POSITIVE, "-", "Positive: tend to rise together"),
            (NEGATIVE, "--", "Negative: one up, other down")]):
        yy = legend_y - 0.1 - 0.5 * i
        ax.plot([0.7, 1.5], [yy, yy], color=color, lw=3, ls=ls)
        ax.text(1.65, yy, text, fontsize=10.5, va="center", color=INK)
    ax.text(0.7, legend_y - 1.2, "Thicker line = stronger link", fontsize=10.5, color=INK)

    ax.text(5.4, 1.75, "Example excerpt from our nutrient FCM (learned weights)",
            ha="center", fontsize=12, style="italic", color="#52606c")

    # How-to-read panel
    items = [
        ("Concept (node)", "One measured variable: a forcing such as weather, "
         "a physical sensor reading, or a nutrient / ecology measure."),
        ("Arrow", "Direction of influence over time: when the source is high "
         "now, the target tends to be higher (solid +) or lower (dashed \u2212) next step."),
        ("Weight", "Strength and sign of that link, from \u22121 to +1. "
         "It is a relative strength, not a physical unit."),
    ]
    y = 6.9
    for number, (head, body) in enumerate(items, start=1):
        badge(ax, number, (11.2, y))
        ax.text(11.65, y + 0.02, head, fontsize=15, weight="bold", color=INK, va="center")
        ax.text(11.65, y - 0.38, body, fontsize=11.5, color=INK, va="top", wrap=True,
                linespacing=1.45)
        y -= 1.95
    ax.plot([10.5, 10.5], [1.2, 7.4], color="#c9d3dd", lw=1.2)

    # Bottom takeaway band
    ax.add_patch(FancyBboxPatch((0.6, 0.55), 14.8, 0.75, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc="#f4f7fa", ec="#c9d3dd", lw=1.2))
    ax.text(8, 0.93, "What-if use: nudge one concept (e.g. warmer water) and follow the arrows "
            "to see which others tend to rise or fall.  Ours are learned from data, not set by experts.",
            ha="center", va="center", fontsize=12, color=INK)
    ax.text(8, 0.25, "Learned lagged associations, NOT established causal effects.",
            ha="center", fontsize=11, color=NEGATIVE)
    save_figure(fig, "01a_what_is_fcm")


def plot_workflow_mini() -> None:
    fig, ax = plt.subplots(figsize=(16, 4.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set(xlim=(0, 16), ylim=(0, 4.6))
    ax.axis("off")

    steps = [
        (1.9, "1  Data", ["Continuous sensors + weather", "(GNN fills gaps)", "Monthly field samples"]),
        (5.9, "2  Concepts", ["Physical FCM: 11", "Nutrient FCM: 16", "(adds nutrient / ecology)"]),
        (9.9, "3  Learn weights", ["This month \u2192 next month", "Lagged ridge regression", "Observed 1, imputed 0.5"]),
    ]
    for x, title, lines in steps:
        ax.add_patch(FancyBboxPatch((x - 1.7, 1.0), 3.4, 2.7, boxstyle="round,pad=0.06,rounding_size=0.2",
                                    fc="#e7f0fc", ec=NAVY, lw=1.8))
        ax.text(x, 3.3, title, ha="center", va="center", fontsize=14.5, weight="bold", color=INK)
        ax.text(x, 2.2, "\n".join(lines), ha="center", va="center", fontsize=11.5,
                color=INK, linespacing=1.7)
    for x in (3.9, 7.9, 12.0):
        ax.add_patch(FancyArrowPatch((x - 0.22, 2.35), (x + 0.22, 2.35), arrowstyle="-|>",
                                    mutation_scale=24, color=NAVY, lw=2))

    # Output box with a tiny signed network
    x0 = 14.1
    ax.add_patch(FancyBboxPatch((x0 - 1.7, 1.0), 3.4, 2.7, boxstyle="round,pad=0.06,rounding_size=0.2",
                                fc="#edf4e5", ec=GROUPS["ecology"][1], lw=1.8))
    ax.text(x0, 3.3, "4  Signed network", ha="center", va="center", fontsize=14.5, weight="bold", color=INK)
    pts = {"A": (x0 - 1.1, 2.05), "B": (x0, 2.65), "C": (x0 + 1.1, 2.05)}
    for src, dst, color, ls in [("A", "B", POSITIVE, "-"), ("B", "C", NEGATIVE, "--")]:
        start, end = np.array(pts[src]), np.array(pts[dst])
        direction = (end - start) / np.linalg.norm(end - start)
        ax.add_patch(FancyArrowPatch(start + direction * 0.22, end - direction * 0.22,
                                    arrowstyle="-|>", mutation_scale=16, lw=2.4, ls=ls, color=color))
    for xy in pts.values():
        ax.add_patch(Circle(xy, 0.2, fc="white", ec=INK, lw=1.6, zorder=4))
    ax.text(x0, 1.45, "Explore what-ifs +\ncheck against held-out data", ha="center",
            va="center", fontsize=11, color=INK, linespacing=1.5)
    ax.text(8, 0.45, "GNN reconstructs sensor values; FCM explores relationships. "
            "Neither step establishes causality.", ha="center", fontsize=12, color=NEGATIVE)
    save_figure(fig, "01b_fcm_workflow_mini")


def main() -> None:
    plot_what_is_fcm()
    plot_workflow_mini()


if __name__ == "__main__":
    main()
