"""Roadmap graphic (08): from imputation and mapping to a tested forecast.

Run: python fcm_roadmap.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fcm_presentation import GROUPS, NEGATIVE, save_figure

STEPS = [
    ("1", "Set the bar", "Simple forecasts\nto beat", "ecology"),
    ("2", "GNN forecast", "Predict DO ahead\nat each node", "ecology"),
    ("3", "Add FCM links", "Use them only\nif error drops", "ecology"),
    ("4", "Merge", "Only if 2 and 3\nbeat the bar", "forcing"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    ax.text(8, 8.0, "Next steps | test a forecast before merging",
            ha="center", va="center", fontsize=26, weight="bold")
    ax.text(8, 7.1, "Today the FCM does not beat 'same as last value', and the GNN is mixed against simple baselines.",
            ha="center", va="center", fontsize=16, color=NEGATIVE)

    w, h, y = 3.1, 2.6, 2.9
    xs = [0.5, 4.5, 8.5, 12.4]
    for x, (num, title, body, group) in zip(xs, STEPS):
        fill, edge = GROUPS[group]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=fill, edgecolor=edge, linewidth=2.5))
        ax.text(x + w / 2, y + h - 0.45, f"{num}. {title}", ha="center", va="top",
                fontsize=18, weight="bold", color="#243443")
        ax.text(x + w / 2, y + h - 1.35, body, ha="center", va="top", fontsize=15,
                color="#243443", linespacing=1.5)

    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + w + 0.1, y + h / 2), (x + 4.0 - 0.1, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=26, linewidth=2.5,
                                     color="#243443"))

    ax.text(8, 1.9, "Gate after steps 2 and 3: does it beat the simple forecast?  If not, stop and report.",
            ha="center", va="center", fontsize=16, weight="bold", color=NEGATIVE)
    ax.text(8, 0.9, "Would help most: more north-bay data, weather and tide inputs, a fixed event definition.",
            ha="center", va="center", fontsize=14, color="#555f69")

    save_figure(fig, "08_next_steps_roadmap")


if __name__ == "__main__":
    main()
