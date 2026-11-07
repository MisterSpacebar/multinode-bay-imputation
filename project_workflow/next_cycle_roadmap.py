"""
next_cycle_roadmap.py
---------------------
Next-steps visual: five-step plan for forecasting low oxygen (hours/day below
4 mg/L), with a stop/go gate after the tabular baseline, the success
criterion, and data in hand vs. still open. Plan only; no results are shown.

Run from the repo root:
    python project_workflow/next_cycle_roadmap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("project_workflow")
OUT_DIR.mkdir(exist_ok=True)

NAVY = "#0d2b4e"
BLUE = "#1565c0"
TEXT = "#333333"
GATE = "#c62828"
DARK = {"#1b5e7a": "#12404f", "#2e7d32": "#1b5e20", "#e69500": "#8a5a00", "#78909c": "#455a64"}

# (title, what we do, done when, accent colour)
STEPS = [
    ("Define the target",
     "Hours/day below 4 mg/L\n(and 2 mg/L)\n\nHorizons: 1, 3, 7, 14 days",
     "Target and horizons\nfixed before modelling", "#1b5e7a"),
    ("Set the bar",
     "Rolling-origin splits\nObserved-only scoring\n\nBaselines: persistence,\n7-day mean, seasonal mean",
     "Baseline skill table\nexists", "#1b5e7a"),
    ("Tabular model first",
     "Lagged DO plus rain,\ntide, wind, air temp\n\nCheapest test of whether\nany signal is there",
     "Beats persistence,\nor we stop here", "#2e7d32"),
    ("GNN forecast head",
     "Horizon targets replace\nmask-and-reconstruct\n\nWith vs without\nFCM-derived features",
     "Ablation shows if the\nFCM adds anything", "#e69500"),
    ("Report",
     "Skill by horizon and node\n\nProbability of a\nhypoxic day\n\nBack-test 2021-22 (monthly)",
     "Held-out autumn,\nskill vs best baseline", "#78909c"),
]


def draw():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7.5, 7.65, "Next cycle: can we forecast low oxygen better than persistence?",
            ha="center", va="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(7.5, 7.2, "North bay  |  target: hours/day below 4 mg/L  |  plan only, nothing below has been run",
            ha="center", va="center", fontsize=10.5, color="#546e7a")

    card_w, gap, x0 = 2.6, 0.35, 0.3
    y0, card_h = 3.25, 3.4
    done_h = 0.85
    xs = [x0 + i * (card_w + gap) for i in range(len(STEPS))]

    for i, (title, what, done, colour) in enumerate(STEPS):
        x = xs[i]
        ax.add_patch(FancyBboxPatch((x, y0), card_w, card_h, boxstyle="round,pad=0.06",
                                    fc="white", ec=colour, lw=2.2))
        ax.add_patch(FancyBboxPatch((x, y0 + card_h - 0.75), card_w, 0.75,
                                    boxstyle="round,pad=0.06", fc=colour, ec=colour, lw=2.2))
        ax.add_patch(plt.Circle((x + 0.38, y0 + card_h - 0.375), 0.22, fc="white", ec="white", zorder=5))
        ax.text(x + 0.38, y0 + card_h - 0.375, str(i + 1), ha="center", va="center",
                fontsize=12, fontweight="bold", color=colour, zorder=6)
        ax.text(x + 0.75 + (card_w - 0.75) / 2, y0 + card_h - 0.375, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x + card_w / 2, y0 + done_h + (card_h - 0.75 - done_h) / 2, what,
                ha="center", va="center", fontsize=9.6, color=TEXT, linespacing=1.35)
        ax.plot([x + 0.15, x + card_w - 0.15], [y0 + done_h, y0 + done_h],
                color=colour, lw=1, ls=":")
        ax.text(x + card_w / 2, y0 + done_h / 2, "Done when:\n" + done, ha="center", va="center",
                fontsize=8.8, color=DARK[colour], fontweight="bold", linespacing=1.25)

        if i < len(STEPS) - 1:
            ax.add_patch(FancyArrowPatch((x + card_w + 0.08, y0 + card_h / 2),
                                         (xs[i + 1] - 0.08, y0 + card_h / 2),
                                         arrowstyle="-|>", mutation_scale=18, lw=2.2, color=NAVY))

    # Stop/go gate under the tabular step
    gx = xs[2] + card_w / 2
    ax.add_patch(FancyArrowPatch((gx, y0 - 0.08), (gx, y0 - 0.5), arrowstyle="-|>",
                                 mutation_scale=16, lw=2, color=GATE, ls="--"))
    ax.add_patch(FancyBboxPatch((xs[1] - 0.1, y0 - 1.0), card_w * 2 + gap + 0.6, 0.45,
                                boxstyle="round,pad=0.05", fc="#ffebee", ec=GATE, lw=1.8, ls="--"))
    ax.text(xs[1] - 0.1 + (card_w * 2 + gap + 0.6) / 2, y0 - 0.775,
            "STOP / GO: if a simple model can't beat persistence, stop here",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=GATE)

    # Bottom boxes
    by, bh = 0.1, 1.85
    ax.add_patch(FancyBboxPatch((0.3, by), 7.3, bh, boxstyle="round,pad=0.06",
                                fc="white", ec=BLUE, lw=2))
    ax.text(0.55, by + bh - 0.28, "Success criteria (set before running)", fontsize=11.5,
            fontweight="bold", color=BLUE, va="center")
    crit = [
        "Skill > 0 vs best of persistence / 7-day mean",
        "at 3+ horizons, on 2+ north-bay nodes, on a fully held-out autumn",
        "Observed-only scoring, no scaling leakage",
        "Holds without Jun-2026 sensor-change nodes, or flagged source-dependent",
    ]
    for j, line in enumerate(crit):
        bullet = "\u2022 " if j in (0, 2, 3) else "   "
        ax.text(0.55, by + bh - 0.62 - j * 0.28, bullet + line, fontsize=9.3, color=TEXT, va="center")

    ax.add_patch(FancyBboxPatch((7.8, by), 6.9, bh, boxstyle="round,pad=0.06",
                                fc="#f3e5f5", ec="#6a1b9a", lw=2))
    ax.text(8.05, by + bh - 0.28, "Data in hand vs. still open", fontsize=11.5,
            fontweight="bold", color="#6a1b9a", va="center")
    status = [
        ("\u2713", "7 sensor nodes (Mar 2025-Sep 2026), grab samples, rainfall"),
        ("\u2713", "Trained GNN and FCM ready to supply features"),
        ("\u25cb", "Tide, wind, water-management data: add if obtainable"),
        ("\u25cb", "Rule for the Jun 2026 sensor change (train/test)"),
        ("\u25cb", "One autumn per node: held-out test is a single season"),
    ]
    for j, (mark, line) in enumerate(status):
        ax.text(8.05, by + bh - 0.62 - j * 0.25, f"{mark} {line}", fontsize=9.3, color=TEXT, va="center")
    out = OUT_DIR / "next_cycle_roadmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    draw()



