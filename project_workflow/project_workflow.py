"""
project_workflow.py
--------------------
Capstone project-lifecycle status diagram, updated to reflect the current
state of the multinode-bay-imputation project (GNN model development
complete, currently in the Validation stage, Interpretation/FCM next).

Mirrors the layout of the original 7-stage template:
  1 Project Framing -> 2 Data Acquisition -> 3 Data Preparation ->
  4 Exploration -> 5 Model Development -> 6 Validation -> 7 Interpretation

Run:
    python project_workflow.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path("project_workflow")
OUT_DIR.mkdir(exist_ok=True)

STATUS_COLOR = {
    "done": ("#e8f5e9", "#2e7d32", "#1b5e20"),      # fc, ec, text
    "in_progress": ("#fff3e0", "#f9a825", "#8a6100"),
    "next": ("#eceff1", "#78909c", "#37474f"),
}

STAGES = [
    (1, "Project Framing", "Explored available sensor\ndatasets, defined the problem", "done"),
    (2, "Data Acquisition", "2025 sleds, 2026 buoys,\ngrab samples, rainfall", "done"),
    (3, "Data Preparation", "Merge co-located platforms,\nresample, quarantine bad values", "done"),
    (4, "Exploration", "Coverage heatmap, channel &\nspatial importance", "done"),
    (5, "Model Development", "BayImputationGNN\n(GAT + GRU), trained", "in_progress"),
    (6, "Validation", "Baseline comparison, die-off\n& rainfall consistency checks", "in_progress"),
    (7, "Interpretation", "Findings, limitations,\nhanded off to FCM", "next"),
]

EVIDENCE = [
    "Artifact produced: BayImputationGNN checkpoint, coverage heatmap, attention weights",
    "Decision made: GAT+GRU chosen over simpler per-node models",
    "Issue found: GNN loses to linear interp on short random gaps; no long-gap benchmark yet",
    "Results obtained: GNN beats naive baselines; die-off signatures reproduce known event pattern",
]


def plot_workflow_status():
    print("Generating project workflow status diagram...")
    fig, ax = plt.subplots(figsize=(15, 7.8))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.3)
    ax.axis("off")

    box_w, box_h = 2.55, 1.55
    top_y = 6.0
    bot_y = 3.0
    top_xs = [2.9, 5.75, 8.6, 11.45]
    bot_xs = [11.45, 8.6, 5.75]

    # INPUT box
    ax.add_patch(FancyBboxPatch((0.1, top_y - 0.2), 2.3, box_h + 0.4,
                                 boxstyle="round,pad=0.08", fc="#0d2b4e", ec="#0d2b4e"))
    ax.text(1.25, top_y + 0.6, "INPUT\n[Need / Data]", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    def draw_stage(num, title, desc, status, x, y):
        fc, ec, tc = STATUS_COLOR[status]
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.08",
                                     fc=fc, ec=ec, lw=2))
        ax.add_patch(plt.Circle((x + 0.3, y + box_h + 0.18), 0.24, color=ec, zorder=5,
                                 ec="white", lw=1.5))
        ax.text(x + 0.3, y + box_h + 0.18, str(num), ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="white", zorder=6)
        ax.text(x + box_w / 2, y + box_h - 0.32, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=tc)
        ax.text(x + box_w / 2, y + 0.62, desc, ha="center", va="center",
                fontsize=8.7, color="#333333")

    for (num, title, desc, status), x in zip(STAGES[:4], top_xs):
        draw_stage(num, title, desc, status, x, top_y)
    for (num, title, desc, status), x in zip(STAGES[4:], bot_xs):
        draw_stage(num, title, desc, status, x, bot_y)

    # OUTPUT box
    ax.add_patch(FancyBboxPatch((0.1, bot_y - 0.2), 2.3, box_h + 0.4,
                                 boxstyle="round,pad=0.08", fc="#1b7a63", ec="#1b7a63"))
    ax.text(1.25, bot_y + 0.6, "OUTPUT\n[Insight / Product]", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    # Top row arrows (INPUT -> 1 -> 2 -> 3 -> 4)
    arrow_pts = [2.4, top_xs[0] + box_w, top_xs[1] + box_w, top_xs[2] + box_w]
    targets = [top_xs[0], top_xs[1], top_xs[2], top_xs[3]]
    for x1, x2 in zip(arrow_pts, targets):
        ax.add_patch(FancyArrowPatch((x1, top_y + box_h / 2), (x2, top_y + box_h / 2),
                                      arrowstyle="-|>", mutation_scale=20, linewidth=2.2, color="#1f4e79"))

    # Down arrow (4 -> 5)
    ax.add_patch(FancyArrowPatch((top_xs[3] + box_w / 2, top_y - 0.05),
                                  (bot_xs[0] + box_w / 2, bot_y + box_h + 0.35),
                                  arrowstyle="-|>", mutation_scale=24, linewidth=2.6, color="#1f4e79"))

    # Bottom row arrows (5 -> 6 -> 7 -> OUTPUT), right to left
    ax.add_patch(FancyArrowPatch((bot_xs[0], bot_y + box_h / 2), (bot_xs[1] + box_w, bot_y + box_h / 2),
                                  arrowstyle="-|>", mutation_scale=20, linewidth=2.2, color="#1f4e79"))
    ax.add_patch(FancyArrowPatch((bot_xs[1], bot_y + box_h / 2), (bot_xs[2] + box_w, bot_y + box_h / 2),
                                  arrowstyle="-|>", mutation_scale=20, linewidth=2.2, color="#1f4e79"))
    ax.add_patch(FancyArrowPatch((bot_xs[2], bot_y + box_h / 2), (2.4, bot_y + box_h / 2),
                                  arrowstyle="-|>", mutation_scale=20, linewidth=2.2, color="#1f4e79"))

    # Outer loop: new sensor data arriving over time re-enters at Data Acquisition,
    # not at the FCM/fine-tuning loop — Interpretation (7) sits directly below
    # Data Acquisition (2), so the branch runs straight up between the two rows
    stage7_cx = bot_xs[2] + box_w / 2
    ax.add_patch(FancyArrowPatch((stage7_cx, bot_y + box_h + 0.05), (stage7_cx, top_y - 0.05),
                                  arrowstyle="-|>", mutation_scale=20, linewidth=2.2,
                                  color="#6a1b9a", linestyle="--"))
    ax.text(stage7_cx + 0.2, (bot_y + box_h + top_y) / 2, "New sensor data\narrives over time →\nback to Data Acquisition",
            ha="left", va="center", fontsize=8.3, color="#6a1b9a", fontweight="bold")

    # Iteration loop note — both Model Development and Validation feed back in
    model_dev_cx = bot_xs[0] + box_w / 2
    validation_cx = bot_xs[1] + box_w / 2
    loop_cx = (model_dev_cx + validation_cx) / 2

    ax.add_patch(FancyArrowPatch((model_dev_cx, bot_y - 0.05), (model_dev_cx, 2.65),
                                  arrowstyle="-|>", mutation_scale=20, linewidth=2, color="#1f4e79"))
    ax.add_patch(FancyArrowPatch((validation_cx, bot_y - 0.05), (validation_cx, 2.65),
                                  arrowstyle="-|>", mutation_scale=20, linewidth=2, color="#1f4e79"))
    ax.text(loop_cx, 2.55, "Iteration loop", ha="center", fontsize=10, fontweight="bold", color="#1f4e79")
    ax.text(loop_cx, 2.15, "Next cycle: FCM causal analysis, alternative\nmodels, fine-tuning + long-gap benchmark",
            ha="center", fontsize=8.7, color="#444444")

    # Evidence box
    ax.add_patch(FancyBboxPatch((0.1, 0.15), 8.0, 1.75, boxstyle="round,pad=0.08",
                                 fc="white", ec="#1565c0", lw=1.6))
    ax.text(0.35, 1.68, "Evidence at the current stage (Validation):",
            fontsize=10.5, fontweight="bold", color="#1565c0")
    for i, line in enumerate(EVIDENCE):
        ax.text(0.35, 1.35 - i * 0.34, f"\u2022 {line}", fontsize=8.3, color="#222222")

    # Legend
    legend_y = 0.55
    for i, (label, key) in enumerate([("Completed", "done"), ("In progress", "in_progress"), ("Next step", "next")]):
        fc, ec, tc = STATUS_COLOR[key]
        lx = 9.2 + i * 2.1
        ax.scatter(lx, legend_y, s=180, color=ec, zorder=5)
        ax.text(lx + 0.25, legend_y, label, va="center", fontsize=9.5, color="#333333")

    ax.set_title("Capstone Project Status: Model Development Complete, Validation In Progress",
                 fontsize=15.5, fontweight="bold", pad=14)
    plt.tight_layout()
    path = OUT_DIR / "capstone_workflow_status.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


if __name__ == "__main__":
    plot_workflow_status()
