"""Reproducible figures for the retrospective warning-policy evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def plot_detection_burden(burden: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('matplotlib is required for figures') from exc
    horizons = sorted(burden.horizon_days.unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.5 * len(horizons), 4.4), sharey=True, squeeze=False)
    for ax, h in zip(axes[0], horizons):
        b = burden[burden.horizon_days == h]
        curve = b[b.series == 'do_history_risk_ranked']
        ax.plot(curve.alerts_per_100, curve.detection, color='tab:blue',
                label='DO-history, risk-ranked in the evaluation sample\n(retrospective; not an implementable alert policy)')
        ax.plot([0, 100], [0, 1], ls=':', color='gray', label='alert at random')
        for model, color, marker, label in (('do_history', 'tab:blue', 'o', 'DO-history, chronological threshold'),
                                            ('clim_state', 'tab:orange', 's', 'state climatology, chronological threshold')):
            s = summary[(summary.horizon_days == h) & (summary.model == model)].sort_values('cost_ratio')
            ax.scatter(s.alerts100, s.det_credited, color=color, marker=marker, zorder=3, label=label)
            for r in s.itertuples():
                ax.annotate(f'{int(r.cost_ratio)}:1', (r.alerts100, r.det_credited), textcoords='offset points',
                            xytext=(5, -11 if model == 'do_history' else 4), fontsize=8, color=color)
        n_ev = int(b.n_evaluable_onsets.iloc[0])
        n_days = int(b.n_issue_days.iloc[0])
        ax.set_title(f'{h}-day horizon ({n_ev} evaluable severe onsets, {n_days} issue-days)')
        ax.set_xlabel('Alert days per 100 policy-evaluable issue-days')
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel('Fraction of severe onsets preceded by a qualifying alert')
    axes[0][0].legend(fontsize=7.5, loc='lower right')
    fig.suptitle('Retrospective hindcast: onset detection vs alert burden (point labels = cost ratio FN:FP)', fontsize=9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description='Figures for trustworthy.warning_evaluation outputs')
    ap.add_argument('--eval-dir', type=Path, default=Path('trustworthy_outputs/warning_evaluation'))
    args = ap.parse_args()
    burden = pd.read_csv(args.eval_dir / 'detection_burden.csv')
    summary = pd.read_csv(args.eval_dir / 'episode_policy_summary.csv')
    out = args.eval_dir / 'figures' / 'detection_vs_burden.png'
    plot_detection_burden(burden, summary, out)
    print(f'Figure saved to {out}')


if __name__ == '__main__':
    main()
