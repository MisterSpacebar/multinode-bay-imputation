"""Descriptive detection-versus-alert-burden curves for the warning hindcasts.

The DO-history curve ranks the evaluated issue-days by predicted risk *within the
evaluation sample* and alerts on the top q%. It is a retrospective description of
ranking quality, NOT an implementable alert policy (a real policy must fix its
threshold before seeing the outcomes; see ``trustworthy.warning_evaluation``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.warning_evaluation import POLICY_HORIZONS, build_metadata, load_decisions, write_json

DEFAULT_QS = tuple(range(1, 101))
CURVE_NOTE = 'descriptive risk-ranked curve in the evaluation sample; not an implementable policy'


def _detected(cases: pd.DataFrame, mask: pd.Series) -> int:
    hit = cases[mask & (cases.label == 1)]
    return int(hit.groupby(['node', 'first_severe_date']).ngroups)


def detection_burden(decisions: pd.DataFrame, qs=DEFAULT_QS, horizons=POLICY_HORIZONS) -> pd.DataFrame:
    """Onset detection when alerting on the top-q% of DO-history risks, plus climatology reference points."""
    rows = []
    for h in horizons:
        d = decisions[(decisions.horizon_days == h) & (decisions.model == 'do_history')
                      & (decisions.cost_ratio == decisions.cost_ratio.min())]
        d = d[d.chronological_available].sort_values(['probability', 'node', 'issue_date'],
                                                     ascending=[False, True, True]).reset_index(drop=True)
        if d.empty:
            continue
        n = len(d)
        onsets = int(d[d.label == 1].groupby(['node', 'first_severe_date']).ngroups)
        base = dict(horizon_days=h, n_issue_days=n, n_evaluable_onsets=onsets)
        rank = pd.Series(np.arange(1, n + 1), index=d.index)
        for q in qs:
            k = int(round(n * q / 100))
            rows.append(dict(base, series='do_history_risk_ranked', alert_pct=q, alerts_per_100=100 * k / n,
                             detection=_detected(d, rank <= k) / onsets if onsets else np.nan, note=CURVE_NOTE))
        moderate = d.origin_state.eq('Moderate')
        k_mod = int(round(n * moderate.mean()))
        rows.append(dict(base, series='clim_state_moderate_origin_only', alert_pct=np.nan,
                         alerts_per_100=100 * moderate.mean(),
                         detection=_detected(d, moderate) / onsets if onsets else np.nan,
                         note='alert every Moderate-origin day'))
        rows.append(dict(base, series='clim_state_every_day', alert_pct=100.0, alerts_per_100=100.0,
                         detection=_detected(d, pd.Series(True, index=d.index)) / onsets if onsets else np.nan,
                         note='alert every evaluable day'))
        rows.append(dict(base, series='do_history_at_moderate_only_burden', alert_pct=np.nan,
                         alerts_per_100=100 * k_mod / n,
                         detection=_detected(d, rank <= k_mod) / onsets if onsets else np.nan,
                         note=CURVE_NOTE + '; same alert count as Moderate-origin-only'))
    return pd.DataFrame(rows)


def run(eval_dir: Path) -> dict:
    eval_dir = Path(eval_dir)
    decisions_path = eval_dir / 'policy_decisions.csv'
    table = detection_burden(load_decisions(decisions_path))
    out = eval_dir / 'detection_burden.csv'
    table.to_csv(out, index=False)
    meta = build_metadata({'policy_decisions': decisions_path}, dict(qs='1..100', ranking='DO-history probability, ties by node/issue_date'),
                          {'rows': int(len(table))}, {'detection_burden.csv': out})
    write_json(eval_dir / 'detection_burden_metadata.json', meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description='Descriptive detection-vs-alert-burden curve')
    ap.add_argument('--eval-dir', type=Path, default=Path('trustworthy_outputs/warning_evaluation'))
    args = ap.parse_args()
    run(args.eval_dir)
    print(f'Outputs saved to {args.eval_dir}')


if __name__ == '__main__':
    main()
