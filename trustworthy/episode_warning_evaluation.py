"""Episode-level effectiveness of the chronological warning policies.

Links each alert to the severe onset it precedes (via the window's ``first_severe_date``),
groups consecutive alert days into warning clusters, credits each cluster to its earliest
qualifying onset (one-to-one) and reports detection, lead time and warning burden.

Policies are the frozen chronological decisions exported by
``trustworthy.warning_evaluation`` (``policy_decisions.csv``); nothing is re-optimised
here, so uncertainty is conditional on those decisions. Retrospective only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.warning_evaluation import (
    COST_RATIOS, N_BOOT, POLICY_HORIZONS, SEED, build_metadata, input_paths, load_decisions, write_json)

POLICIES = ('do_history', 'clim_state')
SUMMARY_METRICS = ('det_credited', 'det_raw', 'alerts100', 'clusters100', 'false_frac',
                   'mean_cluster_days', 'repeat_frac', 'median_lead')
PAIRED_METRICS = ('det_credited', 'alerts100', 'clusters100', 'false_frac')
EXCL_NO_HINDCAST = 'not_in_hindcast'
EXCL_EARLY = 'policy_unavailable_early_fold'
EXCL_NO_ISSUE = 'no_candidate_issue_day'
# Explicit dtypes keep all-NA columns (no detections / no credited onset) stable when frames are concatenated.
DETAIL_DTYPES = {'first_alert_date': 'datetime64[ns]', 'lead_days_raw': 'float64', 'lead_days_credited': 'float64',
                 'n_alert_days': 'Int64', 'detected_raw': 'boolean', 'credited': 'boolean'}
CLUSTER_DTYPES = {'credited_onset': 'datetime64[ns]', 'start': 'datetime64[ns]', 'end': 'datetime64[ns]'}


def qualifying_onsets(episodes: pd.DataFrame) -> pd.DataFrame:
    """Strictly-below-2 mg/L onsets that are identifiable and not left-censored."""
    e = episodes[(episodes.threshold_mgL == 2) & episodes.identifiable_onset.astype(bool)
                 & ~episodes.censored_left.astype(bool)].copy()
    e['onset'] = pd.to_datetime(e.start_date_local)
    out = e[['node', 'onset', 'qc_suspect']].copy()
    out['qc_suspect'] = out.qc_suspect.astype(bool)
    if out.duplicated(['node', 'onset']).any():
        raise ValueError('duplicate station-onset pairs in the episode table')
    return out.reset_index(drop=True)


def hindcast_cases(decisions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """All scored hindcast cases for a horizon, with the (model-independent) availability flag."""
    d = decisions[(decisions.horizon_days == horizon) & (decisions.model == 'do_history')
                  & (decisions.cost_ratio == decisions.cost_ratio.min())]
    return d[['node', 'issue_date', 'test_month', 'label', 'first_severe_date',
              'chronological_available']].rename(columns={'chronological_available': 'avail'}).reset_index(drop=True)


def policy_cases(decisions: pd.DataFrame, model: str, ratio: float, horizon: int) -> pd.DataFrame:
    d = decisions[(decisions.horizon_days == horizon) & (decisions.model == model) & (decisions.cost_ratio == ratio)]
    out = d[['node', 'issue_date', 'test_month', 'label', 'first_severe_date',
             'chronological_available', 'chronological_alert']].rename(
        columns={'chronological_available': 'avail', 'chronological_alert': 'alert'})
    return out.reset_index(drop=True)


def onset_accounting(onsets: pd.DataFrame, cases: pd.DataFrame, windows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Why each qualifying onset is (not) evaluable under the chronological policy at a horizon."""
    win = windows[windows.horizon_days == horizon].copy()
    win['issue_date'] = pd.to_datetime(win.issue_date)
    win['first_severe_date'] = pd.to_datetime(win.first_severe_date)
    rows = []
    for o in onsets.itertuples(index=False):
        has_window = ((win.node == o.node) & (win.first_severe_date == o.onset)).any()
        if has_window:
            c = cases[(cases.node == o.node) & (cases.label == 1) & (cases.first_severe_date == o.onset)]
            if c.empty:
                reason = EXCL_NO_HINDCAST
            elif not c.avail.any():
                reason = EXCL_EARLY
            else:
                reason = 'evaluable'
        else:
            cand = win[(win.node == o.node) & (win.issue_date >= o.onset - pd.Timedelta(days=horizon))
                       & (win.issue_date < o.onset)]
            reason = EXCL_NO_ISSUE if cand.empty else 'window_not_evaluable:' + '/'.join(sorted(cand.status.unique()))
        rows.append(dict(horizon_days=horizon, node=o.node, onset=o.onset, qc_suspect=o.qc_suspect, reason=reason))
    return pd.DataFrame(rows)


def match_policy(d: pd.DataFrame, onsets: pd.DataFrame):
    """Warning clusters and onset detection for one policy.

    ``d`` has one row per evaluable issue-day with a 0/1 ``alert``. A cluster is a run of
    consecutive calendar-day alerts at one station; Unknown/ineligible days or non-alert
    days end it. Each cluster is credited only to its earliest qualifying onset.
    """
    d = d[d.avail].copy()
    d['unit'] = d.node + '|' + d.test_month
    opp = (d[d.label == 1].groupby(['node', 'first_severe_date'])
           .agg(unit=('unit', 'first'), first_opportunity=('issue_date', 'min')).reset_index())
    opp = opp.merge(onsets, left_on=['node', 'first_severe_date'], right_on=['node', 'onset'],
                    how='left', validate='one_to_one')
    if opp.onset.isna().any():
        raise ValueError('a positive window points to an onset that is not a qualifying episode')
    clusters, credited, raw = [], {}, {}
    alerts = d[d.alert == 1].sort_values(['node', 'issue_date'])
    for node, g in alerts.groupby('node'):
        g = g.reset_index(drop=True)
        run_id = g.issue_date.diff().dt.days.fillna(1).ne(1).cumsum()
        for _, cl in g.groupby(run_id):
            hits = cl[cl.label == 1]
            rec = dict(node=node, unit=cl.unit.iloc[0], start=cl.issue_date.min(), end=cl.issue_date.max(),
                       n_days=len(cl), hit=len(hits) > 0, credited=False, credited_onset=pd.NaT)
            for onset, hh in hits.groupby('first_severe_date'):
                r = raw.setdefault((node, onset), dict(n_alert_days=0, first_alert=hh.issue_date.min()))
                r['n_alert_days'] += len(hh)
                r['first_alert'] = min(r['first_alert'], hh.issue_date.min())
            if len(hits):
                first = hits.first_severe_date.min()
                rec['credited_onset'] = first
                if (node, first) not in credited:
                    credited[(node, first)] = (first - hits[hits.first_severe_date == first].issue_date.min()).days
                    rec['credited'] = True
            clusters.append(rec)
    cl = pd.DataFrame(clusters, columns=['node', 'unit', 'start', 'end', 'n_days', 'hit', 'credited', 'credited_onset'])
    keys = list(zip(opp.node, opp.onset))
    opp['detected_raw'] = [k in raw for k in keys]
    opp['credited'] = [k in credited for k in keys]
    opp['first_alert_date'] = [raw[k]['first_alert'] if k in raw else pd.NaT for k in keys]
    opp['lead_days_raw'] = [(k[1] - raw[k]['first_alert']).days if k in raw else np.nan for k in keys]
    opp['lead_days_credited'] = [credited.get(k, np.nan) for k in keys]
    opp['n_alert_days'] = [raw[k]['n_alert_days'] if k in raw else 0 for k in keys]
    units = sorted(set(d.unit))
    n_cases = d.groupby('unit').size().reindex(units, fill_value=0).to_numpy()
    n_alerts = d.groupby('unit').alert.sum().reindex(units, fill_value=0).to_numpy()
    return units, n_cases, n_alerts, cl, opp


def tally(units, n_cases, n_alerts, cl, opp, exclude_qc: bool = False):
    o = opp[~opp.qc_suspect] if exclude_qc else opp
    ix = {u: i for i, u in enumerate(units)}
    n = len(units)
    T = {k: np.zeros(n) for k in ('cases', 'alerts', 'ev', 'raw', 'cred', 'rep', 'clu', 'false', 'cluster_days')}
    T['cases'][:] = n_cases
    T['alerts'][:] = n_alerts
    leads = [[] for _ in range(n)]
    for r in o.itertuples(index=False):
        i = ix[r.unit]
        T['ev'][i] += 1
        T['raw'][i] += r.detected_raw
        T['cred'][i] += r.credited
        T['rep'][i] += r.n_alert_days >= 2
        if r.credited:
            leads[i].append(r.lead_days_credited)
    for r in cl.itertuples(index=False):
        i = ix[r.unit]
        T['clu'][i] += 1
        T['false'][i] += not r.hit
        T['cluster_days'][i] += r.n_days
    return T, leads


def policy_stats(T, leads, pick) -> dict:
    s = lambda k: T[k][pick].sum()
    ev, clu, raw = s('ev'), s('clu'), s('raw')
    lead = np.concatenate([leads[i] for i in pick]) if len(pick) else np.array([])
    return dict(evaluable_onsets=ev,
                det_credited=s('cred') / ev if ev else np.nan, det_raw=raw / ev if ev else np.nan,
                alerts100=100 * s('alerts') / s('cases'), clusters100=100 * clu / s('cases'),
                false_frac=s('false') / clu if clu else np.nan,
                mean_cluster_days=s('cluster_days') / clu if clu else np.nan,
                repeat_frac=s('rep') / raw if raw else np.nan,
                median_lead=float(np.median(lead)) if len(lead) else np.nan)


def _ci(values) -> tuple[float, float]:
    a = np.array(values, dtype=float)
    if not np.isfinite(a).any():
        return float('nan'), float('nan')
    lo, hi = np.nanpercentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_episodes(decisions: pd.DataFrame, onsets: pd.DataFrame, windows: pd.DataFrame,
                      n_boot: int = N_BOOT, seed: int = SEED, ratios=COST_RATIOS, horizons=POLICY_HORIZONS):
    rng = np.random.default_rng(seed)
    accounting, detail, clusters, summary, paired = [], [], [], [], []
    for h in horizons:
        hc = hindcast_cases(decisions, h)
        if hc.empty:
            continue
        acc = onset_accounting(onsets, hc, windows, h)
        accounting.append(acc)
        for r in ratios:
            res = {}
            for model in POLICIES:
                d = policy_cases(decisions, model, r, h)
                units, n_cases, n_alerts, cl, opp = match_policy(d, onsets)
                res[model] = (units, *tally(units, n_cases, n_alerts, cl, opp), cl, opp, tally(units, n_cases, n_alerts, cl, opp, True))
                tag = dict(horizon_days=h, cost_ratio=r, model=model)
                det = acc.merge(opp[['node', 'onset', 'detected_raw', 'credited', 'first_alert_date', 'lead_days_raw',
                                     'lead_days_credited', 'n_alert_days']], on=['node', 'onset'], how='left')
                det = det.assign(evaluable=det.reason.eq('evaluable'), **tag).astype(DETAIL_DTYPES)
                detail.append(det)
                clusters.append(cl.assign(**tag).astype(CLUSTER_DTYPES))
            units = res['do_history'][0]
            n = len(units)
            picks = [rng.integers(0, n, n) for _ in range(n_boot)] if n_boot > 0 and n else []
            boots = {}
            for model in POLICIES:
                _, T, leads, cl, opp, (Tq, lq) = res[model]
                full = policy_stats(T, leads, np.arange(n))
                boot = [policy_stats(T, leads, p) for p in picks]
                boots[model] = (full, boot)
                lead_counts = pd.Series([x for l in leads for x in l]).value_counts()
                fq = policy_stats(Tq, lq, np.arange(n))
                row = dict(horizon_days=h, cost_ratio=r, model=model, n_units=n,
                           n_issue_days=int(T['cases'].sum()), n_onsets_qualifying=len(onsets),
                           evaluable_onsets=int(full['evaluable_onsets']),
                           evaluable_onsets_excl_qc=int(fq['evaluable_onsets']),
                           det_credited_excl_qc=fq['det_credited'],
                           **{f'lead_{k}d_credited': int(lead_counts.get(float(k), 0)) for k in (1, 2, 3)})
                for k in SUMMARY_METRICS:
                    row[k] = full[k]
                    row[f'{k}_lo'], row[f'{k}_hi'] = _ci([b[k] for b in boot]) if boot else (np.nan, np.nan)
                summary.append(row)
            fd, bd = boots['do_history']
            fc, bc = boots['clim_state']
            for k in PAIRED_METRICS:
                lo, hi = _ci([a[k] - b[k] for a, b in zip(bd, bc)]) if bd else (np.nan, np.nan)
                paired.append(dict(horizon_days=h, cost_ratio=r, metric=k, do_history_minus_clim_state=fd[k] - fc[k],
                                   lo=lo, hi=hi, interval_excludes_zero=bool(lo > 0 or hi < 0)))
    cat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return dict(onset_accounting=cat(accounting), episode_detection=cat(detail), warning_clusters=cat(clusters),
                episode_policy_summary=pd.DataFrame(summary), episode_policy_comparisons=pd.DataFrame(paired))


def run(root: Path, eval_dir: Path, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    paths = input_paths(root)
    eval_dir = Path(eval_dir)
    decisions_path = eval_dir / 'policy_decisions.csv'
    decisions = load_decisions(decisions_path)
    onsets = qualifying_onsets(pd.read_csv(paths['episodes']))
    windows = pd.read_csv(paths['windows'])
    tables = evaluate_episodes(decisions, onsets, windows, n_boot, seed)
    outputs = {}
    for name, frame in tables.items():
        path = eval_dir / f'{name}.csv'
        frame.to_csv(path, index=False)
        outputs[f'{name}.csv'] = path
    acc = tables['onset_accounting']
    counts = {'qualifying_onsets': int(len(onsets)), 'qc_suspect_onsets': int(onsets.qc_suspect.sum()),
              'accounting': {int(h): g.reason.value_counts().to_dict() for h, g in acc.groupby('horizon_days')}}
    params = dict(n_boot=n_boot, seed=seed, onset_definition='threshold 2 mg/L, identifiable_onset, not censored_left',
                  matching='consecutive-day alert clusters credited one-to-one to the earliest qualifying onset',
                  bootstrap_unit='station-month (frozen chronological policy)')
    meta = build_metadata({**{k: paths[k] for k in ('episodes', 'windows')}, 'policy_decisions': decisions_path},
                          params, counts, outputs)
    write_json(eval_dir / 'episode_evaluation_metadata.json', meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description='Episode-level detection, lead time and warning burden')
    ap.add_argument('--root', type=Path, default=Path('trustworthy_outputs'))
    ap.add_argument('--eval-dir', type=Path, default=Path('trustworthy_outputs/warning_evaluation'),
                    help='Directory containing policy_decisions.csv from trustworthy.warning_evaluation')
    ap.add_argument('--bootstrap', type=int, default=N_BOOT)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    meta = run(args.root, args.eval_dir, args.bootstrap, args.seed)
    print(f"Qualifying onsets {meta['counts']['qualifying_onsets']}; accounting {meta['counts']['accounting']}")
    print(f'Outputs saved to {args.eval_dir}')


if __name__ == '__main__':
    main()
