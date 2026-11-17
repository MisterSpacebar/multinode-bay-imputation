"""Retrospective evaluation of the observed-only transition-warning hindcasts.

Inputs are the existing out-of-sample hindcast artifacts (``hindcast_predictions.csv``,
``warning_windows.csv``). Nothing here refits a warning model. It adds:

* time-split references (origin-state and unconditional climatology) fitted only on
  windows whose outcome ended before the test month;
* held-out Brier/AUC comparisons with station-month cluster bootstrap intervals;
* chronological cost-sensitive alert thresholds chosen only from earlier
  out-of-sample predictions;
* sensitivity to the smoothing prior.

Everything is retrospective. It is not an operational forecast evaluation and makes
no causal claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.transition_warning import (
    ENV_FEATURES, markov_hit_probability, prepare_states, transition_matrix,
    validate_transitions)

MODELS = ('markov', 'do_history', 'environmental')
REFERENCES = ('clim_state', 'clim_uncond')
POLICY_MODELS = ('do_history', 'clim_state')
POLICY_HORIZONS = (1, 3)
ALL_HORIZONS = (1, 3, 7)
COST_RATIOS = (2, 5, 10)
PRIOR_STRENGTH = 20.0
PRIOR_GRID = (0, 5, 10, 20, 50, 100)
MIN_PRIOR_CASES = 50
MIN_PRIOR_CLASS = 5
N_BOOT = 2000
SEED = 2026
CASE_KEYS = ['node', 'issue_date', 'window_end', 'test_month', 'horizon_days',
             'origin_state', 'label', 'first_severe_date']
DECISION_COLUMNS = CASE_KEYS + [
    'model', 'cost_ratio', 'probability', 'theoretical_threshold', 'theoretical_alert',
    'chronological_threshold', 'chronological_available', 'chronological_alert']
LIMITATIONS = (
    'Retrospective hindcast evaluation only; no operational calibration or live validation.',
    'Overlapping windows and few station-months: intervals are wide and station results are not rankable.',
    'Bootstrap holds the chronological threshold policy fixed (no threshold-selection variability).',
    'Observed-only provenance; GAM/environmental associations are descriptive and non-causal.')


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_version() -> dict:
    """Git commit and dirty flag of the evaluation code, or 'unknown' outside a checkout."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', str(here)], cwd=here,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {'git_commit': commit, 'code_dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': 'unknown', 'code_dirty': None}


def input_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    return {
        'predictions': root / 'transition_warning' / 'hindcast_predictions.csv',
        'windows': root / 'transition_warning' / 'warning_windows.csv',
        'warning_metadata': root / 'transition_warning' / 'warning_metadata.json',
        'episodes': root / 'episode_dynamics' / 'episode_table.csv',
        'states': root / 'episode_dynamics' / 'station_day_states.csv',
        'transitions': root / 'episode_dynamics' / 'observed_transitions.csv'}


def month_start(month: str) -> pd.Timestamp:
    return pd.Period(month, freq='M').start_time


# ------------------------------------------------------------------ cases
def matched_cases(predictions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """One row per station-date-horizon scored by all three models, with its onset date."""
    need = {'node', 'issue_date', 'window_end', 'test_month', 'horizon_days', 'origin_state',
            'model', 'label', 'probability'}
    if not need <= set(predictions.columns):
        raise ValueError(f'predictions missing columns: {sorted(need - set(predictions.columns))}')
    p = predictions.copy()
    if set(p.model.unique()) != set(MODELS):
        raise ValueError(f'expected models {MODELS}, found {sorted(p.model.unique())}')
    if p.duplicated(['node', 'issue_date', 'horizon_days', 'model']).any():
        raise ValueError('duplicate station-date-horizon-model predictions')
    if not p.probability.between(0, 1).all() or not p.label.isin([0, 1]).all():
        raise ValueError('probabilities must be in [0,1] and labels binary')
    index = ['node', 'issue_date', 'window_end', 'test_month', 'horizon_days', 'origin_state', 'label']
    wide = p.pivot_table(index=index, columns='model', values='probability', aggfunc='first').reset_index()
    wide.columns.name = None
    wide = wide.dropna(subset=list(MODELS))
    wide['issue_date'] = pd.to_datetime(wide.issue_date)
    wide['window_end'] = pd.to_datetime(wide.window_end)
    w = windows[['node', 'issue_date', 'horizon_days', 'first_severe_date']].copy()
    w['issue_date'] = pd.to_datetime(w.issue_date)
    w['first_severe_date'] = pd.to_datetime(w.first_severe_date)
    wide = wide.merge(w, on=['node', 'issue_date', 'horizon_days'], how='left', validate='many_to_one')
    if wide.loc[wide.label == 1, 'first_severe_date'].isna().any():
        raise ValueError('positive hindcast case without a first_severe_date in warning_windows')
    return wide.sort_values(['node', 'issue_date', 'horizon_days']).reset_index(drop=True)


def complete_case_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Windows usable for training: evaluable, labelled and with finite issue-day features."""
    w = windows.copy()
    w['issue_date'] = pd.to_datetime(w.issue_date)
    w['window_end'] = pd.to_datetime(w.window_end)
    finite = np.isfinite(w[list(ENV_FEATURES)].apply(pd.to_numeric, errors='coerce')).all(axis=1)
    w['complete_case'] = w.status.eq('evaluable') & w.label.notna() & finite
    return w


def _nested_cummax(frame: pd.DataFrame, column: str) -> pd.Series:
    """Risk within 7 days cannot be below 3-day risk (the pipeline's own post-processing)."""
    key = frame[['node', 'issue_date', 'horizon_days']].assign(v=frame[column])
    key = key.sort_values(['node', 'issue_date', 'horizon_days'])
    return key.groupby(['node', 'issue_date']).v.cummax().reindex(frame.index)


def state_climatology(cases: pd.DataFrame, windows: pd.DataFrame, prior: float = PRIOR_STRENGTH):
    """Origin-state climatology shrunk toward the unconditional training rate.

    Training uses only complete-case windows with ``window_end`` before the test month
    starts, so 7-day outcomes crossing a month boundary never leak.
    Returns (state_prob, uncond_prob, n_train, normal_state_prob_by_fold).
    """
    if prior < 0:
        raise ValueError('prior must be nonnegative')
    w = complete_case_windows(windows) if 'complete_case' not in windows else windows
    state = pd.Series(np.nan, index=cases.index, dtype=float)
    uncond = state.copy()
    ntrain = state.copy()
    normal_by_fold = []
    for (month, h), test in cases.groupby(['test_month', 'horizon_days']):
        tr = w[w.complete_case & w.horizon_days.eq(h) & w.window_end.lt(month_start(month))]
        if tr.empty:
            continue
        p0 = tr.label.mean()
        by = tr.groupby('origin_state').label.agg(['sum', 'size'])
        sp = {s: ((by.loc[s, 'sum'] + prior * p0) / (by.loc[s, 'size'] + prior)) if s in by.index else p0
              for s in ('Normal', 'Moderate')}
        state.loc[test.index] = test.origin_state.map(sp)
        uncond.loc[test.index] = p0
        ntrain.loc[test.index] = len(tr)
        normal_by_fold.append((int(h), sp['Normal']))
    tmp = cases.assign(_s=state, _u=uncond)
    return _nested_cummax(tmp, '_s'), _nested_cummax(tmp, '_u'), ntrain, normal_by_fold


def with_references(cases: pd.DataFrame, windows: pd.DataFrame, prior: float = PRIOR_STRENGTH):
    """Attach held-out climatology references; cases without training history are dropped and counted."""
    s, u, n, _ = state_climatology(cases, windows, prior)
    out = cases.assign(clim_state=s, clim_uncond=u, n_train=n)
    kept = out.dropna(subset=['clim_state']).reset_index(drop=True)
    return kept, len(out) - len(kept)


# ------------------------------------------------------------ statistics
def auc(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def unit_index(frame: pd.DataFrame):
    """Station-month resampling units (overlapping windows make rows non-independent)."""
    keys = (frame.node + '|' + frame.test_month).to_numpy()
    uniq, inv = np.unique(keys, return_inverse=True)
    return [np.where(inv == i)[0] for i in range(len(uniq))]


def cluster_bootstrap(frame: pd.DataFrame, stat, n_boot: int, rng) -> tuple[float, float, int]:
    groups = unit_index(frame)
    if n_boot <= 0 or not groups:
        return float('nan'), float('nan'), len(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(stat(frame.iloc[np.concatenate([groups[i] for i in pick])]))
    lo, hi = np.nanpercentile(np.array(out, dtype=float), [2.5, 97.5])
    return float(lo), float(hi), len(groups)


def _brier(frame, col):
    return float(((frame[col] - frame.label) ** 2).mean())


def brier_tables(cases: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED):
    """Held-out Brier/AUC per model and paired Brier differences with clustered intervals."""
    rng = np.random.default_rng(seed)
    scores, comps = [], []
    names = list(REFERENCES) + list(MODELS)
    for h, d in cases.groupby('horizon_days'):
        y = d.label.to_numpy()
        for name in names:
            lo, hi, ncl = cluster_bootstrap(d, lambda x, c=name: _brier(x, c), n_boot, rng)
            alo, ahi, _ = cluster_bootstrap(d, lambda x, c=name: auc(x.label, x[c]), n_boot, rng)
            scores.append(dict(horizon_days=h, model=name, n=len(d), events=int(y.sum()),
                               prevalence=y.mean(), n_stations=d.node.nunique(), n_units=ncl,
                               brier=_brier(d, name), brier_lo=lo, brier_hi=hi,
                               auc=auc(y, d[name]), auc_lo=alo, auc_hi=ahi,
                               mean_predicted=float(d[name].mean())))
        pairs = [(r, m) for r in REFERENCES for m in MODELS] + [
            ('markov', 'do_history'), ('markov', 'environmental'), ('clim_uncond', 'clim_state')]
        for ref, m in pairs:
            f = lambda x, ref=ref, m=m: _brier(x, ref) - _brier(x, m)
            lo, hi, ncl = cluster_bootstrap(d, f, n_boot, rng)
            pt = f(d)
            comps.append(dict(horizon_days=h, reference=ref, model=m, n=len(d),
                              brier_reference_minus_model=pt, lo=lo, hi=hi,
                              skill_vs_reference=pt / _brier(d, ref),
                              interval_excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(scores), pd.DataFrame(comps)


# ----------------------------------------------------------- policies
def pick_threshold(p, y, ratio: float) -> float:
    """Minimise ratio*FN + FP for the rule 'alert if p >= t'; ties go to the highest t."""
    p = np.asarray(p, float)
    y = np.asarray(y)
    best_t, best_c = np.inf, np.inf
    for t in np.append(np.unique(p), np.inf):
        a = p >= t
        c = ratio * ((~a) & (y == 1)).sum() + (a & (y == 0)).sum()
        if c < best_c or (c == best_c and t > best_t):
            best_t, best_c = float(t), c
    return best_t


def policy_decisions(cases: pd.DataFrame, ratios=COST_RATIOS, horizons=POLICY_HORIZONS,
                     models=POLICY_MODELS, min_cases: int = MIN_PRIOR_CASES,
                     min_class: int = MIN_PRIOR_CLASS) -> pd.DataFrame:
    """Per case/model/cost-ratio alert decisions under theoretical and chronological thresholds.

    A chronological threshold for a test month uses only earlier out-of-sample cases
    (``window_end`` before the month starts) and is then frozen for that month. When
    history is insufficient the threshold is unavailable and the case is flagged.
    """
    frames = []
    for h in horizons:
        d = cases[cases.horizon_days == h]
        if d.empty:
            continue
        for model in models:
            for r in ratios:
                thr = {}
                for month in d.test_month.unique():
                    prior = d[d.window_end < month_start(month)]
                    ok = (len(prior) >= min_cases and prior.label.sum() >= min_class
                          and (prior.label == 0).sum() >= min_class)
                    thr[month] = pick_threshold(prior[model], prior.label, r) if ok else np.nan
                t = d.test_month.map(thr).astype(float)
                avail = t.notna()
                p = d[model]
                out = d[CASE_KEYS].copy()
                out['model'] = model
                out['cost_ratio'] = r
                out['probability'] = p
                out['theoretical_threshold'] = 1 / (1 + r)
                out['theoretical_alert'] = (p >= 1 / (1 + r)).astype(int)
                out['chronological_threshold'] = t
                out['chronological_available'] = avail
                out['chronological_alert'] = ((p >= t) & avail).astype(int)
                frames.append(out)
    if not frames:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    return pd.concat(frames, ignore_index=True)[DECISION_COLUMNS]


def load_decisions(path: Path) -> pd.DataFrame:
    """Read an exported ``policy_decisions.csv`` with dates and booleans restored."""
    d = pd.read_csv(path, parse_dates=['issue_date', 'window_end', 'first_severe_date'])
    d['chronological_available'] = d.chronological_available.astype(bool)
    missing = set(DECISION_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError(f'policy decisions missing columns: {sorted(missing)}')
    return d


def cost_per_100(frame: pd.DataFrame, col: str, ratio: float) -> float:
    a = frame[col].to_numpy().astype(bool)
    y = frame.label.to_numpy() == 1
    return float((ratio * ((~a) & y).sum() + (a & ~y).sum()) / len(frame) * 100)


def alert_metrics(frame: pd.DataFrame, col: str, ratio: float) -> dict:
    a = frame[col].to_numpy().astype(bool)
    y = frame.label.to_numpy() == 1
    tp, fp, fn, tn = (a & y).sum(), (a & ~y).sum(), ((~a) & y).sum(), ((~a) & ~y).sum()
    return dict(alerts_per_100=float(a.mean() * 100), recall=float(tp / max(tp + fn, 1)),
                precision=float(tp / (tp + fp)) if tp + fp else float('nan'),
                false_positive_rate=float(fp / max(fp + tn, 1)),
                false_alarm_fraction=float(fp / (tp + fp)) if tp + fp else float('nan'),
                cost_per_100=cost_per_100(frame, col, ratio))


def policy_frame(decisions: pd.DataFrame, horizon: int, ratio: float) -> pd.DataFrame:
    """Evaluated cases for one horizon/ratio with a 0/1 column per alert policy."""
    d = decisions[(decisions.horizon_days == horizon) & (decisions.cost_ratio == ratio)]
    keys = ['node', 'issue_date', 'horizon_days']
    base = d[d.model == 'do_history'][CASE_KEYS + ['chronological_available']]
    base = base[base.chronological_available].drop(columns='chronological_available').copy()
    for model, tag in (('do_history', 'do'), ('clim_state', 'clim')):
        m = d[d.model == model][keys + ['theoretical_alert', 'chronological_alert']].rename(
            columns={'theoretical_alert': f'{tag}_theory', 'chronological_alert': f'{tag}_chrono'})
        base = base.merge(m, on=keys, how='left', validate='one_to_one')
    base['never'] = 0
    base['always'] = 1
    return base.reset_index(drop=True)


POLICY_LABELS = {'never': 'never alert', 'always': 'always alert',
                 'clim_theory': 'state climatology @theoretical', 'clim_chrono': 'state climatology @chronological',
                 'do_theory': 'DO-history @theoretical', 'do_chrono': 'DO-history @chronological'}
COMPARISONS = (('never', 'do_chrono'), ('always', 'do_chrono'), ('always', 'do_theory'),
               ('clim_chrono', 'do_chrono'), ('do_theory', 'do_chrono'))


def policy_tables(decisions: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED):
    """Metrics per alert policy and paired cost differences (positive = second policy cheaper)."""
    rng = np.random.default_rng(seed)
    rows, comps = [], []
    for h in sorted(decisions.horizon_days.unique()):
        for r in sorted(decisions.cost_ratio.unique()):
            s = policy_frame(decisions, h, r)
            if s.empty:
                continue
            for key, label in POLICY_LABELS.items():
                rows.append(dict(horizon_days=h, cost_ratio=r, policy=key, description=label,
                                 n_issue_days=len(s), n_events=int(s.label.sum()),
                                 theoretical_threshold=1 / (1 + r), **alert_metrics(s, key, r)))
            for base, alt in COMPARISONS:
                f = lambda x, base=base, alt=alt, r=r: cost_per_100(x, base, r) - cost_per_100(x, alt, r)
                lo, hi, ncl = cluster_bootstrap(s, f, n_boot, rng)
                comps.append(dict(horizon_days=h, cost_ratio=r, baseline_policy=base, compared_policy=alt,
                                  n_issue_days=len(s), n_units=ncl, cost_saved_per_100=f(s), lo=lo, hi=hi,
                                  interval_excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows), pd.DataFrame(comps)


# ----------------------------------------------------- prior sensitivity
def prior_sensitivity(cases: pd.DataFrame, windows: pd.DataFrame, transitions: pd.DataFrame,
                      grid=PRIOR_GRID, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Brier sensitivity to (a) the climatology shrinkage prior and (b) the Markov station prior.

    The primary analysis stays at PRIOR_STRENGTH; these rows are sensitivity checks only.
    """
    rng = np.random.default_rng(seed)
    w = complete_case_windows(windows)
    rows = []
    for a in grid:
        col, _, _, normal_by_fold = state_climatology(cases, w, a)
        c2 = cases.assign(alt=col)
        for h, d in c2.dropna(subset=['alt']).groupby('horizon_days'):
            f = lambda x: _brier(x, 'alt') - _brier(x, 'do_history')
            lo, hi, _ = cluster_bootstrap(d, f, n_boot, rng)
            nps = [v for hh, v in normal_by_fold if hh == h]
            rows.append(dict(component='state_climatology_prior', prior=a, horizon_days=h, n=len(d),
                             brier=_brier(d, 'alt'), comparison='clim_minus_do_history',
                             difference=f(d), lo=lo, hi=hi,
                             normal_state_prob_min=min(nps) if nps else np.nan,
                             normal_state_prob_max=max(nps) if nps else np.nan))
    ref = cases[['clim_state']].rename(columns={'clim_state': 'ref'})
    for a in grid:
        pr = pd.Series(np.nan, index=cases.index, dtype=float)
        for month, mdf in cases.groupby('test_month'):
            hist = transitions.loc[transitions.next_date.lt(month_start(month))]
            mats = {n: transition_matrix(hist, n, a) for n in mdf.node.unique()}
            for i, r in zip(mdf.index, mdf.itertuples(index=False)):
                pr.loc[i] = markov_hit_probability(mats[r.node], r.origin_state, int(r.horizon_days))
        c2 = cases.join(ref).assign(alt=_nested_cummax(cases.assign(_m=pr), '_m'))
        for h, d in c2.dropna(subset=['alt', 'ref']).groupby('horizon_days'):
            f = lambda x: _brier(x, 'ref') - _brier(x, 'alt')
            lo, hi, _ = cluster_bootstrap(d, f, n_boot, rng)
            rows.append(dict(component='markov_station_prior', prior=a, horizon_days=h, n=len(d),
                             brier=_brier(d, 'alt'), comparison='clim_prior20_minus_markov',
                             difference=f(d), lo=lo, hi=hi,
                             normal_state_prob_min=np.nan, normal_state_prob_max=np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- metadata
def build_metadata(paths: dict[str, Path], params: dict, counts: dict, outputs: dict[str, Path] | None = None) -> dict:
    meta = {
        'inputs': {k: {'path': str(p), 'sha256': sha256_file(p)} for k, p in paths.items() if Path(p).exists()},
        'parameters': params,
        'counts': counts,
        'code': code_version(),
        'limitations': list(LIMITATIONS),
        'provenance': 'observed_only'}
    if outputs:
        meta['outputs'] = {k: {'path': str(p), 'sha256': sha256_file(p)} for k, p in outputs.items()}
    return meta


def write_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='utf-8')


def run(root: Path, out_dir: Path, n_boot: int = N_BOOT, seed: int = SEED, sensitivity: bool = True,
        min_cases: int = MIN_PRIOR_CASES, min_class: int = MIN_PRIOR_CLASS) -> dict:
    paths = input_paths(root)
    predictions = pd.read_csv(paths['predictions'])
    windows = pd.read_csv(paths['windows'])
    cases = matched_cases(predictions, windows)
    cases, n_dropped = with_references(cases, windows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    def emit(name, frame):
        path = out_dir / name
        frame.to_csv(path, index=False)
        outputs[name] = path

    emit('heldout_climatology.csv', cases)
    scores, comps = brier_tables(cases, n_boot, seed)
    emit('brier_scores.csv', scores)
    emit('brier_comparisons.csv', comps)
    decisions = policy_decisions(cases, min_cases=min_cases, min_class=min_class)
    emit('policy_decisions.csv', decisions)
    pscores, pcomps = policy_tables(decisions, n_boot, seed)
    emit('policy_scores.csv', pscores)
    emit('policy_comparisons.csv', pcomps)
    if sensitivity:
        states = prepare_states(pd.read_csv(paths['states']))
        trans = validate_transitions(pd.read_csv(paths['transitions']), states)
        emit('prior_sensitivity.csv', prior_sensitivity(cases, windows, trans, n_boot=n_boot, seed=seed))
    counts = {
        'matched_cases': int(len(cases)),
        'matched_cases_by_horizon': {int(h): int(n) for h, n in cases.groupby('horizon_days').size().items()},
        'positive_matched_cases': int(cases.label.sum()),
        'cases_without_training_reference': int(n_dropped),
        'policy_cases_by_horizon': {
            int(h): int(len(policy_frame(decisions, h, COST_RATIOS[0]))) for h in sorted(decisions.horizon_days.unique())},
        'policy_excluded_early_fold_cases_by_horizon': {
            int(h): int(len(decisions[(decisions.horizon_days == h) & (decisions.cost_ratio == COST_RATIOS[0])
                                       & (decisions.model == 'do_history')
                                       & ~decisions.chronological_available]))
            for h in sorted(decisions.horizon_days.unique())}}
    params = dict(n_boot=n_boot, seed=seed, prior_strength=PRIOR_STRENGTH, prior_grid=list(PRIOR_GRID),
                  cost_ratios=list(COST_RATIOS), policy_horizons=list(POLICY_HORIZONS),
                  min_prior_cases=min_cases, min_prior_class=min_class,
                  bootstrap_unit='station-month', alert_rule='alert if probability >= threshold')
    meta = build_metadata(paths, params, counts, outputs)
    write_json(out_dir / 'evaluation_metadata.json', meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description='Held-out climatology comparison and chronological cost-sensitive warning policies')
    ap.add_argument('--root', type=Path, default=Path('trustworthy_outputs'))
    ap.add_argument('--out-dir', type=Path, default=Path('trustworthy_outputs/warning_evaluation'))
    ap.add_argument('--bootstrap', type=int, default=N_BOOT)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--skip-prior-sensitivity', action='store_true')
    ap.add_argument('--min-prior-cases', type=int, default=MIN_PRIOR_CASES,
                    help='Earlier out-of-sample cases required before a chronological threshold is chosen')
    ap.add_argument('--min-prior-class', type=int, default=MIN_PRIOR_CLASS,
                    help='Earlier events and non-events each required before a threshold is chosen')
    args = ap.parse_args()
    meta = run(args.root, args.out_dir, args.bootstrap, args.seed, not args.skip_prior_sensitivity,
               args.min_prior_cases, args.min_prior_class)
    c = meta['counts']
    print(f"Matched cases {c['matched_cases']} ({c['matched_cases_by_horizon']}); policy cases {c['policy_cases_by_horizon']}")
    print(f'Outputs saved to {args.out_dir}')


if __name__ == '__main__':
    main()
