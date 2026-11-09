"""Phase 3: censoring-aware daily hypoxia persistence, recovery and Markov baseline.

Inputs are the *observed-only* Phase 1 daily episodes and Phase 2 state/
transition tables. This module never creates missing observations, joins
reconstructions, or calls probabilistic forecasts an operational warning.

The Kaplan–Meier endpoint is the number of consecutively observed eligible
hypoxic calendar days before an OBSERVED next-day recovery. Last observed days
preceding unknown data / record end are right-censored. The discrete-time
hazard data include only episode days where the *next* calendar day is observed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from trustworthy.episode_transitions import STATES, SCHEMES, _station_block_cubes, _count_cube

SURVIVAL_COLUMNS = (
    'scope', 'node', 'threshold_mgL', 'duration_days', 'n_at_risk',
    'n_recoveries', 'n_censored', 'survival_probability', 'greenwood_variance',
    'greenwood_se', 'provenance',
)
SURVIVAL_SUMMARY_COLUMNS = (
    'scope', 'node', 'threshold_mgL', 'n_identifiable_onsets',
    'n_recovered', 'n_right_censored', 'observed_median_days',
    'km_median_days', 'km_median_reached', 'max_observed_days', 'provenance',
)
RISK_COLUMNS = (
    'episode_id', 'node', 'threshold_mgL', 'date', 'next_date',
    'elapsed_days', 'log_elapsed_days', 'hours_below_threshold',
    'exposure_fraction_24h', 'next_day_recovery', 'next_state',
    'qc_suspect', 'provenance',
)
HAZARD_COLUMNS = ('threshold_mgL', 'model', 'term', 'coefficient',
                  'relative_hazard', 'ci_lower', 'ci_upper', 'std_error',
                  'n_episode_days', 'n_episodes', 'n_recoveries',
                  'n_stations', 'converged', 'caution')
MARKOV_COLUMNS = ('block_scheme', 'metric', 'point_estimate', 'bootstrap_median',
                  'ci_lower', 'ci_upper', 'n_valid', 'n_bootstrap',
                  'n_station_blocks', 'seed', 'provenance')
MARKOV_METRICS = (
    'expected_consecutive_severe_days',
    'expected_consecutive_moderate_days',
    'expected_severe_to_normal_days',
    'p_severe_within_7_days_from_normal',
    'p_severe_within_7_days_from_moderate',
)
TRAJECTORY_COLUMNS = ('block_scheme', 'origin_state', 'horizon_days',
                      'n_trajectories', 'n_hit_normal', 'prob_hit_normal',
                      'median_days_to_normal_if_hit', 'n_censored_at_horizon',
                      'seed', 'provenance')


def _flags(s: pd.Series, name: str) -> pd.Series:
    mapped = s.astype('string').str.strip().str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False})
    if mapped.isna().any():
        raise ValueError(f'{name} has missing or non-boolean values')
    return mapped.astype(bool)


def _require_observed(df: pd.DataFrame, name: str) -> None:
    if 'provenance' not in df or df['provenance'].isna().any() or not df['provenance'].eq('observed_only').all():
        raise ValueError(f'{name} is missing explicit observed_only provenance')


def _episode_input(episodes: pd.DataFrame) -> pd.DataFrame:
    needed = {'episode_id', 'node', 'threshold_mgL', 'start_date_local',
              'last_hypoxic_date_local', 'calendar_duration_days',
              'identifiable_onset', 'censored_left', 'censored_right',
              'observed_recovery', 'provenance'}
    if not needed.issubset(episodes):
        raise ValueError(f'episode table missing fields {sorted(needed-set(episodes))}')
    _require_observed(episodes, 'episodes')
    e = episodes.copy()
    if e.duplicated('episode_id').any():
        raise ValueError('Duplicate episode_id')
    for key in ('identifiable_onset', 'censored_left', 'censored_right', 'observed_recovery'):
        e[key] = _flags(e[key], key)
    e['threshold_mgL'] = pd.to_numeric(e['threshold_mgL'], errors='raise')
    if not e.threshold_mgL.isin([2, 4]).all():
        raise ValueError('Only 2 and 4 mg/L episodes are supported')
    e['calendar_duration_days'] = pd.to_numeric(e['calendar_duration_days'], errors='raise')
    if (e.calendar_duration_days < 1).any() or not np.all(
            e.calendar_duration_days == e.calendar_duration_days.astype('int64')):
        raise ValueError('Episode calendar duration must be positive integer')
    for date_col in ('start_date_local', 'last_hypoxic_date_local'):
        e[date_col] = pd.to_datetime(e[date_col], errors='raise')
        if e[date_col].isna().any() or e[date_col].dt.tz is not None:
            raise ValueError(f'{date_col} is not a local calendar date')
        if not (e[date_col].dt.normalize() == e[date_col]).all():
            raise ValueError(f'{date_col} must be midnight calendar dates')
    actual_days = (e.last_hypoxic_date_local - e.start_date_local).dt.days + 1
    if not actual_days.eq(e.calendar_duration_days).all():
        raise ValueError('Reported episode duration and dates disagree')
    if ((e.identifiable_onset == e.censored_left)
        | (e.observed_recovery & e.censored_right)
        | (~e.observed_recovery & ~e.censored_right)).any():
        raise ValueError('Contradictory censoring/recovery flags')
    return e


def km_curves(episodes: pd.DataFrame, by_station: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KM survival conditional on *observed identifiable onsets*.

    Greenwood is an approximate sampling variance, not a correction for
    spatial/serial dependence or potentially informative censoring.
    """
    e = _episode_input(episodes)
    e = e.loc[e.identifiable_onset]
    groups = [('pooled', '', t, g) for t, g in e.groupby('threshold_mgL', sort=True)]
    if by_station:
        groups += [('station', n, t, g)
                   for (n, t), g in e.groupby(['node', 'threshold_mgL'], sort=True)]
    curves, summaries = [], []
    for scope, node, threshold, group in groups:
        durations = group.calendar_duration_days.to_numpy(dtype=int)
        event = group.observed_recovery.to_numpy(dtype=bool)
        surv, sum_greenwood = 1.0, 0.0
        for time in np.sort(np.unique(durations)):
            at_risk = int((durations >= time).sum())
            deaths = int(((durations == time) & event).sum())
            censored = int(((durations == time) & ~event).sum())
            if deaths:
                surv *= (1.0 - deaths / at_risk)
                if deaths < at_risk:
                    sum_greenwood += deaths / (at_risk * (at_risk - deaths))
                else:
                    # KM has reached zero; Greenwood log variance has a
                    # singular denominator. Store the well-defined zero limit.
                    sum_greenwood = np.nan
            variance = 0.0 if surv == 0 else (surv ** 2 * sum_greenwood)
            curves.append({
                'scope': scope, 'node': node, 'threshold_mgL': int(threshold),
                'duration_days': int(time), 'n_at_risk': at_risk,
                'n_recoveries': deaths, 'n_censored': censored,
                'survival_probability': float(surv),
                'greenwood_variance': float(variance),
                'greenwood_se': float(np.sqrt(variance)),
                'provenance': 'observed_only',
            })
        current = [r for r in curves if r['scope'] == scope and r['node'] == node
                   and r['threshold_mgL'] == int(threshold)]
        passed = [r['duration_days'] for r in current if r['survival_probability'] <= .5]
        summaries.append({
            'scope': scope, 'node': node, 'threshold_mgL': int(threshold),
            'n_identifiable_onsets': len(group),
            'n_recovered': int(event.sum()),
            'n_right_censored': int((~event).sum()),
            'observed_median_days': float(np.median(durations)),
            'km_median_days': float(min(passed)) if passed else np.nan,
            'km_median_reached': bool(passed),
            'max_observed_days': int(durations.max()),
            'provenance': 'observed_only',
        })
    return (pd.DataFrame(curves, columns=SURVIVAL_COLUMNS),
            pd.DataFrame(summaries, columns=SURVIVAL_SUMMARY_COLUMNS))


def recovery_risk_rows(episodes: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Observed next-day recovery risk sets; no gap crossing or leak-prone fill.

    Current-day severity is available by the end of that day, while the label
    is recovery on tomorrow's eligible day. Terminal unknown days are excluded
    from the hazard likelihood (but retained as right censoring in the KM).
    """
    e = _episode_input(episodes)
    e = e.loc[e.identifiable_onset].copy()
    needed = {'node', 'date', 'state', 'hours_lt2', 'hours_lt4', 'qc_suspect', 'provenance'}
    if not needed.issubset(states):
        raise ValueError(f'state table missing fields: {sorted(needed - set(states))}')
    _require_observed(states, 'station-day states')
    s = states.copy()
    if s.duplicated(['node', 'date']).any():
        raise ValueError('Duplicate state node/date')
    s['date'] = pd.to_datetime(s['date'], errors='raise')
    if s.date.isna().any() or s.date.dt.tz is not None:
        raise ValueError('State dates must be local calendar dates')
    s['qc_suspect'] = _flags(s.qc_suspect, 'qc_suspect')
    state_lookup = {(row.node, row.date): row
                    for row in s.itertuples(index=False)}
    rows = []
    for ep in e.itertuples(index=False):
        threshold = int(ep.threshold_mgL)
        d_start = ep.start_date_local
        d_final = ep.last_hypoxic_date_local
        expected = int(ep.calendar_duration_days)
        for j in range(expected):
            date = d_start + pd.Timedelta(days=j)
            nxt = date + pd.Timedelta(days=1)
            today = state_lookup.get((ep.node, date))
            if today is None or today.state not in STATES:
                raise ValueError(f'{ep.episode_id}: missing known event day {date}')
            hours = today.hours_lt2 if threshold == 2 else today.hours_lt4
            if not np.isfinite(hours) or not (0 < hours <= 24):
                raise ValueError(f'{ep.episode_id}: non-hypoxic event day {date}')
            following = state_lookup.get((ep.node, nxt))
            next_is_hypoxic = (following is not None
                and following.state in STATES
                and (following.state == 'Severe' if threshold == 2
                     else following.state in ('Moderate', 'Severe')))
            if j < expected-1 and not next_is_hypoxic:
                raise ValueError(f'{ep.episode_id}: run crosses unknown/nonhypoxic day')
            if j == expected-1:
                if ep.observed_recovery and (following is None or
                    following.state not in STATES or next_is_hypoxic):
                    raise ValueError(f'{ep.episode_id}: recovery does not match states')
                if not ep.observed_recovery:
                    # An unknown next day is *censored*, never a negative
                    # recovery label. That final day has no observed outcome.
                    if following is not None and following.state in STATES:
                        raise ValueError(f'{ep.episode_id}: censored end has known next day')
                    continue
            if following is None or following.state not in STATES:
                raise ValueError(f'{ep.episode_id}: unobserved next day in likelihood')
            if not next_is_hypoxic and not (j == expected - 1 and ep.observed_recovery):
                raise ValueError('Unexpected next-day nonhypoxic state')
            rows.append({
                'episode_id': ep.episode_id, 'node': ep.node,
                'threshold_mgL': threshold,
                'date': date.date().isoformat(), 'next_date': nxt.date().isoformat(),
                'elapsed_days': j + 1,
                'log_elapsed_days': float(np.log(j + 1)),
                'hours_below_threshold': float(hours),
                'exposure_fraction_24h': float(hours / 24),
                'next_day_recovery': bool(j == expected - 1 and ep.observed_recovery),
                'next_state': following.state,
                'qc_suspect': bool(today.qc_suspect),
                'provenance': 'observed_only',
            })
    return pd.DataFrame(rows, columns=RISK_COLUMNS)


def fit_recovery_hazard(risk: pd.DataFrame, station_effects: bool = False) -> pd.DataFrame:
    """CLogLog GLM with episode-clustered sandwich SE (statsmodels optional).

    Each row is a distinct eligible episode day with known *next-day* outcome.
    This baseline uses only observed exposure + elapsed duration. No forcing
    or GNN data. Resulting associations are descriptive, not causal.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError('Hazard GLM requires: pip install "statsmodels>=0.14,<0.16"') from exc
    from scipy.stats import norm
    rows = []
    for threshold, group in risk.groupby('threshold_mgL', sort=True):
        if group.next_day_recovery.sum() < 5 or group.episode_id.nunique() < 10:
            warnings.warn(f'{threshold} mg/L: fewer than 5 recoveries or 10 episodes; skipping GLM')
            continue
        if station_effects:
            counts = group.groupby('node').next_day_recovery.agg(['sum', 'count'])
            if (counts['sum'].eq(0) | counts['sum'].eq(counts['count'])).any():
                warnings.warn(f'{threshold} mg/L: at least one station has complete/quasi-separation; '
                              'skipping unstable station-fixed-effects fit')
                continue
        # Scale low-oxygen hours in 6-hour increments, interpretable without
        # post-hoc standardization or borrowing predictor values from tomorrow.
        X = pd.DataFrame({
            'intercept': np.ones(len(group)),
            'log_elapsed_days': group.log_elapsed_days.to_numpy(dtype=float),
            'hours_below_per_6h': group.hours_below_threshold.to_numpy(dtype=float) / 6,
        })
        if station_effects:
            dummies = pd.get_dummies(group.node.astype(str), prefix='node', drop_first=True,
                                     dtype=float).reset_index(drop=True)
            X = pd.concat([X, dummies], axis=1)
        y = group.next_day_recovery.astype(int).to_numpy()
        if np.linalg.matrix_rank(X.to_numpy()) < len(X.columns):
            warnings.warn(f'{threshold} mg/L: rank-deficient design; skipping GLM')
            continue
        try:
            model = sm.GLM(y, X, family=sm.families.Binomial(
                link=sm.families.links.CLogLog()))
            result = model.fit(maxiter=150, disp=False, cov_type='cluster',
                               cov_kwds={'groups': group.episode_id.to_numpy()})
            if not result.converged or not np.all(np.isfinite(result.params)):
                warnings.warn(f'{threshold} mg/L: model failed to converge')
                continue
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            warnings.warn(f'{threshold} mg/L: model fitting failed ({exc})')
            continue
        for term, coeff in result.params.items():
            se = float(result.bse[term])
            lo, hi = coeff - norm.ppf(.975)*se, coeff + norm.ppf(.975)*se
            rows.append({
                'threshold_mgL': int(threshold),
                'model': 'cloglog_station_FE' if station_effects else 'cloglog_pooled',
                'term': term, 'coefficient': float(coeff),
                'relative_hazard': float(np.exp(np.clip(coeff, -700, 700))),
                'ci_lower': float(np.exp(np.clip(lo, -700, 700))),
                'ci_upper': float(np.exp(np.clip(hi, -700, 700))),
                'std_error': se, 'n_episode_days': len(group),
                'n_episodes': int(group.episode_id.nunique()),
                'n_recoveries': int(y.sum()), 'n_stations': int(group.node.nunique()),
                'converged': bool(result.converged),
                'caution': 'descriptive association; episode-cluster sandwich only; station/time dependence may remain',
            })
    return pd.DataFrame(rows, columns=HAZARD_COLUMNS)


def _markov_metrics(counts: np.ndarray) -> dict[str, float]:
    """Exact conditional metrics under a *stationary* three-state Markov chain.

    Computing conditional means exactly for each bootstrap matrix reduces
    Monte Carlo trajectory noise; uncertainty comes from matrix resampling.
    """
    if counts.shape != (3, 3) or (counts < 0).any() or not np.isfinite(counts).all():
        raise ValueError('Invalid Markov counts')
    total = counts.sum(axis=1)
    values = dict.fromkeys(MARKOV_METRICS, np.nan)
    if (total == 0).any():
        return values
    p = counts / total[:, None]
    if p[2, 2] < 1:
        values['expected_consecutive_severe_days'] = float(1 / (1-p[2, 2]))
    if p[1, 1] < 1:
        values['expected_consecutive_moderate_days'] = float(1 / (1-p[1, 1]))
    # First-passage from Severe to Normal; states Moderate/Severe are transient.
    q = p[np.ix_([1, 2], [1, 2])]
    try:
        h = np.linalg.solve(np.eye(2)-q, np.ones(2))
        if np.all(h >= 0) and np.isfinite(h).all():
            values['expected_severe_to_normal_days'] = float(h[1])
    except np.linalg.LinAlgError:
        pass
    # First hit of Severe in the next seven transitions from N or M.
    not_severe = p[np.ix_([0, 1], [0, 1])]
    no_hit = np.linalg.matrix_power(not_severe, 7) @ np.ones(2)
    values['p_severe_within_7_days_from_normal'] = float(np.clip(1 - no_hit[0], 0, 1))
    values['p_severe_within_7_days_from_moderate'] = float(np.clip(1 - no_hit[1], 0, 1))
    return values


def markov_matrix_bootstrap(transitions: pd.DataFrame,
                            n_bootstrap: int = 2000,
                            schemes=SCHEMES, seed: int = 2026,
                            save_draws: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw full row-coherent matrices by station-stratified block resampling.

    Each source-date station/block is sampled as one cube. An observed pair is
    never split or spliced with another block to manufacture a new transition.
    The bootstrap does not fully preserve regional cross-station dependence.
    """
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive')
    if not {'node','date','from_state','to_state','provenance'}.issubset(transitions):
        raise ValueError('Missing observed transition input fields')
    _require_observed(transitions, 'transitions')
    x = transitions.copy()
    x['date'] = pd.to_datetime(x.date)
    if x.duplicated(['node','date']).any():
        raise ValueError('Multiple transitions from one station-day')
    if not x.from_state.isin(STATES).all() or not x.to_state.isin(STATES).all():
        raise ValueError('Invalid state transition')
    rng = np.random.default_rng(seed)
    point = _markov_metrics(_count_cube(x))
    summary, records = [], []
    for scheme in schemes:
        blocks = _station_block_cubes(x, scheme)
        pool = np.zeros((n_bootstrap, 3, 3), dtype=np.int64)
        nblocks = 0
        for node, cubes in sorted(blocks.items()):
            k = len(cubes)
            nblocks += k
            draws = rng.integers(0, k, size=(n_bootstrap, k))
            pool += cubes[draws].sum(axis=1)
        output = {metric: [] for metric in MARKOV_METRICS}
        for draw_id, matrix in enumerate(pool):
            vals = _markov_metrics(matrix)
            for metric in MARKOV_METRICS:
                output[metric].append(vals[metric])
                if save_draws:
                    records.append({'block_scheme': scheme, 'draw_id': draw_id,
                                    'metric': metric, 'value': vals[metric],
                                    'seed': seed, 'provenance': 'observed_only'})
        for metric, raw in output.items():
            arr = np.asarray(raw, dtype=float)
            finite = arr[np.isfinite(arr)]
            # Even high-repetition sparse data may fail to identify some
            # transition rows in the bootstrap sample.
            reliable = len(finite) >= int(np.ceil(.8*n_bootstrap))
            lo, mid, hi = (np.quantile(finite, [.025, .5, .975])
                           if reliable else (np.nan, np.nan, np.nan))
            summary.append({
                'block_scheme': scheme, 'metric': metric,
                'point_estimate': point[metric],
                'bootstrap_median': float(mid), 'ci_lower': float(lo),
                'ci_upper': float(hi), 'n_valid': len(finite),
                'n_bootstrap': n_bootstrap, 'n_station_blocks': nblocks,
                'seed': seed, 'provenance': 'observed_only',
            })
    return (pd.DataFrame(summary, columns=MARKOV_COLUMNS),
            pd.DataFrame(records, columns=['block_scheme','draw_id','metric','value','seed','provenance']))


def simulate_markov_paths(transitions: pd.DataFrame, n_bootstrap: int = 2000,
                          schemes=SCHEMES, seed: int = 2026,
                          n_paths_per_matrix: int = 10,
                          horizon_days: int = 120) -> pd.DataFrame:
    """Optional path simulation drawing complete block-bootstrap matrices.

    A Severe trajectory ends the first time it reaches Normal; censored
    trajectories are retained as censored, not interpreted as recovery.
    The resulting distribution includes *both* parameter and path stochasticity.
    """
    if n_paths_per_matrix < 1 or horizon_days < 1:
        raise ValueError('Paths and horizon must be positive')
    _require_observed(transitions, 'transitions')
    transitions = transitions.copy()
    transitions['date'] = pd.to_datetime(transitions['date'], errors='raise')
    if transitions['date'].isna().any() or transitions.duplicated(['node', 'date']).any():
        raise ValueError('Invalid or repeated transition source date')
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive')
    rng = np.random.default_rng(seed)
    output=[]
    for scheme in schemes:
        blocks=_station_block_cubes(transitions, scheme)
        matrices=np.zeros((n_bootstrap,3,3),dtype=np.int64)
        for node,cubes in sorted(blocks.items()):
            k=len(cubes)
            matrices+=cubes[rng.integers(0,k,size=(n_bootstrap,k))].sum(axis=1)
        total=matrices.sum(axis=2, keepdims=True)
        probs=np.zeros(matrices.shape,dtype=float)
        np.divide(matrices,total,out=probs,where=total>0)
        # Sample a path for each bootstrap matrix; a state row with no
        # transitions is nonidentifiable and excluded rather than invented.
        good=(total[:, :, 0]>0).all(axis=1)
        p=probs[good]
        if not len(p):
            for start in (1,2):
                output.append({'block_scheme':scheme,'origin_state':STATES[start],
                               'horizon_days':horizon_days,'n_trajectories':0,
                               'n_hit_normal':0,'prob_hit_normal':np.nan,
                               'median_days_to_normal_if_hit':np.nan,
                               'n_censored_at_horizon':0,'seed':seed,
                               'provenance':'observed_only'})
            continue
        for start in (1,2):
            n=len(p)*n_paths_per_matrix
            sims=np.repeat(p,n_paths_per_matrix,axis=0)
            state=np.full(n,start,dtype=np.int8)
            recovered=np.zeros(n,dtype=bool)
            time_to_recover=np.zeros(n,dtype=np.int32)
            rows=np.arange(n)
            for day in range(1,horizon_days+1):
                active=~recovered
                if not active.any():
                    break
                chosen=sims[rows[active],state[active]]
                u=rng.random(active.sum())
                state[active]=(u >= chosen[:,0]).astype(np.int8)+(u >= (chosen[:,0]+chosen[:,1])).astype(np.int8)
                just_recovered=(state==0)&active
                time_to_recover[just_recovered]=day
                recovered|=just_recovered
            times=time_to_recover[recovered]
            output.append({'block_scheme':scheme,'origin_state':STATES[start],
                           'horizon_days':horizon_days,'n_trajectories':n,
                           'n_hit_normal':int(recovered.sum()),
                           'prob_hit_normal':float(recovered.mean()),
                           'median_days_to_normal_if_hit':float(np.median(times)) if len(times) else np.nan,
                           'n_censored_at_horizon':int((~recovered).sum()),
                           'seed':seed,'provenance':'observed_only'})
    return pd.DataFrame(output,columns=TRAJECTORY_COLUMNS)


def make_figures(km: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; Phase 3 plots skipped')
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5))
    for th,g in km.loc[km.scope.eq('pooled')].groupby('threshold_mgL'):
        x=[0,*g.duration_days.tolist()]
        y=[1.,*g.survival_probability.tolist()]
        ax.step(x,y,where='post',label=f'DO < {th:g} mg/L')
        cens=g.loc[g.n_censored.gt(0)]
        if not cens.empty:
            ax.scatter(cens.duration_days, cens.survival_probability,
                       marker='+',s=45)
    ax.set(xlabel='Consecutive observed hypoxic days',
           ylabel='Probability unresolved (KM)',ylim=(0,1.05),
           title='Onset-defined hypoxia persistence; censored runs retained')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir/'episode_km.png',dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description='Observed-only episode survival, next-day recovery and Markov baseline')
    p.add_argument('--episodes',type=Path,default=Path('trustworthy_outputs/episode_dynamics/episode_table.csv'))
    p.add_argument('--states',type=Path,default=Path('trustworthy_outputs/episode_dynamics/station_day_states.csv'))
    p.add_argument('--transitions',type=Path,default=Path('trustworthy_outputs/episode_dynamics/observed_transitions.csv'))
    p.add_argument('--out-dir',type=Path,default=Path('trustworthy_outputs/episode_dynamics'))
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--blocks',nargs='+',choices=SCHEMES,default=list(SCHEMES))
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--station-effects',action='store_true',help='Additional station-fixed-effects sensitivity model (may be unstable)')
    p.add_argument('--skip-hazard',action='store_true',help='Skip optional statsmodels GLM fitting; still write risk table')
    p.add_argument('--save-markov-draws',action='store_true')
    p.add_argument('--path-simulations',type=int,default=10,metavar='N',help='Trajectories per bootstrap matrix; 0 disables path simulation')
    p.add_argument('--path-horizon',type=int,default=120)
    p.add_argument('--no-figures',action='store_true')
    args=p.parse_args()
    for f in [args.episodes,args.states,args.transitions]:
        if not f.exists():
            p.error(f'Missing {f}; regenerate Phase 1/2 outputs first')
    episodes=pd.read_csv(args.episodes)
    states=pd.read_csv(args.states)
    transitions=pd.read_csv(args.transitions)
    km,summary=km_curves(episodes)
    risk=recovery_risk_rows(episodes,states)
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    km.to_csv(out/'survival_curves.csv',index=False)
    summary.to_csv(out/'survival_summary.csv',index=False)
    risk.to_csv(out/'recovery_risk_days.csv',index=False)
    if not args.skip_hazard:
        fit_recovery_hazard(risk).to_csv(out/'recovery_hazard.csv',index=False)
        if args.station_effects:
            fit_recovery_hazard(risk,station_effects=True).to_csv(out/'recovery_hazard_station_FE.csv',index=False)
    markov,draws=markov_matrix_bootstrap(transitions,args.bootstrap,args.blocks,args.seed,args.save_markov_draws)
    markov.to_csv(out/'markov_monte_carlo_summary.csv',index=False)
    if args.save_markov_draws:
        draws.to_csv(out/'markov_matrix_bootstrap_draws.csv',index=False)
    if args.path_simulations:
        simulate_markov_paths(transitions,args.bootstrap,args.blocks,args.seed,
                              args.path_simulations,args.path_horizon).to_csv(
                                  out/'markov_path_simulation_summary.csv',index=False)
    if not args.no_figures:
        make_figures(km,summary,out/'figures')
    metadata={
        'inputs': {n:{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
                   for n,f in [('episodes',args.episodes),('states',args.states),('transitions',args.transitions)]},
        'bootstrap_replicates':args.bootstrap,'block_schemes':args.blocks,'seed':args.seed,
        'n_identifiable_onsets':len(_episode_input(episodes).loc[lambda d:d.identifiable_onset]),
        'n_recovery_risk_days':len(risk),'n_recoveries_risk':int(risk.next_day_recovery.sum()),
        'hazard_model':'cloglog duration + exposure, episode-cluster SE' if not args.skip_hazard else 'not fitted',
        'path_simulations_per_matrix':args.path_simulations,
        'path_simulation_horizon_days':args.path_horizon,
        'censoring_rule':'Unknown next day has no observed outcome, not a recovery failure',
        'markov_rule':'draw whole block-bootstrap matrices, never independent marginal probabilities',
        'model_scope':'descriptive; not a prospective forecast or causal inference',
        'provenance':'observed_only',
    }
    (out/'survival_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f'Phase 3: {len(summary.loc[summary.scope.eq("pooled")])} thresholds, '
          f'{len(risk)} observed episode risk-days; {int(risk.next_day_recovery.sum())} recoveries')
    print(f'Outputs saved to {out}')


if __name__=='__main__':
    main()