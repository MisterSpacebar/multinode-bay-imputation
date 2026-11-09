"""Observed-only three-state daily hypoxia transitions and block uncertainty.

Unknown station-days are never normal. Adjacent transitions require two
eligible *consecutive local dates*. This is descriptive, not forecasting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("Normal", "Moderate", "Severe")
SEASONS = ("DJF", "MAM", "JJA", "SON")
SCHEMES = ("14day", "28day", "month")
STATE_COLUMNS = ("node", "date", "state", "coverage", "hours_lt2", "hours_lt4",
                 "qc_suspect", "provenance")
TRANSITION_COLUMNS = ("node", "date", "next_date", "from_state", "to_state",
                      "season", "qc_suspect_pair", "provenance")
MATRIX_COLUMNS = ("from_state", "to_state", "n_transitions", "row_total",
                  "probability")
BOOT_COLUMNS = ("scope", "node", "block_scheme", "from_state", "to_state",
                "probability", "ci_lower", "ci_upper", "n_boot_valid",
                "n_bootstrap", "n_station_blocks", "row_total", "seed")
ONSET_BOOT_COLUMNS = ("scope", "node", "block_scheme", "threshold_mgL",
                      "observed_onsets", "bootstrap_median", "ci_lower", "ci_upper",
                      "n_bootstrap", "n_station_blocks", "seed")
_STATE_TO_I = {name: i for i, name in enumerate(STATES)}


def _flags(x: pd.Series, field: str) -> pd.Series:
    mapped = x.astype('string').str.strip().str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False})
    if mapped[x.notna()].isna().any():
        raise ValueError(f"Invalid {field} flag")
    return mapped.astype('boolean').fillna(False).astype(bool)


def _season(date: pd.Series) -> pd.Series:
    return date.dt.month.map({1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM',
                              5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA',
                              9: 'SON', 10: 'SON', 11: 'SON', 12: 'DJF'})


def daily_states(regimes: pd.DataFrame, min_coverage: float = .95) -> pd.DataFrame:
    """Derive N/M/S/Unknown from both independently eligible thresholds.

    A sample equal to 2 is Moderate if below 4; equal to 4 is Normal.
    Do not use retrospective `regime` strings to infer current oxygen state.
    """
    required = {'node', 'date', 'threshold_mgL', 'coverage', 'hours_below', 'eligible'}
    if not required.issubset(regimes.columns):
        raise ValueError(f"Missing regime columns: {sorted(required-set(regimes.columns))}")
    if not 0 < min_coverage <= 1:
        raise ValueError('min_coverage must be in (0,1]')
    d = regimes.copy()
    if d['node'].isna().any():
        raise ValueError('Station identity is missing')
    d['node'] = d['node'].astype(str)
    d['date'] = pd.to_datetime(d['date'], errors='raise')
    if d['date'].isna().any() or d['date'].dt.tz is not None or (d['date'].dt.normalize() != d['date']).any():
        raise ValueError('Dates must be timezone-naive local calendar days')
    d['threshold_mgL'] = pd.to_numeric(d['threshold_mgL'], errors='raise')
    if not set(d['threshold_mgL'].unique()) == {2, 4}:
        raise ValueError('Exactly the 2 and 4 mg/L threshold series are required')
    if d.duplicated(['node', 'date', 'threshold_mgL']).any():
        raise ValueError('Duplicate node/date/threshold input')
    d['coverage'] = pd.to_numeric(d['coverage'], errors='coerce')
    d['hours_below'] = pd.to_numeric(d['hours_below'], errors='coerce')
    d['eligible'] = _flags(d['eligible'], 'eligible')
    d['qc_suspect'] = (_flags(d['qc_suspect'], 'qc_suspect')
                       if 'qc_suspect' in d else False)
    left = d.loc[d.threshold_mgL.eq(2), ['node', 'date', 'coverage', 'hours_below', 'eligible', 'qc_suspect']]
    right = d.loc[d.threshold_mgL.eq(4), ['node', 'date', 'coverage', 'hours_below', 'eligible', 'qc_suspect']]
    merged = left.merge(right, on=['node', 'date'], how='outer', suffixes=('_2', '_4'), validate='one_to_one')
    records = []
    for node, g in merged.groupby('node', sort=True):
        g = g.set_index('date').sort_index().reindex(pd.date_range(g.date.min(), g.date.max(), freq='D'))
        valid_2 = (_flags(g['eligible_2'], 'eligible_2') & g.coverage_2.ge(min_coverage)
                   & g.hours_below_2.notna() & np.isfinite(g.hours_below_2)
                   & g.hours_below_2.between(0, 24))
        valid_4 = (_flags(g['eligible_4'], 'eligible_4') & g.coverage_4.ge(min_coverage)
                   & g.hours_below_4.notna() & np.isfinite(g.hours_below_4)
                   & g.hours_below_4.between(0, 24))
        valid = valid_2 & valid_4
        if (valid & g.hours_below_2.gt(g.hours_below_4 + 1e-8)).any():
            raise ValueError(f'{node}: hours below 2 exceed hours below 4')
        states = np.full(len(g), 'Unknown', dtype=object)
        states[valid & g.hours_below_4.eq(0)] = 'Normal'
        states[valid & g.hours_below_4.gt(0) & g.hours_below_2.eq(0)] = 'Moderate'
        states[valid & g.hours_below_2.gt(0)] = 'Severe'
        qc = (_flags(g['qc_suspect_2'], 'qc_suspect_2')
              | _flags(g['qc_suspect_4'], 'qc_suspect_4'))
        records.append(pd.DataFrame({
            'node': node, 'date': g.index, 'state': states,
            'coverage': np.minimum(g.coverage_2, g.coverage_4),
            'hours_lt2': g.hours_below_2.where(valid),
            'hours_lt4': g.hours_below_4.where(valid),
            'qc_suspect': qc.to_numpy(bool), 'provenance': 'observed_only',
        }))
    return (pd.concat(records, ignore_index=True)[list(STATE_COLUMNS)]
            if records else pd.DataFrame(columns=STATE_COLUMNS))


def observed_transitions(states: pd.DataFrame) -> pd.DataFrame:
    """A transition is a real adjacent known-day pair at the same station."""
    if states.empty:
        return pd.DataFrame(columns=TRANSITION_COLUMNS)
    if states.duplicated(['node', 'date']).any():
        raise ValueError('Duplicate station/date states')
    x = states.sort_values(['node', 'date']).copy()
    for key in ('date', 'state', 'qc_suspect'):
        x['next_' + key] = x.groupby('node', sort=False)[key].shift(-1)
    ok = (x['next_date'].sub(x['date']).eq(pd.Timedelta(days=1))
          & x.state.isin(STATES) & x.next_state.isin(STATES))
    x = x.loc[ok]
    return pd.DataFrame({
        'node': x.node.to_numpy(), 'date': x.date.to_numpy(),
        'next_date': x.next_date.to_numpy(),
        'from_state': x.state.to_numpy(), 'to_state': x.next_state.to_numpy(),
        'season': _season(x.date).to_numpy(),
        'qc_suspect_pair': (x.qc_suspect.astype(bool).to_numpy()
                            | x.next_qc_suspect.astype(bool).to_numpy()),
        'provenance': 'observed_only',
    }, columns=TRANSITION_COLUMNS)


def _count_cube(transitions: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    if not transitions.empty:
        i = transitions.from_state.map(_STATE_TO_I).to_numpy(dtype=int)
        j = transitions.to_state.map(_STATE_TO_I).to_numpy(dtype=int)
        np.add.at(matrix, (i, j), 1)
    return matrix


def _matrix_rows(counts: np.ndarray, **labels) -> list[dict]:
    totals = counts.sum(axis=1)
    return [dict(labels, from_state=a, to_state=b,
                 n_transitions=int(counts[i, j]), row_total=int(totals[i]),
                 probability=(float(counts[i, j]/totals[i])
                              if totals[i] else np.nan))
            for i, a in enumerate(STATES) for j, b in enumerate(STATES)]


def transition_matrices(transitions: pd.DataFrame,
                        stations: list[str]) -> dict[str, pd.DataFrame]:
    """All nine cells including zero-denominator rows; never silently drop sites."""
    pooled = pd.DataFrame(_matrix_rows(_count_cube(transitions)), columns=MATRIX_COLUMNS)
    by_station = pd.DataFrame([
        row for node in sorted(stations)
        for row in _matrix_rows(_count_cube(transitions.loc[transitions.node.eq(node)]), node=node)
    ], columns=['node', *MATRIX_COLUMNS])
    by_season = pd.DataFrame([
        row for season in SEASONS
        for row in _matrix_rows(_count_cube(transitions.loc[transitions.season.eq(season)]), season=season)
    ], columns=['season', *MATRIX_COLUMNS])
    return {'pooled': pooled, 'by_station': by_station, 'by_season': by_season}


def _block(date: pd.Series, scheme: str) -> pd.Series:
    if scheme == 'month':
        return date.dt.strftime('%Y-%m')
    if scheme not in ('14day', '28day'):
        raise ValueError(f'Unsupported bootstrap block {scheme}')
    days = 14 if scheme == '14day' else 28
    return ((date - pd.Timestamp('1970-01-01')).dt.days // days).astype(str)


def _station_block_cubes(transitions: pd.DataFrame, scheme: str) -> dict[str, np.ndarray]:
    d = transitions.copy()
    d['block'] = _block(d.date, scheme)
    result = {}
    for node, group in d.groupby('node', sort=True):
        cubes = [_count_cube(g) for _, g in group.groupby('block', sort=True)]
        result[node] = np.stack(cubes)
    return result


def transition_bootstrap(transitions: pd.DataFrame, n_bootstrap: int = 2000,
                         schemes=SCHEMES, seed: int = 2026) -> pd.DataFrame:
    """Resample *whole observed transition blocks*, separately by station.

    Transition date denotes its SOURCE day. No transitions are fabricated at
    resampled block joins. Cross-station temporal dependence is not fully
    preserved by the station-stratified bootstrap, so intervals are approximate.
    """
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive')
    rng = np.random.default_rng(seed)
    rows = []
    for scheme in schemes:
        blocks = _station_block_cubes(transitions, scheme)
        simulated = {}
        for node, cubes in blocks.items():
            k = len(cubes)
            draws = rng.integers(0, k, size=(n_bootstrap, k))
            simulated[node] = cubes[draws].sum(axis=1)
        scopes = [('pooled', '', transitions, list(simulated))]
        scopes += [('station', node, transitions.loc[transitions.node.eq(node)], [node])
                   for node in simulated]
        for scope, node, original, chosen in scopes:
            counts = _count_cube(original)
            point = counts.sum(axis=1)
            samples = sum((simulated[k] for k in chosen),
                          np.zeros((n_bootstrap, 3, 3), dtype=np.int64))
            denom = samples.sum(axis=2, keepdims=True)
            probabilities = np.full(samples.shape, np.nan, dtype=float)
            np.divide(samples, denom, out=probabilities, where=denom > 0)
            n_blocks = sum(len(blocks[k]) for k in chosen)
            for i, a in enumerate(STATES):
                for j, b in enumerate(STATES):
                    vals = probabilities[:, i, j]
                    vals = vals[np.isfinite(vals)]
                    # Do not report nominal 95% intervals when bootstrap rows
                    # have too few nonzero denominators to identify a state.
                    reliable = len(vals) >= max(1, int(np.ceil(.8 * n_bootstrap)))
                    ci = np.quantile(vals, [.025, .975]) if reliable else (np.nan, np.nan)
                    rows.append({
                        'scope': scope, 'node': node, 'block_scheme': scheme,
                        'from_state': a, 'to_state': b,
                        'probability': float(counts[i, j]/point[i]) if point[i] else np.nan,
                        'ci_lower': float(ci[0]), 'ci_upper': float(ci[1]),
                        'n_boot_valid': len(vals), 'n_bootstrap': n_bootstrap,
                        'n_station_blocks': n_blocks, 'row_total': int(point[i]),
                        'seed': seed,
                    })
    return pd.DataFrame(rows, columns=BOOT_COLUMNS)


def onset_count_bootstrap(states: pd.DataFrame, episodes: pd.DataFrame,
                          n_bootstrap: int = 2000, schemes=SCHEMES,
                          seed: int = 2026) -> pd.DataFrame:
    """Descriptive block-resampled onset counts, conditional on observed blocks.

    Counts episodes once at identifiable start date. Only blocks containing
    at least one known station-day can enter the resampling pool. Do not read
    resulting intervals as prospective incidence-rate or forecast uncertainty.
    """
    required = {'node', 'threshold_mgL', 'start_date_local', 'identifiable_onset'}
    if not required.issubset(episodes):
        raise ValueError(f'Missing episode columns: {sorted(required-set(episodes))}')
    e = episodes.copy()
    e['identifiable_onset'] = _flags(e['identifiable_onset'], 'identifiable_onset')
    e = e.loc[e.identifiable_onset].copy()
    e['start_date_local'] = pd.to_datetime(e.start_date_local)
    if e.start_date_local.isna().any():
        raise ValueError('Invalid identifiable onset dates')
    valid_days = states.loc[states.state.isin(STATES), ['node', 'date']]
    if e[['node', 'start_date_local']].rename(columns={'start_date_local': 'date'}).merge(
            valid_days, on=['node', 'date'], how='left', indicator=True)['_merge'].ne('both').any():
        raise ValueError('Episode onset date does not have a valid observed station-day')
    thresholds = sorted(pd.to_numeric(episodes.threshold_mgL).unique().tolist())
    rng = np.random.default_rng(seed)
    rows = []
    for scheme in schemes:
        v = valid_days.copy()
        v['block'] = _block(v.date, scheme)
        e['block'] = _block(e.start_date_local, scheme)
        for threshold in thresholds:
            sub = e.loc[e.threshold_mgL.eq(threshold)]
            counts = sub.groupby(['node', 'block']).size().to_dict()
            sim = {}
            nblocks = {}
            for node, group in v.groupby('node', sort=True):
                blocks = sorted(group.block.unique())
                if not blocks:
                    continue
                per_block = np.array([counts.get((node, block), 0) for block in blocks])
                k = len(blocks)
                drawn = rng.integers(0, k, size=(n_bootstrap, k))
                sim[node] = per_block[drawn].sum(axis=1)
                nblocks[node] = k
            for scope, node, chosen in ([('pooled', '', list(sim))] +
                                        [('station', s, [s]) for s in sim]):
                vals = sum((sim[s] for s in chosen), np.zeros(n_bootstrap, dtype=np.int64))
                ci = np.quantile(vals, [.025, .975])
                rows.append({
                    'scope': scope, 'node': node, 'block_scheme': scheme,
                    'threshold_mgL': threshold,
                    'observed_onsets': int(sub.loc[sub.node.isin(chosen)].shape[0]),
                    'bootstrap_median': float(np.median(vals)),
                    'ci_lower': float(ci[0]), 'ci_upper': float(ci[1]),
                    'n_bootstrap': n_bootstrap,
                    'n_station_blocks': sum(nblocks[s] for s in chosen),
                    'seed': seed,
                })
    return pd.DataFrame(rows, columns=ONSET_BOOT_COLUMNS)


def make_figure(matrices: dict[str, pd.DataFrame], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; transition heatmap skipped')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    p = matrices['pooled'].pivot(index='from_state', columns='to_state', values='probability')
    p = p.reindex(index=STATES, columns=STATES)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(p, vmin=0, vmax=1, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{p.iloc[i, j]:.1%}' if np.isfinite(p.iloc[i, j]) else 'N/A',
                    ha='center', va='center', color='black')
    ax.set_xticks(range(3), STATES)
    ax.set_yticks(range(3), STATES)
    ax.set(xlabel='Next eligible calendar day', ylabel='Current eligible day',
           title='Observed-only daily oxygen transitions')
    fig.colorbar(im, ax=ax, label='Transition probability')
    fig.tight_layout()
    fig.savefig(out_dir / 'transition_heatmap.png', dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Observed-only three-state transition analysis')
    parser.add_argument('--regimes', type=Path,
                        default=Path('trustworthy_outputs/onset/regime_days.csv'))
    parser.add_argument('--episodes', type=Path, default=None,
                        help='Optional Phase 1 episode_table.csv for onset-count uncertainty')
    parser.add_argument('--out-dir', type=Path,
                        default=Path('trustworthy_outputs/episode_dynamics'))
    parser.add_argument('--min-coverage', type=float, default=.95)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--blocks', nargs='+', choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    if not args.regimes.exists():
        parser.error(f'File not found: {args.regimes}')
    states = daily_states(pd.read_csv(args.regimes), min_coverage=args.min_coverage)
    transitions = observed_transitions(states)
    matrices = transition_matrices(transitions, sorted(states.node.unique()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    states.to_csv(args.out_dir / 'station_day_states.csv', index=False)
    transitions.to_csv(args.out_dir / 'observed_transitions.csv', index=False)
    for name, file in [('pooled', 'transition_matrix_pooled.csv'),
                       ('by_station', 'transition_matrix_by_station.csv'),
                       ('by_season', 'transition_matrix_by_season.csv')]:
        matrices[name].to_csv(args.out_dir / file, index=False)
    transition_bootstrap(transitions, args.bootstrap, args.blocks, args.seed).to_csv(
        args.out_dir / 'transition_bootstrap.csv', index=False)
    if args.episodes is not None:
        if not args.episodes.exists():
            parser.error(f'File not found: {args.episodes}')
        onset_count_bootstrap(states, pd.read_csv(args.episodes), args.bootstrap,
                              args.blocks, args.seed).to_csv(
            args.out_dir / 'onset_count_bootstrap.csv', index=False)
    if not args.no_figures:
        make_figure(matrices, args.out_dir / 'figures')
    metadata = {
        'regimes_path': str(args.regimes),
        'regimes_sha256': hashlib.sha256(args.regimes.read_bytes()).hexdigest(),
        'episodes_path': str(args.episodes) if args.episodes else None,
        'episodes_sha256': (hashlib.sha256(args.episodes.read_bytes()).hexdigest()
                            if args.episodes else None),
        'coverage_minimum': args.min_coverage, 'n_known_days': int(states.state.isin(STATES).sum()),
        'n_observed_transitions': len(transitions), 'n_stations': int(states.node.nunique()),
        'block_schemes': args.blocks, 'bootstrap_replicates': args.bootstrap,
        'random_seed': args.seed,
        'state_definition': 'daily 2/4 mg/L observed-only any-five-minute crossing',
        'transition_definition': 'two consecutive, eligible, same-station local calendar days',
        'season_assignment': 'origin/source local calendar day',
        'block_assignment': 'origin local date; resample complete station-blocks',
        'bootstrap_caveat': 'approximate; cross-station and cross-block dependence remains',
        'provenance': 'observed_only',
    }
    (args.out_dir / 'transition_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'{len(transitions)} valid transitions from {states.state.isin(STATES).sum()} known station-days')
    print(f'Saved transition outputs to {args.out_dir}')


if __name__ == '__main__':
    main()