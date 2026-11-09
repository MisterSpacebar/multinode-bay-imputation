"""Synthetic tests for observed-only daily state transitions and uncertainty."""
import unittest

import numpy as np
import pandas as pd

from trustworthy.episode_transitions import (
    STATES, _station_block_cubes, daily_states, observed_transitions,
    onset_count_bootstrap, transition_bootstrap, transition_matrices,
)


def regimes(node, start, states):
    days = pd.date_range(start, periods=len(states), freq='D')
    rows = []
    for date, state in zip(days, states):
        for threshold in (2, 4):
            hours = (2 if state == 'Severe' else 0) if threshold == 2 else (
                4 if state in ('Severe', 'Moderate') else 0)
            rows.append({
                'node': node, 'date': date.strftime('%Y-%m-%d'),
                'threshold_mgL': threshold, 'hours_below': hours if state != 'Unknown' else np.nan,
                'coverage': 1 if state != 'Unknown' else 0,
                'eligible': state != 'Unknown', 'qc_suspect': state == 'Severe',
            })
    return pd.DataFrame(rows)


class EpisodeTransitionsTests(unittest.TestCase):

    def test_three_state_transitions_and_unknown_boundaries(self):
        d = regimes('A', '2026-07-01', [
            'Normal', 'Moderate', 'Severe', 'Unknown',
            'Severe', 'Moderate', 'Normal'])
        s = daily_states(d.sample(frac=1, random_state=10))
        self.assertEqual(s.state.tolist(), [
            'Normal', 'Moderate', 'Severe', 'Unknown', 'Severe', 'Moderate', 'Normal'])
        t = observed_transitions(s)
        self.assertEqual(len(t), 4)
        self.assertEqual(list(zip(t.from_state, t.to_state)), [
            ('Normal', 'Moderate'), ('Moderate', 'Severe'),
            ('Severe', 'Moderate'), ('Moderate', 'Normal')])
        self.assertEqual(set(t.provenance), {'observed_only'})
        self.assertTrue(bool(t.qc_suspect_pair.any()))

    def test_absent_dates_do_not_create_transitions(self):
        d = regimes('A', '2026-03-07', ['Normal', 'Severe', 'Moderate'])
        d = d.loc[~d.date.eq('2026-03-08')]
        s = daily_states(d)
        self.assertEqual(s.state.tolist(), ['Normal', 'Unknown', 'Moderate'])
        self.assertEqual(len(observed_transitions(s)), 0)

    def test_strict_threshold_state_boundaries(self):
        # Exactly 4 mg/L has zero hours below either threshold; exactly 2
        # has hours below 4 but no hours below 2.
        d = regimes('A', '2026-07-01', ['Normal', 'Moderate', 'Severe'])
        s = daily_states(d)
        self.assertEqual(s.state.tolist(), ['Normal', 'Moderate', 'Severe'])
        self.assertEqual(s.hours_lt2.tolist(), [0., 0., 2.])

    def test_ineligible_one_threshold_blocks_entire_state(self):
        d = regimes('A', '2026-07-01', ['Normal', 'Severe', 'Normal'])
        d['coverage'] = d['coverage'].astype(float)
        d.loc[d.date.eq('2026-07-02') & d.threshold_mgL.eq(2), 'coverage'] = .94
        s = daily_states(d)
        self.assertEqual(s.state.tolist(), ['Normal', 'Unknown', 'Normal'])
        self.assertEqual(len(observed_transitions(s)), 0)

    def test_inconsistent_nested_hours_and_duplicates_rejected(self):
        d = regimes('A', '2026-07-01', ['Severe'])
        d.loc[d.threshold_mgL.eq(4), 'hours_below'] = 1
        with self.assertRaisesRegex(ValueError, 'exceed'):
            daily_states(d)
        d = regimes('A', '2026-07-01', ['Normal'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            daily_states(pd.concat([d, d]))

    def test_station_and_season_matrices_have_explicit_denominators(self):
        d = pd.concat([regimes('A', '2026-08-30', ['Normal', 'Moderate', 'Severe']),
                       regimes('B', '2026-08-30', ['Unknown', 'Normal', 'Unknown'])])
        s = daily_states(d)
        t = observed_transitions(s)
        matrices = transition_matrices(t, sorted(s.node.unique()))
        self.assertEqual(len(t), 2)
        self.assertEqual(len(matrices['pooled']), 9)
        self.assertEqual(len(matrices['by_station']), 18)
        self.assertEqual(len(matrices['by_season']), 36)
        # Aug 31 -> Sep 1 is assigned to the August source season.
        self.assertTrue((t.season == 'JJA').all())
        row = matrices['by_station'].query("node == 'B' and from_state == 'Severe'")
        self.assertTrue(row.probability.isna().all())
        self.assertTrue((row.row_total == 0).all())

    def test_whole_station_blocks_bootstrapped_deterministically(self):
        s = daily_states(pd.concat([
            regimes('A', '2026-07-01', ['Normal', 'Severe', 'Severe', 'Moderate']),
            regimes('B', '2026-07-01', ['Normal', 'Normal', 'Moderate', 'Severe']),
        ]))
        t = observed_transitions(s)
        cubes = _station_block_cubes(t, '14day')
        self.assertEqual(int(sum(c.sum() for c in cubes['A'])), 3)
        self.assertEqual(int(sum(c.sum() for c in cubes['B'])), 3)
        x = transition_bootstrap(t, 40, ('14day', 'month'), 42)
        y = transition_bootstrap(t, 40, ('14day', 'month'), 42)
        pd.testing.assert_frame_equal(x, y)
        self.assertEqual(len(x), (1 + 2) * 9 * 2)
        self.assertTrue((x.n_boot_valid <= x.n_bootstrap).all())
        self.assertTrue(x.loc[x.row_total.eq(0), 'probability'].isna().all())
        with self.assertRaises(ValueError):
            transition_bootstrap(t, 0)

    def test_onset_block_counts_preserve_station_and_threshold(self):
        s = daily_states(pd.concat([
            regimes('A', '2026-07-01', ['Normal', 'Severe', 'Normal', 'Moderate']),
            regimes('B', '2026-07-01', ['Normal', 'Moderate', 'Normal', 'Severe']),
        ]))
        episode_rows = pd.DataFrame([
            {'node': 'A', 'start_date_local': '2026-07-02', 'threshold_mgL': 2, 'identifiable_onset': True},
            {'node': 'B', 'start_date_local': '2026-07-04', 'threshold_mgL': 2, 'identifiable_onset': True},
            {'node': 'A', 'start_date_local': '2026-07-04', 'threshold_mgL': 4, 'identifiable_onset': True},
            {'node': 'A', 'start_date_local': '2026-07-01', 'threshold_mgL': 4, 'identifiable_onset': False},
        ])
        out = onset_count_bootstrap(s, episode_rows, 10, ('14day',), 10)
        pooled = out[out.scope.eq('pooled')]
        self.assertEqual(pooled.set_index('threshold_mgL').observed_onsets.to_dict(), {2: 2, 4: 1})
        self.assertEqual(out.loc[(out.scope == 'station') & (out.node == 'B') &
                                 (out.threshold_mgL == 4), 'observed_onsets'].iloc[0], 0)
        bad = episode_rows.iloc[[0]].copy()
        bad['start_date_local'] = '2027-01-01'
        with self.assertRaises(ValueError):
            onset_count_bootstrap(s, bad, 10)


if __name__ == '__main__':
    unittest.main()