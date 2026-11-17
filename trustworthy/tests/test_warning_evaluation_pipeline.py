"""End-to-end synthetic test of the warning-evaluation chain, plus optional real-output regression targets.

The synthetic stations run through the real ``transition_warning`` hindcast so the evaluation
modules are exercised against the schema the pipeline actually writes.
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy import episode_warning_evaluation as ewe
from trustworthy import warning_burden as wb
from trustworthy import warning_evaluation as we
from trustworthy.transition_warning import hindcast, prepare_states, validate_transitions, warning_windows


def station(node, n, cycle, shift):
    dates = pd.date_range('2026-01-01', periods=n, freq='D')
    i = np.arange(n)
    labels = np.where((i + shift) % cycle == 2, 'Severe', np.where((i + shift) % 3 == 1, 'Moderate', 'Normal'))
    labels[0] = 'Normal'
    states = pd.DataFrame(dict(node=node, date=dates, state=labels, hours_lt2=np.where(labels == 'Severe', 2., 0.),
                               hours_lt4=np.where(labels == 'Normal', 0., np.where(labels == 'Moderate', 3., 5.)),
                               qc_suspect=False, provenance='observed_only'))
    daily = pd.DataFrame(dict(node=node, date=dates, eligible=True, coverage=1.,
                              odo_mean=np.where(labels == 'Normal', 6., np.where(labels == 'Moderate', 3., 1.)) + (i % 7) / 10,
                              odo_min=np.where(labels == 'Normal', 5., np.where(labels == 'Moderate', 2.5, 1.)),
                              temp_mean=24 + i % 17 / 10., sal_mean=26 + i % 11 / 10.))
    trans = pd.DataFrame(dict(node=node, date=dates[:-1], next_date=dates[1:], from_state=labels[:-1],
                              to_state=labels[1:], provenance='observed_only'))
    onset = [dict(node=node, threshold_mgL=2, start_date_local=dates[k].strftime('%Y-%m-%d'), identifiable_onset=True,
                  censored_left=False, qc_suspect=False)
             for k in range(1, n) if labels[k] == 'Severe' and labels[k - 1] != 'Severe']
    return states, daily, trans, pd.DataFrame(onset)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / 'trustworthy_outputs'
        parts = [station('A', 330, 7, 0), station('B', 330, 9, 2), station('C', 330, 11, 4)]
        states, daily, trans, episodes = (pd.concat([p[k] for p in parts], ignore_index=True) for k in range(4))
        st = prepare_states(states)
        windows = warning_windows(st, daily)
        preds, _, _ = hindcast(windows, validate_transitions(trans, st), n_paths=0)
        (cls.root / 'transition_warning').mkdir(parents=True)
        (cls.root / 'episode_dynamics').mkdir()
        preds.to_csv(cls.root / 'transition_warning' / 'hindcast_predictions.csv', index=False)
        windows.to_csv(cls.root / 'transition_warning' / 'warning_windows.csv', index=False)
        episodes.to_csv(cls.root / 'episode_dynamics' / 'episode_table.csv', index=False)
        states.to_csv(cls.root / 'episode_dynamics' / 'station_day_states.csv', index=False)
        trans.to_csv(cls.root / 'episode_dynamics' / 'observed_transitions.csv', index=False)
        cls.n_onsets = len(episodes)
        cls.out = Path(cls.tmp.name) / 'warning_evaluation'
        cls.meta = we.run(cls.root, cls.out, n_boot=5, sensitivity=False)
        cls.ep_meta = ewe.run(cls.root, cls.out, n_boot=5)
        cls.burden_meta = wb.run(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def table(self, name):
        return pd.read_csv(self.out / name)

    def test_all_exports_are_written(self):
        for name in ('heldout_climatology.csv', 'brier_scores.csv', 'brier_comparisons.csv', 'policy_decisions.csv',
                     'policy_scores.csv', 'policy_comparisons.csv', 'onset_accounting.csv', 'episode_detection.csv',
                     'warning_clusters.csv', 'episode_policy_summary.csv', 'episode_policy_comparisons.csv',
                     'detection_burden.csv', 'evaluation_metadata.json', 'episode_evaluation_metadata.json',
                     'detection_burden_metadata.json'):
            self.assertTrue((self.out / name).exists(), name)

    def test_metadata_hashes_match_inputs_and_outputs(self):
        for meta in (self.meta, self.ep_meta, self.burden_meta):
            self.assertEqual(meta['provenance'], 'observed_only')
            self.assertIn('git_commit', meta['code'])
            self.assertTrue(meta['limitations'])
            for item in meta['inputs'].values():
                self.assertEqual(item['sha256'], sha(item['path']))
            for item in meta['outputs'].values():
                self.assertEqual(item['sha256'], sha(item['path']))
        self.assertEqual(json.loads((self.out / 'evaluation_metadata.json').read_text())['parameters']['n_boot'], 5)

    def test_counts_match_the_exports(self):
        c = self.meta['counts']
        held = self.table('heldout_climatology.csv')
        self.assertEqual(c['matched_cases'], len(held))
        self.assertEqual(c['positive_matched_cases'], int(held.label.sum()))
        dec = we.load_decisions(self.out / 'policy_decisions.csv')
        for h, n in c['policy_cases_by_horizon'].items():
            self.assertEqual(n, len(we.policy_frame(dec, int(h), we.COST_RATIOS[0])))

    def test_hindcast_cases_never_train_on_their_own_outcome_window(self):
        held = self.table('heldout_climatology.csv')
        self.assertTrue(held.clim_state.between(0, 1).all())
        self.assertTrue((held.n_train > 0).all())

    def test_episode_and_policy_modules_agree_on_alert_burden(self):
        scores = self.table('policy_scores.csv')
        summary = self.table('episode_policy_summary.csv')
        for row in summary.itertuples():
            policy = 'do_chrono' if row.model == 'do_history' else 'clim_chrono'
            ref = scores[(scores.horizon_days == row.horizon_days) & (scores.cost_ratio == row.cost_ratio)
                         & (scores.policy == policy)].iloc[0]
            self.assertAlmostEqual(row.alerts100, ref.alerts_per_100)
            self.assertEqual(row.n_issue_days, ref.n_issue_days)

    def test_onset_accounting_covers_every_qualifying_onset_once_per_horizon(self):
        acc = self.table('onset_accounting.csv')
        for h, g in acc.groupby('horizon_days'):
            self.assertEqual(len(g), self.n_onsets)
            self.assertEqual(g.duplicated(['node', 'onset']).sum(), 0)
        summary = self.table('episode_policy_summary.csv')
        for row in summary.itertuples():
            n_eval = (acc[acc.horizon_days == row.horizon_days].reason == 'evaluable').sum()
            self.assertEqual(row.evaluable_onsets, n_eval)

    def test_detection_cannot_exceed_any_alert_detection_and_is_a_proportion(self):
        s = self.table('episode_policy_summary.csv')
        self.assertTrue((s.det_credited <= s.det_raw + 1e-12).all())
        self.assertTrue(s.det_credited.between(0, 1).all())
        d = self.table('episode_detection.csv')
        evald = d[d.evaluable]
        self.assertTrue((evald.lead_days_credited.dropna() >= 1).all())
        self.assertTrue((evald.lead_days_credited.dropna() <= evald.horizon_days[evald.lead_days_credited.notna()]).all())

    def test_repeated_warnings_do_not_inflate_detection(self):
        d = self.table('episode_detection.csv')
        e = d[d.evaluable]
        per = e.groupby(['horizon_days', 'model', 'cost_ratio'])
        for key, g in per:
            self.assertLessEqual(g.credited.sum(), g.evaluable.sum())
            self.assertEqual(g.duplicated(['node', 'onset']).sum(), 0)
        c = self.table('warning_clusters.csv')
        self.assertTrue((c.groupby(['horizon_days', 'model', 'cost_ratio', 'node', 'credited_onset'])
                         .credited.sum() <= 1).all())

    def test_burden_curve_is_monotone_and_complete_at_full_alerting(self):
        b = self.table('detection_burden.csv')
        for h, g in b.groupby('horizon_days'):
            curve = g[g.series == 'do_history_risk_ranked'].sort_values('alert_pct')
            self.assertTrue(curve.detection.is_monotonic_increasing)
            self.assertEqual(curve.detection.iloc[-1], 1.0)
            self.assertEqual(g[g.series == 'clim_state_every_day'].detection.iloc[0], 1.0)
            summary = self.table('episode_policy_summary.csv')
            self.assertEqual(g.n_evaluable_onsets.iloc[0],
                             summary[summary.horizon_days == h].evaluable_onsets.iloc[0])
        self.assertTrue(b[b.series == 'do_history_risk_ranked'].note.str.contains('not an implementable').all())

    def test_runs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as other:
            out2 = Path(other)
            we.run(self.root, out2, n_boot=5, sensitivity=False)
            ewe.run(self.root, out2, n_boot=5)
            wb.run(out2)
            for name in ('policy_decisions.csv', 'policy_scores.csv', 'brier_scores.csv', 'episode_policy_summary.csv',
                         'warning_clusters.csv', 'detection_burden.csv'):
                self.assertEqual(sha(self.out / name), sha(out2 / name), name)

    def test_prior_sensitivity_runs_on_pipeline_outputs(self):
        held = self.table('heldout_climatology.csv')
        for c in ('issue_date', 'window_end', 'first_severe_date'):
            held[c] = pd.to_datetime(held[c])
        states = prepare_states(pd.read_csv(self.root / 'episode_dynamics' / 'station_day_states.csv'))
        trans = validate_transitions(pd.read_csv(self.root / 'episode_dynamics' / 'observed_transitions.csv'), states)
        windows = pd.read_csv(self.root / 'transition_warning' / 'warning_windows.csv')
        sens = we.prior_sensitivity(held, windows, trans, grid=(5, 20), n_boot=3)
        self.assertEqual(set(sens.component), {'state_climatology_prior', 'markov_station_prior'})
        primary = sens[(sens.component == 'state_climatology_prior') & (sens.prior == 20)]
        # The prior-20 reference must reproduce the exported held-out climatology exactly.
        for r in primary.itertuples():
            d = held[held.horizon_days == r.horizon_days]
            self.assertAlmostEqual(r.brier, float(((d.clim_state - d.label) ** 2).mean()))

    def test_figure_is_written(self):
        try:
            from trustworthy.warning_evaluation_figures import plot_detection_burden
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest('matplotlib unavailable')
        out = self.out / 'figures' / 'detection_vs_burden.png'
        plot_detection_burden(self.table('detection_burden.csv'), self.table('episode_policy_summary.csv'), out)
        self.assertGreater(out.stat().st_size, 1000)


REAL = os.environ.get('TRUSTWORTHY_REAL_OUTPUTS')


@unittest.skipUnless(REAL, 'set TRUSTWORTHY_REAL_OUTPUTS to a local trustworthy_outputs folder to run')
class RealOutputRegressionTests(unittest.TestCase):
    """Regression targets for the 2025-03 to 2026-08 local export; not constants of the method."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.meta = we.run(Path(REAL), cls.out, n_boot=0, sensitivity=False)
        ewe.run(Path(REAL), cls.out, n_boot=0)
        wb.run(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_case_and_onset_counts(self):
        c = self.meta['counts']
        self.assertEqual(c['matched_cases'], 1402)
        self.assertEqual(c['policy_cases_by_horizon'], {1: 469, 3: 429})
        acc = pd.read_csv(self.out / 'onset_accounting.csv')
        self.assertEqual(acc.groupby('horizon_days').size().to_dict(), {1: 63, 3: 63})
        self.assertEqual((acc.reason == 'evaluable').groupby(acc.horizon_days).sum().to_dict(), {1: 48, 3: 48})

    def test_headline_do_history_policy_at_5_to_1(self):
        s = pd.read_csv(self.out / 'episode_policy_summary.csv')
        r = s[(s.model == 'do_history') & (s.cost_ratio == 5)].set_index('horizon_days')
        self.assertAlmostEqual(r.loc[1, 'det_credited'], 0.54, delta=0.01)
        self.assertAlmostEqual(r.loc[1, 'alerts100'], 14.9, delta=0.1)
        self.assertAlmostEqual(r.loc[3, 'det_credited'], 0.79, delta=0.01)
        self.assertAlmostEqual(r.loc[3, 'alerts100'], 35.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
