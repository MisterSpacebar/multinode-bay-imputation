"""Synthetic warning-evaluation tests: no private observations."""
import unittest

import numpy as np
import pandas as pd

from trustworthy.transition_warning import ENV_FEATURES
from trustworthy.warning_evaluation import (
    CASE_KEYS, DECISION_COLUMNS, auc, brier_tables, cluster_bootstrap, matched_cases, pick_threshold,
    policy_decisions, policy_frame, policy_tables, state_climatology, with_references)


def make_windows(rows):
    df = pd.DataFrame(rows, columns=['issue_date', 'window_end', 'horizon_days', 'origin_state', 'label'])
    for c in ('issue_date', 'window_end'):
        df[c] = pd.to_datetime(df[c])
    df['node'] = 'A'
    df['status'] = 'evaluable'
    for f in ENV_FEATURES:
        df[f] = 1.0
    return df


def synth_cases(seed=0, months=('2026-01', '2026-02', '2026-03', '2026-04'), nodes=('A', 'B')):
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        for issue in pd.date_range(m + '-01', periods=pd.Period(m).days_in_month, freq='D'):
            for node in nodes:
                p = float(rng.uniform(0.02, 0.9))
                y = int(rng.uniform() < p)
                rows.append(dict(node=node, issue_date=issue, window_end=issue + pd.Timedelta(days=1),
                                 test_month=m, horizon_days=1, origin_state='Moderate', label=y,
                                 first_severe_date=issue + pd.Timedelta(days=1) if y else pd.NaT,
                                 markov=p * 0.8, do_history=p, environmental=p * 0.9,
                                 clim_state=0.3, clim_uncond=0.3))
    return pd.DataFrame(rows)


class ThresholdTests(unittest.TestCase):
    def test_perfect_separation_picks_lowest_positive_probability(self):
        self.assertEqual(pick_threshold([.1, .2, .8, .9], [0, 0, 1, 1], 2), .8)

    def test_no_events_never_alerts(self):
        self.assertEqual(pick_threshold([.5, .5, .6], [0, 0, 0], 5), np.inf)

    def test_ties_choose_highest_threshold(self):
        # Alerting at .5 and at 0.9 cost the same on this sample; fewer alerts must win.
        self.assertLessEqual(pick_threshold([.5, .9], [1, 1], 1), .5)
        self.assertEqual(pick_threshold([.2, .2, .9], [0, 1, 1], 1), .9)

    def test_auc_basic(self):
        self.assertEqual(auc([0, 0, 1, 1], [.1, .2, .8, .9]), 1.0)
        self.assertTrue(np.isnan(auc([0, 0], [.1, .2])))


class ClimatologyLeakageTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(10):
            d = pd.Timestamp('2026-02-01') + pd.Timedelta(days=i)
            rows.append((d, d + pd.Timedelta(days=1), 1, 'Normal', int(i < 2)))
            rows.append((d, d + pd.Timedelta(days=1), 1, 'Moderate', int(i < 6)))
        self.rows = rows
        self.cases = pd.DataFrame(dict(
            node=['A', 'A'], issue_date=pd.to_datetime(['2026-03-10', '2026-03-10']),
            test_month=['2026-03', '2026-03'], horizon_days=[1, 1], origin_state=['Normal', 'Moderate']))

    def probs(self, rows):
        s, u, n, _ = state_climatology(self.cases, make_windows(rows), prior=20)
        return s.tolist(), u.tolist(), n.tolist()

    def test_shrinkage_formula(self):
        s, u, n = self.probs(self.rows)
        # unconditional 8/20=.4; Normal (2 + 20*.4)/(10+20); Moderate (6 + 20*.4)/(10+20)
        self.assertAlmostEqual(u[0], .4)
        self.assertAlmostEqual(s[0], 10 / 30)
        self.assertAlmostEqual(s[1], 14 / 30)
        self.assertEqual(n[0], 20)

    def test_windows_ending_on_or_after_test_month_start_are_not_used(self):
        base = self.probs(self.rows)
        # Window issued Feb 28 ends Mar 1 == test-month start; one ending Mar 3 straddles the month.
        for end in ('2026-03-01', '2026-03-03'):
            for label in (0, 1):
                extra = [(pd.Timestamp('2026-02-28'), pd.Timestamp(end), 1, 'Normal', label)]
                self.assertEqual(self.probs(self.rows + extra), base)

    def test_window_ending_before_month_start_is_used(self):
        base = self.probs(self.rows)
        extra = [(pd.Timestamp('2026-02-27'), pd.Timestamp('2026-02-28'), 1, 'Normal', 1)]
        self.assertNotEqual(self.probs(self.rows + extra), base)

    def test_incomplete_windows_are_not_training_data(self):
        w = make_windows(self.rows)
        extra = make_windows([(pd.Timestamp('2026-02-20'), pd.Timestamp('2026-02-21'), 1, 'Normal', 1)])
        extra.loc[:, 'status'] = 'unknown_future'
        base = state_climatology(self.cases, w, prior=20)[0].tolist()
        self.assertEqual(state_climatology(self.cases, pd.concat([w, extra]), prior=20)[0].tolist(), base)

    def test_case_without_training_history_is_dropped_and_counted(self):
        kept, dropped = with_references(self.cases.assign(markov=.1, do_history=.1, environmental=.1, label=0,
                                                         window_end=pd.Timestamp('2026-03-11')),
                                        make_windows([]))
        self.assertEqual((len(kept), dropped), (0, 2))


class ChronologicalPolicyTests(unittest.TestCase):
    def decisions(self, cases):
        return policy_decisions(cases, ratios=(2, 5), min_cases=10, min_class=2)

    def thresholds(self, dec):
        d = dec[(dec.model == 'do_history') & (dec.cost_ratio == 5)]
        return d.groupby('test_month').chronological_threshold.first()

    def test_schema_and_alert_rule(self):
        dec = self.decisions(synth_cases())
        self.assertEqual(list(dec.columns), DECISION_COLUMNS)
        a = dec[dec.chronological_available]
        self.assertTrue(((a.probability >= a.chronological_threshold).astype(int) == a.chronological_alert).all())
        self.assertTrue((dec.loc[~dec.chronological_available, 'chronological_alert'] == 0).all())
        self.assertTrue(((dec.probability >= dec.theoretical_threshold).astype(int) == dec.theoretical_alert).all())

    def test_first_month_has_no_chronological_threshold(self):
        dec = self.decisions(synth_cases())
        self.assertFalse(dec[dec.test_month == '2026-01'].chronological_available.any())
        self.assertTrue(dec[dec.test_month == '2026-02'].chronological_available.all())

    def test_thresholds_ignore_outcomes_that_ended_on_or_after_the_month(self):
        cases = synth_cases()
        flipped = cases.copy()
        later = flipped.window_end >= pd.Timestamp('2026-03-01')
        flipped.loc[later, 'label'] = 1 - flipped.loc[later, 'label']
        a, b = self.thresholds(self.decisions(cases)), self.thresholds(self.decisions(flipped))
        pd.testing.assert_series_equal(a.loc[:'2026-03'], b.loc[:'2026-03'])
        # Control: April's threshold does consume March outcomes, so the flip must reach it.
        self.assertNotEqual(a.loc['2026-04'], b.loc['2026-04'])

    def test_case_ending_on_first_day_of_month_is_not_used(self):
        # Controlled case: three Feb positives at .9 and three negatives at .2 give threshold .9.
        # Five positives at .5 whose window ends on Mar 1 would pull it to .5 if (wrongly) used.
        def row(node, issue, label, p, month):
            issue = pd.Timestamp(issue)
            return dict(node=node, issue_date=issue, window_end=issue + pd.Timedelta(days=1), test_month=month,
                        horizon_days=1, origin_state='Moderate', label=label,
                        first_severe_date=issue + pd.Timedelta(days=1) if label else pd.NaT,
                        markov=p, do_history=p, environmental=p, clim_state=.3, clim_uncond=.3)
        rows = [row('A', f'2026-02-{d:02d}', 1, .9, '2026-02') for d in (1, 2, 3)]
        rows += [row('A', f'2026-02-{d:02d}', 0, .2, '2026-02') for d in (4, 5, 6)]
        rows += [row(n, '2026-02-28', 1, .5, '2026-02') for n in 'BCDEF']
        rows += [row('A', '2026-03-10', 0, .7, '2026-03')]
        dec = policy_decisions(pd.DataFrame(rows), ratios=(1,), models=('do_history',), min_cases=6, min_class=2)
        march = dec[dec.test_month == '2026-03'].iloc[0]
        self.assertEqual(march.chronological_threshold, .9)
        self.assertEqual(march.chronological_alert, 0)

    def test_random_case_ending_on_first_day_of_month_leaves_thresholds_unchanged(self):
        cases = synth_cases()
        flipped = cases.copy()
        edge = flipped.window_end == pd.Timestamp('2026-03-01')
        self.assertTrue(edge.any())
        flipped.loc[edge, 'label'] = 1 - flipped.loc[edge, 'label']
        a, b = self.thresholds(self.decisions(cases)), self.thresholds(self.decisions(flipped))
        self.assertEqual(a.loc['2026-03'], b.loc['2026-03'])

    def test_exported_decisions_reproduce_reported_metrics(self):
        dec = self.decisions(synth_cases())
        scores, comps = policy_tables(dec, n_boot=0)
        for h, r in ((1, 2), (1, 5)):
            s = policy_frame(dec, h, r)
            row = scores[(scores.horizon_days == h) & (scores.cost_ratio == r) & (scores.policy == 'do_chrono')].iloc[0]
            fn = ((s.do_chrono == 0) & (s.label == 1)).sum()
            fp = ((s.do_chrono == 1) & (s.label == 0)).sum()
            self.assertAlmostEqual(row.cost_per_100, (r * fn + fp) / len(s) * 100)
            self.assertAlmostEqual(row.alerts_per_100, s.do_chrono.mean() * 100)
            never = scores[(scores.horizon_days == h) & (scores.cost_ratio == r) & (scores.policy == 'never')].iloc[0]
            self.assertAlmostEqual(never.cost_per_100, r * s.label.mean() * 100)
            self.assertEqual(row.n_issue_days, len(s))
            self.assertEqual(row.n_issue_days, never.n_issue_days)
        self.assertTrue(comps.cost_saved_per_100.notna().all())

    def test_policy_frame_excludes_cases_without_a_threshold(self):
        dec = self.decisions(synth_cases())
        s = policy_frame(dec, 1, 2)
        self.assertEqual(s.issue_date.min().month, 2)
        self.assertEqual(len(s), int(dec[(dec.model == 'do_history') & (dec.cost_ratio == 2)].chronological_available.sum()))


class MatchedCaseTests(unittest.TestCase):
    def preds(self):
        rows = []
        for model, p in (('markov', .1), ('do_history', .2), ('environmental', .3)):
            for day, label in (('2026-03-01', 0), ('2026-03-02', 1)):
                rows.append(dict(node='A', issue_date=day, window_end='2026-03-03', test_month='2026-03',
                                 horizon_days=1, origin_state='Normal', model=model, label=label, probability=p))
        return pd.DataFrame(rows)

    def windows(self):
        return pd.DataFrame(dict(node=['A', 'A'], issue_date=['2026-03-01', '2026-03-02'], horizon_days=[1, 1],
                                 first_severe_date=[None, '2026-03-03']))

    def test_cases_are_matched_across_models_and_carry_onset_dates(self):
        c = matched_cases(self.preds(), self.windows())
        self.assertEqual(len(c), 2)
        self.assertEqual(c.loc[c.label == 1, 'first_severe_date'].iloc[0], pd.Timestamp('2026-03-03'))
        self.assertTrue(set(CASE_KEYS) <= set(c.columns))

    def test_case_scored_by_only_two_models_is_dropped(self):
        p = self.preds()
        p = p.drop(p[(p.model == 'environmental') & (p.issue_date == '2026-03-02')].index)
        self.assertEqual(len(matched_cases(p, self.windows())), 1)

    def test_duplicates_and_invalid_probabilities_rejected(self):
        p = self.preds()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            matched_cases(pd.concat([p, p.iloc[:1]]), self.windows())
        bad = self.preds()
        bad.loc[0, 'probability'] = 1.2
        with self.assertRaisesRegex(ValueError, r'\[0,1\]'):
            matched_cases(bad, self.windows())

    def test_unknown_model_rejected(self):
        p = self.preds()
        p.loc[0, 'model'] = 'oracle'
        with self.assertRaisesRegex(ValueError, 'expected models'):
            matched_cases(p, self.windows())

    def test_positive_case_needs_an_onset_date(self):
        w = self.windows()
        w.loc[1, 'first_severe_date'] = None
        with self.assertRaisesRegex(ValueError, 'first_severe_date'):
            matched_cases(self.preds(), w)


class BootstrapTests(unittest.TestCase):
    def test_seeded_bootstrap_is_reproducible_and_clustered(self):
        c = synth_cases()
        stat = lambda x: float(x.label.mean())
        a = cluster_bootstrap(c, stat, 50, np.random.default_rng(1))
        b = cluster_bootstrap(c, stat, 50, np.random.default_rng(1))
        self.assertEqual(a, b)
        self.assertEqual(a[2], c.node.add('|').add(c.test_month).nunique())
        self.assertLessEqual(a[0], a[1])

    def test_brier_tables_point_estimates_do_not_need_a_bootstrap(self):
        c = synth_cases()
        scores, comps = brier_tables(c, n_boot=0)
        row = scores[(scores.model == 'do_history') & (scores.horizon_days == 1)].iloc[0]
        self.assertAlmostEqual(row.brier, float(((c.do_history - c.label) ** 2).mean()))
        self.assertTrue(np.isnan(row.brier_lo))
        pair = comps[(comps.reference == 'clim_state') & (comps.model == 'do_history')].iloc[0]
        self.assertAlmostEqual(pair.brier_reference_minus_model,
                               float(((c.clim_state - c.label) ** 2).mean() - ((c.do_history - c.label) ** 2).mean()))


if __name__ == '__main__':
    unittest.main()
