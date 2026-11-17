"""Synthetic episode-matching tests: no private observations."""
import unittest

import numpy as np
import pandas as pd

from trustworthy.episode_warning_evaluation import (
    EXCL_EARLY, EXCL_NO_HINDCAST, EXCL_NO_ISSUE, match_policy, onset_accounting, policy_stats, qualifying_onsets, tally)


def frame(rows):
    """rows: (date, label, first_severe_date or None, alert[, avail])"""
    out = pd.DataFrame([dict(node='A', issue_date=pd.Timestamp(r[0]), test_month=r[0][:7], label=r[1],
                             first_severe_date=pd.Timestamp(r[2]) if r[2] else pd.NaT, alert=r[3],
                             avail=r[4] if len(r) > 4 else True) for r in rows])
    return out


def onsets(*dates, qc=False):
    return pd.DataFrame(dict(node='A', onset=pd.to_datetime(list(dates)), qc_suspect=qc))


def run(rows, on):
    units, n_cases, n_alerts, cl, opp = match_policy(frame(rows), on)
    T, leads = tally(units, n_cases, n_alerts, cl, opp)
    return cl, opp, policy_stats(T, leads, np.arange(len(units)))


class OnsetDefinitionTests(unittest.TestCase):
    def episodes(self):
        base = dict(threshold_mgL=2, identifiable_onset=True, censored_left=False, qc_suspect=False)
        rows = [dict(base, node='A', start_date_local='2026-01-10'),
                dict(base, node='A', start_date_local='2026-02-10', censored_left=True),
                dict(base, node='A', start_date_local='2026-03-10', identifiable_onset=False),
                dict(base, node='A', start_date_local='2026-04-10', threshold_mgL=4),
                dict(base, node='B', start_date_local='2026-01-10', qc_suspect=True)]
        return pd.DataFrame(rows)

    def test_only_identifiable_uncensored_two_mgl_onsets_qualify(self):
        o = qualifying_onsets(self.episodes())
        self.assertEqual(sorted(zip(o.node, o.onset.dt.strftime('%Y-%m-%d'))), [('A', '2026-01-10'), ('B', '2026-01-10')])
        self.assertEqual(o.qc_suspect.tolist(), [False, True])

    def test_station_identities_stay_distinct(self):
        self.assertEqual(len(qualifying_onsets(self.episodes())), 2)

    def test_duplicate_station_onset_rejected(self):
        e = self.episodes().iloc[[0, 0]]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            qualifying_onsets(e)


class MatchingTests(unittest.TestCase):
    def test_repeated_alerts_before_one_onset_count_once(self):
        rows = [('2026-01-07', 1, '2026-01-10', 1), ('2026-01-08', 1, '2026-01-10', 1), ('2026-01-09', 1, '2026-01-10', 1)]
        cl, opp, st = run(rows, onsets('2026-01-10'))
        self.assertEqual(len(opp), 1)
        self.assertEqual(len(cl), 1)
        self.assertEqual((cl.n_days.iloc[0], bool(cl.credited.iloc[0])), (3, True))
        self.assertEqual(opp.n_alert_days.iloc[0], 3)
        self.assertEqual(opp.lead_days_credited.iloc[0], 3)
        self.assertEqual(st['det_credited'], 1.0)
        self.assertEqual(st['evaluable_onsets'], 1)
        self.assertEqual(st['repeat_frac'], 1.0)
        self.assertEqual(st['median_lead'], 3)

    def test_two_clusters_for_one_onset_credit_the_first_only(self):
        rows = [('2026-01-07', 1, '2026-01-10', 1), ('2026-01-08', 1, '2026-01-10', 0),
                ('2026-01-09', 1, '2026-01-10', 1)]
        cl, opp, st = run(rows, onsets('2026-01-10'))
        self.assertEqual(cl.credited.tolist(), [True, False])
        self.assertEqual(cl.hit.tolist(), [True, True])
        self.assertEqual(st['det_credited'], 1.0)
        self.assertEqual(st['false_frac'], 0.0)
        self.assertEqual(opp.lead_days_credited.iloc[0], 3)

    def test_missing_calendar_day_breaks_a_cluster(self):
        # Jan 8 is absent (Unknown/ineligible): two alerts that skip it are separate warnings.
        rows = [('2026-01-07', 1, '2026-01-10', 1), ('2026-01-09', 1, '2026-01-10', 1)]
        cl, _, _ = run(rows, onsets('2026-01-10'))
        self.assertEqual(len(cl), 2)

    def test_alerts_without_an_onset_are_false_clusters(self):
        rows = [('2026-01-05', 0, None, 1), ('2026-01-06', 0, None, 1), ('2026-01-07', 1, '2026-01-10', 0),
                ('2026-01-08', 1, '2026-01-10', 0)]
        cl, opp, st = run(rows, onsets('2026-01-10'))
        self.assertEqual((len(cl), int(cl.hit.sum())), (1, 0))
        self.assertEqual(st['false_frac'], 1.0)
        self.assertEqual(st['det_credited'], 0.0)
        self.assertFalse(opp.detected_raw.iloc[0])
        self.assertTrue(np.isnan(opp.lead_days_credited.iloc[0]))

    def test_unavailable_early_fold_cases_are_neither_alerts_nor_opportunities(self):
        rows = [('2026-01-09', 1, '2026-01-10', 1, False)]
        _, opp, _ = run(rows + [('2026-02-01', 0, None, 0)], onsets('2026-01-10'))
        self.assertEqual(len(opp), 0)

    def test_lead_time_is_days_from_earliest_alert_to_onset(self):
        rows = [('2026-01-08', 1, '2026-01-10', 1), ('2026-01-09', 1, '2026-01-10', 1)]
        _, opp, _ = run(rows, onsets('2026-01-10'))
        self.assertEqual(opp.lead_days_raw.iloc[0], 2)
        self.assertEqual(opp.lead_days_credited.iloc[0], 2)

    def test_alert_burden_counts_alert_days_over_evaluable_issue_days(self):
        rows = [('2026-01-07', 0, None, 1), ('2026-01-08', 0, None, 0), ('2026-01-09', 1, '2026-01-10', 1),
                ('2026-01-10', 0, None, 0)]
        cl, _, st = run(rows, onsets('2026-01-10'))
        self.assertAlmostEqual(st['alerts100'], 50.0)
        self.assertAlmostEqual(st['clusters100'], 50.0)
        self.assertEqual(len(cl), 2)

    def test_positive_window_must_point_to_a_qualifying_onset(self):
        rows = [('2026-01-09', 1, '2026-01-10', 1)]
        with self.assertRaisesRegex(ValueError, 'qualifying'):
            run(rows, onsets('2026-02-10'))

    def test_qc_suspect_onsets_can_be_excluded_from_the_denominator(self):
        rows = [('2026-01-09', 1, '2026-01-10', 1), ('2026-02-09', 1, '2026-02-10', 0)]
        on = pd.DataFrame(dict(node='A', onset=pd.to_datetime(['2026-01-10', '2026-02-10']), qc_suspect=[False, True]))
        units, n_cases, n_alerts, cl, opp = match_policy(frame(rows), on)
        T, leads = tally(units, n_cases, n_alerts, cl, opp, exclude_qc=True)
        st = policy_stats(T, leads, np.arange(len(units)))
        self.assertEqual((st['evaluable_onsets'], st['det_credited']), (1, 1.0))


class AccountingTests(unittest.TestCase):
    def test_every_exclusion_reason_is_reported(self):
        on = onsets('2026-01-10', '2026-02-10', '2026-03-10', '2026-04-10', '2026-05-10')
        cases = pd.DataFrame(dict(
            node='A', issue_date=pd.to_datetime(['2026-01-09', '2026-02-09']), test_month=['2026-01', '2026-02'],
            label=[1, 1], first_severe_date=pd.to_datetime(['2026-01-10', '2026-02-10']), avail=[True, False]))
        windows = pd.DataFrame(dict(
            node='A', issue_date=['2026-01-09', '2026-02-09', '2026-03-09', '2026-04-09'], horizon_days=1,
            first_severe_date=['2026-01-10', '2026-02-10', '2026-03-10', None],
            status=['evaluable', 'evaluable', 'evaluable', 'unknown_future']))
        acc = onset_accounting(on, cases, windows, 1).set_index('onset').reason
        self.assertEqual(acc[pd.Timestamp('2026-01-10')], 'evaluable')
        self.assertEqual(acc[pd.Timestamp('2026-02-10')], EXCL_EARLY)
        self.assertEqual(acc[pd.Timestamp('2026-03-10')], EXCL_NO_HINDCAST)
        self.assertEqual(acc[pd.Timestamp('2026-04-10')], 'window_not_evaluable:unknown_future')
        self.assertEqual(acc[pd.Timestamp('2026-05-10')], EXCL_NO_ISSUE)


if __name__ == '__main__':
    unittest.main()
