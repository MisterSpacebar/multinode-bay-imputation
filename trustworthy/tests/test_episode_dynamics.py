"""Regression checks for canonical observed-only episode summaries.

No private station data or GNN values are required.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.episode_dynamics import (
    daily_episode_table, subdaily_episode_table, summarize_episodes,
)


def regime_frame(node, threshold, dates, hours, eligible=None, qc_suspect=None):
    n = len(dates)
    return pd.DataFrame({
        "node": [node] * n,
        "date": dates,
        "threshold_mgL": [threshold] * n,
        "coverage": [1.0] * n,
        "hours_below": hours,
        "eligible": eligible if eligible is not None else [True] * n,
        "qc_suspect": qc_suspect if qc_suspect is not None else [False] * n,
    })


class EpisodeDynamicsTests(unittest.TestCase):

    def test_missing_day_censors_both_sides_and_is_not_recovery(self):
        dates = pd.date_range("2026-07-01", periods=6, freq="D")
        df = regime_frame("A", 2, dates, [0, 3, np.nan, 1, 2, 0],
                          [True, True, False, True, True, True])
        e = daily_episode_table(df)
        self.assertEqual(len(e), 2)
        self.assertTrue(bool(e.iloc[0]["identifiable_onset"]))
        self.assertTrue(bool(e.iloc[0]["censored_right"]))
        self.assertFalse(bool(e.iloc[0]["observed_recovery"]))
        self.assertEqual(e.iloc[0]["right_boundary"], "unknown_day")
        self.assertTrue(bool(e.iloc[1]["censored_left"]))
        self.assertFalse(bool(e.iloc[1]["identifiable_onset"]))
        self.assertTrue(bool(e.iloc[1]["observed_recovery"]))
        self.assertEqual(e.iloc[1]["calendar_duration_days"], 2)
        self.assertTrue(pd.isna(e.iloc[1]["inter_onset_days"]))

    def test_absent_calendar_date_is_unknown(self):
        dates = ["2026-07-01", "2026-07-02", "2026-07-04", "2026-07-05"]
        e = daily_episode_table(regime_frame("A", 4, dates, [0, 1, 2, 0]))
        self.assertEqual(len(e), 2)
        self.assertTrue(bool(e.iloc[0]["censored_right"]))
        self.assertTrue(bool(e.iloc[1]["censored_left"]))

    def test_coverage_and_suspect_flags_do_not_invent_events(self):
        dates = pd.date_range("2026-07-01", periods=5, freq="D")
        df = regime_frame("A", 2, dates, [0, 0.5, 2, 1, 0],
                          qc_suspect=[False, False, True, False, False])
        df.loc[2, "coverage"] = 0.94
        e = daily_episode_table(df)
        self.assertEqual(len(e), 2)
        self.assertEqual(e.iloc[0]["hours_below_threshold"], 0.5)
        self.assertTrue(bool(e.iloc[0]["censored_right"]))
        self.assertTrue(bool(e.iloc[1]["censored_left"]))
        # Suspect row excluded due to coverage failure; no automatic exclusion
        # of the other eligible measured events.
        self.assertFalse(bool(e["qc_suspect"].any()))

    def test_thresholds_are_strict_and_independent(self):
        dates = ["2026-07-01", "2026-07-02", "2026-07-03"]
        # On the second day the minimum is exactly 2 mg/L: below 4 but not 2.
        # On the first and last days the minimum is exactly 4 mg/L.
        d = pd.concat([
            regime_frame("A", 2, dates, [0, 0, 0]),
            regime_frame("A", 4, dates, [0, 1 / 12, 0]),
        ], ignore_index=True)
        episodes = daily_episode_table(d)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes.iloc[0]["threshold_mgL"], 4)
        self.assertEqual(episodes.iloc[0]["calendar_duration_days"], 1)
        self.assertTrue(bool(episodes.iloc[0]["observed_recovery"]))

    def test_recurrence_stays_within_station_threshold_and_is_auditable(self):
        dates = pd.date_range("2026-07-01", periods=6, freq="D")
        a = regime_frame("A", 2, dates, [0, 1, 0, 0, 2, 0])
        b = regime_frame("B", 2, dates, [0, 0, 1, 0, 0, 0])
        c = regime_frame("A", 4, dates, [0, 1, 0, 0, 2, 0])
        episodes = daily_episode_table(pd.concat([a, b, c]))
        a2 = episodes.loc[(episodes.node == "A") &
                          (episodes.threshold_mgL == 2)]
        self.assertEqual(len(a2), 2)
        self.assertEqual(a2.iloc[1]["inter_onset_days"], 3)
        self.assertEqual(a2.iloc[1]["nonhypoxic_days_since_prior_run"], 2)
        self.assertEqual(summarize_episodes(episodes).loc[
            lambda x: (x.node == "A") & (x.threshold_mgL == 2),
            "n_identifiable_onsets"].iloc[0], 2)

    def test_record_boundaries_are_censored(self):
        dates = ["2026-07-01", "2026-07-02"]
        episodes = daily_episode_table(regime_frame("A", 2, dates, [1, 3]))
        self.assertEqual(len(episodes), 1)
        self.assertTrue(bool(episodes.iloc[0]["censored_left"]))
        self.assertTrue(bool(episodes.iloc[0]["censored_right"]))
        self.assertEqual(episodes.iloc[0]["left_boundary"], "record_start")
        self.assertEqual(episodes.iloc[0]["right_boundary"], "record_end")
        self.assertEqual(episodes.iloc[0]["provenance"], "observed_only")

    def test_invalid_inputs_rejected(self):
        d = regime_frame("A", 2, ["2026-07-01"], [1])
        with self.assertRaises(ValueError):
            daily_episode_table(pd.concat([d, d]))
        d["eligible"] = "unknown"
        with self.assertRaises(ValueError):
            daily_episode_table(d)

    def test_subdaily_observed_flags_exclude_imputed_placeholders(self):
        times = pd.date_range("2026-07-01T00:00:00Z", periods=6, freq="5min")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            pd.DataFrame({
                "datetime": times.astype(str),
                "odo_mgL": [5, 1, 0, 0, 1, 5],
                "odo_mgL_observed": [1, 1, 0, 0, 1, 1],
                "temp_c": [28] * 6,
                "temp_c_observed": [1] * 6,
                "sal_ppt": [25] * 6,
                "sal_ppt_observed": [1] * 6,
            }).to_csv(directory / "A_imputed.csv", index=False)
            pd.DataFrame({"node": ["A"]}).to_csv(
                directory / "node_provenance.csv", index=False)
            detail = subdaily_episode_table(directory)
            severe = detail.loc[detail.threshold_mgL == 2].reset_index(drop=True)
            self.assertEqual(len(severe), 2)
            self.assertEqual(severe["n_observed_samples"].tolist(), [1, 1])
            self.assertTrue(bool(severe.iloc[0]["censored_right"]))
            self.assertTrue(bool(severe.iloc[1]["censored_left"]))
            self.assertEqual(severe["do_deficit_mgL_hours"].sum(), 1 / 6)
            self.assertTrue((detail["provenance"] == "observed_only").all())


if __name__ == "__main__":
    unittest.main()
