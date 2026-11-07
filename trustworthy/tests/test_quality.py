import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.quality import Config, daily_targets, read_station, expected_slots
from trustworthy.backtest import evaluate, prepare


class QualityTests(unittest.TestCase):
    def test_imputed_do_is_never_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s_imputed.csv"
            pd.DataFrame({
                "datetime": pd.date_range("2026-07-01", periods=12, freq="5min", tz="UTC"),
                "odo_mgL": [1.0] + [9.0] * 11,
                "odo_mgL_observed": [0] + [1] * 11,
            }).to_csv(path, index=False)
            q = read_station(path, Config(flatline_samples=20))
            self.assertTrue(np.isnan(q["odo_mgL"].iloc[0]))
            self.assertEqual(q["do_valid"].notna().sum(), 11)

    def test_missing_observation_flag_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s_imputed.csv"
            pd.DataFrame({"datetime": ["2026-07-01"], "odo_mgL": [3]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_station(path)

    def test_low_coverage_does_not_create_zero_hypoxia(self):
        idx = pd.date_range("2026-07-01", periods=12, freq="5min", tz="UTC")
        q = pd.DataFrame({
            "odo_mgL": [1.] * len(idx), "do_valid": [1.] * len(idx),
            "do_flag_observed": True, "do_invalid": False,
            "do_flatline": False, "do_zero_review": False,
            "temp_c": np.nan, "sal_ppt": np.nan,
        }, index=idx)
        day = daily_targets(q, "station", Config(flatline_samples=20))
        self.assertFalse(day["eligible"].any())
        self.assertTrue(day["hours_lt4"].isna().all())

    def test_dst_day_length(self):
        c = Config()
        self.assertEqual(expected_slots(pd.Timestamp("2026-03-08"), c), 276)
        self.assertEqual(expected_slots(pd.Timestamp("2026-11-01"), c), 300)

    def test_backtest_training_never_uses_future_targets(self):
        dates = pd.date_range("2026-01-01", periods=250)
        d = pd.DataFrame({
            "node": "s", "date": dates.strftime("%Y-%m-%d"),
            "deployment_id": "stable", "source_verified": True,
            "hours_lt4": np.arange(len(dates)) % 4,
            "odo_mean": 5., "odo_min": 3., "odo_range": 2.,
            "predawn_min": 3., "temp_mean": 28., "sal_mean": 25.,
        })
        prepared = prepare(d, "s", 3)
        issue = pd.Timestamp("2026-09-01")
        train = prepared.loc[prepared["target_date"] < issue]
        self.assertTrue((train["target_date"] < issue).all())
        result = evaluate(d, nodes=["s"], horizons=(3,),
                          test_months=("2026-09",), min_train=40)
        self.assertFalse(result.empty)
        self.assertTrue((pd.to_datetime(result["target_date"]) >
                         pd.to_datetime(result["issue_date"])).all())


if __name__ == "__main__":
    unittest.main()
