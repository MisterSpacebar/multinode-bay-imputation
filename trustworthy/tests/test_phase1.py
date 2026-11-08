"""Phase 1 regression tests. Synthetic signals only: no private measurements."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trustworthy.quality import Config, read_station, daily_targets
from trustworthy.events import EventConfig, detect_quality_events
from trustworthy.targets import detect_hypoxia_events
from trustworthy.backtest import evaluate
from preprocess import build_station_sources


def write_export(path, dates, values, flags=None):
    n = len(values)
    frame = pd.DataFrame({
        "datetime": dates,
        "odo_mgL": values,
        "odo_mgL_observed": [int(v) for v in (flags if flags is not None else [True] * n)],
        "temp_c": [28.] * n,
        "temp_c_observed": [1] * n,
        "sal_ppt": [25.] * n,
        "sal_ppt_observed": [1] * n,
    })
    frame.to_csv(path, index=False)


class PhaseOneTests(unittest.TestCase):

    def test_missing_intervals_are_outages_and_not_low_oxygen(self):
        dates = pd.date_range("2026-07-01", periods=40, freq="5min", tz="UTC")
        # A 2h gap of absent observed readings; imputed placeholder values
        # could look severe but must not be used as measured DO.
        values = [5.] * 40
        flags = [True] * 40
        for i in range(8, 32):
            values[i] = 0.
            flags[i] = False
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "node_imputed.csv"
            write_export(path, dates, values, flags)
            qc = read_station(path)
            events = detect_quality_events(qc, "node")
            out = [e for e in events if e["event_type"] == "outage"]
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["duration_minutes"], 120)
            self.assertFalse(detect_hypoxia_events(qc, "node"))

    def test_steady_low_oxygen_is_flagged_not_erased(self):
        dates = pd.date_range("2026-07-01", periods=32, freq="5min", tz="UTC")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "node_imputed.csv"
            write_export(path, dates, [0.0] * 32)
            qc = read_station(path)
            self.assertTrue(qc["do_flatline"].any())
            self.assertTrue(qc["do_valid"].eq(0).all())
            hypoxia = detect_hypoxia_events(qc, "node")
            self.assertEqual(len(hypoxia), 2)
            self.assertEqual(hypoxia[0]["min_odo_mgL"], 0)
            self.assertTrue(any(e["event_type"] == "flatline_suspect"
                                for e in detect_quality_events(qc, "node")))

    def test_sustained_low_oxygen_not_classified_as_isolated_spike(self):
        dates = pd.date_range("2026-07-01", periods=40, freq="5min", tz="UTC")
        vals = [5.] * 8 + [1.] * 16 + [5.] * 16
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "node_imputed.csv"
            write_export(path, dates, vals)
            qc = read_station(path)
            self.assertFalse(qc["do_spike_suspect"].any())
            self.assertEqual(len(detect_hypoxia_events(qc, "node")), 2)

    def test_coverage_blocks_low_sample_days(self):
        dates = pd.date_range("2026-07-01", periods=40, freq="5min", tz="UTC")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "node_imputed.csv"
            write_export(path, dates, [1.] * len(dates))
            qc = read_station(path)
            daily = daily_targets(qc, "node")
            self.assertFalse(daily["eligible"].any())
            self.assertTrue(daily["hours_lt4"].isna().all())

    def test_source_separation_even_if_coordinates_identical(self):
        dates = pd.date_range("2026-07-01", periods=2, freq="5min", tz="UTC")
        a = {"name": "A", "lat": 25.8, "lon": -80.1,
             "df": pd.DataFrame({"odo_mgL": [2., 2.]}, index=dates)}
        b = {"name": "B", "lat": 25.8, "lon": -80.1,
             "df": pd.DataFrame({"odo_mgL": [8., 8.]}, index=dates)}
        with patch("preprocess.load_2025_stations", return_value=[a]), \
             patch("preprocess.load_2026_stations", return_value=[b]), \
             patch("preprocess.load_consolidated_2026", return_value=[]):
            result = build_station_sources()
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0]["name"], result[1]["name"])
        self.assertEqual(result[0]["df"]["odo_mgL"].iloc[0], 2)
        self.assertEqual(result[1]["df"]["odo_mgL"].iloc[0], 8)

    def test_backtest_works_without_deployment_metadata(self):
        dates = pd.date_range("2026-01-01", periods=280, freq="D")
        daily = pd.DataFrame({
            "node": "s", "date": dates,
            "hours_lt4": np.arange(len(dates)) % 4,
            "odo_mean": 5., "odo_min": 3., "odo_range": 2.,
            "predawn_min": 3., "temp_mean": 28., "sal_mean": 25.,
        })
        result = evaluate(daily, nodes=["s"], horizons=(1,),
                          test_months=("2026-09",), min_train=40)
        self.assertFalse(result.empty)
        self.assertTrue((pd.to_datetime(result["target_date"]) >
                         pd.to_datetime(result["issue_date"])).all())
        self.assertTrue({"persistence", "trail7", "ridge", "train_mean",
                         "ridge_do", "ridge_physical"}.issubset(set(result["model"])))


if __name__ == "__main__":
    unittest.main()
