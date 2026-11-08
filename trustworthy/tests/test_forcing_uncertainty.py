"""Synthetic tests for publication-time filtering and paired onset uncertainty."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.time_valid_forcing import (
    FIELDS, read_forcing, materialize_lags, materialize,
)
from trustworthy.onset_uncertainty import (
    summarize_uncertainty, paired_cases, validate_predictions,
)


def forcing_row(node: str, date: str, rain: float | None,
                available_utc: str | None, wind=None, level=None):
    start = pd.Timestamp(date).tz_localize("America/New_York")
    end_utc = (start + pd.DateOffset(days=1)).tz_convert("UTC").isoformat()
    record = {"node": node, "date": date, "source_id": "test_era"}
    for field, value in [
        ("rain_mm", rain), ("wind_mps", wind), ("water_level_m", level)
    ]:
        record[field] = value
        record[f"{field}_observed_end_utc"] = end_utc if value is not None else None
        record[f"{field}_available_at_utc"] = (
            available_utc if value is not None else None)
        record[f"{field}_coverage"] = 1.0 if value is not None else None
    return record


def prediction_series(n=100):
    rows = []
    for node in ("bay", "canal"):
        for i, date in enumerate(pd.date_range("2026-01-01", periods=n)):
            label = int(i % 12 in (4, 5, 6))
            event_date = str(date.date()) if label else ""
            for model, improvement in (
                ("history_rate", 0.0), ("logit_do", .03),
                ("logit_physical", .06),
            ):
                prob = (0.25 + improvement) if label else (0.25 - improvement)
                rows.append({
                    "node": node, "issue_date": str(date.date()),
                    "threshold_mgL": 2, "horizon_days": 7,
                    "model": model, "observed_onset": label,
                    "event_date": event_date,
                    "predicted_probability": prob,
                })
    return pd.DataFrame(rows)


class AvailabilityTests(unittest.TestCase):
    def test_midnight_published_day_is_only_available_after_release(self):
        d = pd.DataFrame({
            "node": ["bay"] * 3,
            "date": ["2026-07-01", "2026-07-02", "2026-07-03"],
        })
        # July 1 daily report arrives July 3 at 06:00 UTC, deliberately late.
        f = pd.DataFrame([
            forcing_row("bay", "2026-07-01", 12.0, "2026-07-03T06:00:00Z"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "forcing.csv"
            f.to_csv(filename, index=False)
            values = read_forcing(filename)
        lag = materialize_lags(d, values, lags=(1, 2))
        two = lag.set_index("date")
        self.assertTrue(np.isnan(two.loc[pd.Timestamp("2026-07-02"), "rain_mm_lag1"]))
        self.assertAlmostEqual(two.loc[pd.Timestamp("2026-07-03"), "rain_mm_lag2"], 12.0)
        self.assertTrue(np.isnan(two.loc[pd.Timestamp("2026-07-03"), "wind_mps_lag2"]))
        self.assertTrue(np.isnan(two.loc[pd.Timestamp("2026-07-03"), "water_level_m_lag2"]))

    def test_regional_weather_explicit_and_not_regional_water_level(self):
        d = pd.DataFrame({
            "node": ["bay"], "date": ["2026-07-03"]})
        f = pd.DataFrame([
            forcing_row("regional", "2026-07-01", 10.0,
                        "2026-07-02T10:00:00Z", wind=3.0, level=2.0),
        ])
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "regional.csv"
            f.to_csv(filename, index=False)
            raw = read_forcing(filename)
        off = materialize_lags(d, raw, lags=(2,), allow_regional_weather=False)
        on = materialize_lags(d, raw, lags=(2,), allow_regional_weather=True)
        self.assertTrue(np.isnan(off.loc[0, "rain_mm_lag2"]))
        self.assertAlmostEqual(on.loc[0, "rain_mm_lag2"], 10)
        self.assertAlmostEqual(on.loc[0, "wind_mps_lag2"], 3)
        self.assertTrue(np.isnan(on.loc[0, "water_level_m_lag2"]))

    def test_does_not_substitute_region_for_late_local_report(self):
        d = pd.DataFrame({"node": ["bay"], "date": ["2026-07-03"]})
        f = pd.DataFrame([
            forcing_row("bay", "2026-07-01", 20.,
                        "2026-07-04T10:00:00Z"),
            forcing_row("regional", "2026-07-01", 9.,
                        "2026-07-02T10:00:00Z"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "forcing.csv"
            f.to_csv(filename, index=False)
            raw = read_forcing(filename)
        on = materialize_lags(d, raw, lags=(2,), allow_regional_weather=True)
        self.assertTrue(np.isnan(on.loc[0, "rain_mm_lag2"]))

    def test_invalid_release_before_observation_rejected(self):
        f = pd.DataFrame([forcing_row(
            "bay", "2026-07-01", 10.0, "2026-07-01T01:00:00Z")])
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "forcing.csv"
            f.to_csv(filename, index=False)
            with self.assertRaises(ValueError):
                read_forcing(filename)

    def test_dst_issue_end_is_local_midnight(self):
        d = pd.DataFrame({"node": ["bay"], "date": ["2026-11-02"]})
        f = pd.DataFrame([forcing_row(
            "bay", "2026-11-01", 2., "2026-11-02T04:30:00Z")])
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "forcing.csv"
            f.to_csv(filename, index=False)
            raw = read_forcing(filename)
        # Nov 2 starts 05:00 UTC due to DST change. Publication 04:30
        # precedes issue-day end, so lag1 should be available.
        a = materialize_lags(d, raw, lags=(1,))
        self.assertEqual(a.loc[0, "rain_mm_lag1"], 2.)

    def test_duplicate_source_day_is_not_silently_averaged(self):
        f = pd.DataFrame([
            forcing_row("bay", "2026-07-01", 1., "2026-07-02T10:00:00Z"),
            forcing_row("bay", "2026-07-01", 2., "2026-07-02T10:00:00Z"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "forcing.csv"
            f.to_csv(filename, index=False)
            with self.assertRaises(ValueError):
                read_forcing(filename)


class UncertaintyTests(unittest.TestCase):
    def test_candidate_and_reference_matched_cases(self):
        p = validate_predictions(prediction_series())
        match = paired_cases(p, "logit_physical", "history_rate")
        self.assertEqual(len(match), 200)
        self.assertTrue((match["candidate_sq"] < match["reference_sq"]).all())

    def test_uncertainty_reports_horizon_and_distinct_events(self):
        p = prediction_series()
        r = summarize_uncertainty(p, n_bootstrap=100,
                                  schemes=("month", "14day"))
        self.assertEqual(len(r), 6)
        self.assertTrue((r["n_forecast_windows"] == 200).all())
        self.assertTrue((r["n_distinct_onsets"] > 0).all())
        self.assertTrue((r["n_distinct_onsets"] <= r["n_positive_windows"]).all())
        positive = r[r["candidate"].eq("logit_physical")
                     & r["reference"].eq("history_rate")]
        self.assertTrue((positive["delta_brier"] < 0).all())

    def test_duplicate_issue_model_rejected(self):
        p = prediction_series()
        with self.assertRaises(ValueError):
            validate_predictions(pd.concat([p, p.iloc[[0]]], ignore_index=True))

    def test_low_n_blocks_does_not_claim_interval(self):
        p = prediction_series(n=5)
        r = summarize_uncertainty(p, n_bootstrap=100,
                                  min_station_blocks=50)
        self.assertTrue(r["interval_status"].eq("too_few_time_blocks").all())
        self.assertTrue(r["brier_skill_ci_low_approx"].isna().all())


if __name__ == "__main__":
    unittest.main()
