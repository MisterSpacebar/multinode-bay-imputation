"""Regression tests for the final forcing-builder QC handoff.

Tests are synthetic; no NOAA network calls or private station data required.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trustworthy.build_forcing import (
    build_wind,
    daily_aggregate,
    read_rain,
    splice_wind,
)


class ForcingBuilderTests(unittest.TestCase):
    def test_utc_year_boundary_is_one_miami_day(self):
        # December 31, 2025 18:00 UTC is 13:00 EST. The Jan 1 UTC samples
        # before 05:00 belong to December 31 in Miami.
        ndbc = pd.Series(
            [2.0, 4.0],
            index=pd.to_datetime([
                "2025-12-31T23:48Z", "2026-01-01T00:00Z"]),
        )
        coops = pd.Series(
            [6.0, 8.0, 10.0],
            index=pd.to_datetime([
                "2026-01-01T00:00Z",  # overlap; NDBC must prevail
                "2026-01-01T00:06Z",
                "2026-01-01T00:12Z",
            ]),
        )
        series, origins = splice_wind(ndbc, coops)
        self.assertEqual(len(series), 4)
        self.assertEqual(series.loc[pd.Timestamp(
            "2026-01-01T00:00Z")], 4.0)
        daily = daily_aggregate(
            series, lag_hours=3.0, source_id="mixed",
            name="wind_mps", origins=origins)
        date = pd.Timestamp("2025-12-31")
        self.assertIn(date, daily.index)
        self.assertAlmostEqual(daily.loc[date, "wind_mps"], 6.0)
        self.assertEqual(
            daily.loc[date, "wind_mps_source"],
            "ndbc_vakf1+coops_8723214_wind")
        self.assertAlmostEqual(
            daily.loc[date, "wind_mps_coverage"], 4 / 240)
        self.assertFalse(daily.index.duplicated().any())

    def test_build_wind_queries_last_ndbc_utc_date(self):
        ndbc = pd.Series(
            [1.0], index=pd.to_datetime(["2026-01-01T00:00Z"]))
        coops = pd.DataFrame(
            {"Speed": [3.0]},
            index=pd.to_datetime(["2026-01-01T00:06Z"]),
        )
        with patch("trustworthy.build_forcing.fetch_coops", return_value=coops) as mock:
            daily = build_wind(
                ndbc, pd.Timestamp("2025-12-30"),
                pd.Timestamp("2026-01-04"), Path("/not-accessed"),
                lag_hours=3.0)
        args = mock.call_args.args
        self.assertEqual(args[0], "wind")
        self.assertEqual(args[1], pd.Timestamp("2026-01-01"))
        self.assertAlmostEqual(
            daily.loc[pd.Timestamp("2025-12-31"), "wind_mps"], 2.0)

    def test_invalid_ndbc_timestamps_do_not_block_coops(self):
        ndbc = pd.Series(
            [np.nan, 4.0],
            index=pd.to_datetime([
                "2026-01-01T01:00Z", "2025-12-31T23:54Z"]))
        coops = pd.Series(
            [5.0, 6.0],
            index=pd.to_datetime([
                "2026-01-01T00:00Z", "2026-01-01T00:06Z"]))
        combined, sources = splice_wind(ndbc, coops)
        self.assertEqual(len(combined), 3)
        self.assertEqual(
            sources.loc[pd.Timestamp("2026-01-01T00:00Z")],
            "coops_8723214_wind")

    def test_rain_coverage_flag_is_record_presence_only(self):
        with tempfile.TemporaryDirectory() as td:
            rainfall = pd.DataFrame({
                "year": [2025], "month": [6], "day": [5],
                "rain_in": [1.0],
            })
            rainfall.to_csv(
                Path(td) / "rainfall_2025_mar_oct_combined.csv",
                index=False)
            out = read_rain(Path(td), lag_hours=24.0)
        row = out.loc[pd.Timestamp("2025-06-05")]
        self.assertAlmostEqual(row["rain_mm"], 25.4)
        # Numerical field remains backward-compatible with the validator.
        self.assertEqual(row["rain_mm_coverage"], 1.0)
        self.assertEqual(
            row["rain_mm_available_at_utc"],
            pd.Timestamp("2025-06-07T04:00:00Z"),
        )

    def test_dst_day_uses_true_expected_six_minute_slots(self):
        # November 1 2026 has 25 hours in Miami.
        time = pd.to_datetime(
            ["2026-11-01T05:30Z", "2026-11-01T06:30Z"])
        series = pd.Series([1.0, 3.0], index=time)
        out = daily_aggregate(series, 3, "test", "wind_mps")
        self.assertAlmostEqual(
            out.loc[pd.Timestamp("2026-11-01"), "wind_mps_coverage"],
            2 / 250,
        )


if __name__ == "__main__":
    unittest.main()
