"""Synthetic checks for forecast coverage, paired skill and FCM-ready QA."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.backtest import (
    evaluate, eligible_forecasts, prepare, summarize, MODEL_FEATURES,
)
from trustworthy.coverage_report import coverage_report


def daily_frame(start="2026-01-01", n=300, missing_physical=False):
    dates = pd.date_range(start, periods=n, freq="D")
    hours = (np.arange(n) % 7 == 0).astype(float) * 2.0
    frame = pd.DataFrame({
        "node": "node_A", "date": dates,
        "eligible": True, "coverage": 1.0,
        "hours_lt4": hours, "hours_lt2": np.zeros(n),
        "odo_mean": 6.0 - hours / 4,
        "odo_min": 5.0 - hours / 2,
        "odo_range": 1.0 + hours / 2,
        "predawn_min": 5.0 - hours / 2,
        "temp_mean": np.full(n, 29.0),
        "sal_mean": np.full(n, 29.0),
        "n_flatline": np.zeros(n), "n_spike_suspect": np.zeros(n),
    })
    if missing_physical:
        frame["temp_mean"] = np.nan
        frame["sal_mean"] = np.nan
    return frame


class CoverageAndAblationTests(unittest.TestCase):
    def test_do_models_survive_missing_physical_predictors(self):
        data = daily_frame(missing_physical=True)
        pred, diag = eligible_forecasts(
            data, horizons=(1,), test_months=("2026-09",), min_train=40)
        self.assertFalse(pred.empty)
        self.assertTrue({"persistence", "trail7", "train_mean",
                         "ridge_do"}.issubset(set(pred["model"])))
        self.assertFalse({"ridge_physical", "ridge"} & set(pred["model"]))
        self.assertTrue(((diag["model"] == "ridge_physical") &
                         (diag["reason"] == "missing_model_features")).any())

    def test_paired_persistence_skill_is_on_identical_dates(self):
        data = daily_frame()
        data.loc[data["date"] == pd.Timestamp("2026-09-04"), "predawn_min"] = np.nan
        pred = evaluate(data, horizons=(1,), test_months=("2026-09",),
                        min_train=40)
        summary = summarize(pred)
        for _, s in summary.iterrows():
            chosen = pred.loc[
                (pred["node"] == s["node"]) &
                (pred["test_month"] == s["test_month"]) &
                (pred["horizon_days"] == s["horizon_days"]) &
                (pred["model"] == s["model"])
            ]
            p = chosen["persistence_prediction"].to_numpy(float)
            y = chosen["actual_hours_lt4"].to_numpy(float)
            self.assertEqual(int(s["n_test"]), len(chosen))
            self.assertAlmostEqual(
                s["paired_persistence_rmse"], float(np.sqrt(np.mean((y-p)**2))))
        self.assertTrue((summary["n_test"] > 0).all())

    def test_training_targets_never_reach_issue_date(self):
        data = daily_frame()
        # Change a future target. Predictions issued before the change cannot
        # use its value in the fit.
        base = evaluate(data, horizons=(3,), test_months=("2026-09",),
                        min_train=40)
        change_date = pd.Timestamp("2026-09-20")
        data.loc[data["date"] == change_date, "hours_lt4"] = 24.0
        changed = evaluate(data, horizons=(3,), test_months=("2026-09",),
                           min_train=40)
        keys = ["node", "issue_date", "target_date", "model"]
        earlier = base[
            pd.to_datetime(base["issue_date"]) < change_date - pd.Timedelta(days=3)
        ].merge(changed, on=keys, suffixes=("_old", "_new"))
        self.assertGreater(len(earlier), 0)
        self.assertTrue(np.allclose(
            earlier["predicted_hours_lt4_old"],
            earlier["predicted_hours_lt4_new"]))

    def test_missing_target_not_converted_to_non_event(self):
        data = daily_frame()
        data.loc[data["date"] == pd.Timestamp("2026-09-12"), "hours_lt4"] = np.nan
        pred, diag = eligible_forecasts(
            data, horizons=(1,), test_months=("2026-09",),
            min_train=40)
        self.assertFalse((pred["target_date"] == "2026-09-12").any())
        self.assertTrue((diag.loc[
            diag["target_date"].eq("2026-09-12"), "reason"
        ] == "missing_target").all())

    def test_coverage_reports_partial_months_and_missing_covariates(self):
        data = daily_frame("2026-08-28", n=12, missing_physical=True)
        data.loc[data["date"] == pd.Timestamp("2026-09-03"),
                 ["hours_lt4", "hours_lt2", "eligible"]] = [np.nan, np.nan, False]
        monthly = coverage_report(data)
        self.assertEqual(set(monthly["month"]), {"2026-08", "2026-09"})
        sept = monthly.set_index("month").loc["2026-09"]
        self.assertEqual(int(sept["missing_do_days"]), 1)
        self.assertEqual(int(sept["temperature_available_days"]), 0)
        self.assertEqual(int(sept["eligible_days"]), 7)

    def test_each_ablation_uses_its_own_requirements(self):
        self.assertNotIn("temp_mean", MODEL_FEATURES["ridge_do"])
        self.assertIn("temp_mean", MODEL_FEATURES["ridge_physical"])
        self.assertIn("predawn_min", MODEL_FEATURES["ridge"])

    def test_no_pre_measurement_extrapolation(self):
        data = daily_frame("2026-09-10", n=10)
        pred, diag = eligible_forecasts(
            data, horizons=(7,), test_months=("2026-09",), min_train=40)
        # Persistence needs no fitted model and can still forecast once an
        # issue-day observation exists. Ridge cannot with <40 train examples.
        self.assertEqual(set(pred["model"]), {"persistence", "trail7"})
        self.assertTrue((pd.to_datetime(pred["issue_date"]) >=
                         pd.Timestamp("2026-09-10")).all())
        self.assertTrue((diag["reason"].isin([
            "no_issue_history", "insufficient_training",
            "missing_model_features", "missing_issue_do",
        ])).any())


if __name__ == "__main__":
    unittest.main()
