"""Synthetic coverage, state, and time-valid onset regression tests."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trustworthy.regimes import (
    regime_days, regime_summary, stratified_forecast_scores,
)
from trustworthy.onset import (
    build_windows, backtest, fit_logistic_ridge, average_precision,
    paired_risk_scores, summarize,
)


def daily_signal(n=220, start="2026-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    # Transient low-oxygen events every ~14 days with a few longer episodes.
    state = np.zeros(n, dtype=float)
    for k in range(5, n, 14):
        state[k:min(n, k + 2)] = 2.0
    deep = np.zeros(n)
    for k in range(20, n, 42):
        deep[k] = 0.5
    state = np.maximum(state, deep)  # severe oxygen depletion implies <4mg/L
    do_min = np.where(deep > 0, 1.5, np.where(state > 0, 3.0, 6.0))
    return pd.DataFrame({
        "node": "bay_A", "date": idx.strftime("%Y-%m-%d"),
        "eligible": True, "coverage": 1.0,
        "hours_lt4": state, "hours_lt2": deep,
        "odo_min": do_min, "odo_mean": do_min + 1.5,
        "odo_range": 2., "temp_mean": 27 + np.sin(np.arange(n)/8),
        "sal_mean": 24 + np.cos(np.arange(n)/9),
        "n_flatline": 0, "n_spike_suspect": 0, "n_zero_review": 0,
    })


class RegimeTests(unittest.TestCase):
    def test_four_regimes_and_first_day_unknown(self):
        d = daily_signal(n=7)
        out = regime_days(d, thresholds=(4,))
        self.assertEqual(out["regime"].tolist(), [
            "unknown", "normal", "normal", "normal", "normal",
            "onset", "persistent_hypoxia",
        ])
        summary = regime_summary(out)
        self.assertEqual(summary["n_days"].sum(), 7)

    def test_gap_breaks_transition_and_does_not_become_recovery(self):
        d = daily_signal(n=9)
        d.loc[5, ["eligible", "coverage", "hours_lt4", "hours_lt2"]] = [
            False, .2, np.nan, np.nan]
        out = regime_days(d, thresholds=(4,))
        self.assertEqual(out.loc[5, "regime"], "unknown")
        self.assertEqual(out.loc[6, "regime"], "unknown")
        self.assertEqual(out.loc[7, "regime"], "recovery")

    def test_recovery_normal_and_onset_from_contiguous_history(self):
        d = daily_signal(n=9)
        out = regime_days(d, thresholds=(4,))
        self.assertEqual(out.loc[5, "regime"], "onset")
        self.assertEqual(out.loc[6, "regime"], "persistent_hypoxia")
        self.assertEqual(out.loc[7, "regime"], "recovery")
        self.assertEqual(out.loc[8, "regime"], "normal")

    def test_stratification_is_retrospective_not_feature(self):
        d = daily_signal(n=11)
        r = regime_days(d)
        forecast = pd.DataFrame([{
            "node": "bay_A",
            "issue_date": d.loc[4, "date"],
            "target_date": d.loc[5, "date"],
            "horizon_days": 1,
            "model": "persistence",
            "actual_hours_lt4": 2.,
            "predicted_hours_lt4": 0.,
            "persistence_prediction": 0.,
        }])
        report = stratified_forecast_scores(r, forecast)
        self.assertEqual(report.iloc[0]["regime"], "onset")
        self.assertEqual(report.iloc[0]["n_cases"], 1)


class OnsetTests(unittest.TestCase):
    def test_no_lookahead_or_unknown_future_as_negative(self):
        d = daily_signal(n=12)
        d.loc[6, ["eligible", "coverage", "hours_lt4", "hours_lt2"]] = [
            False, 0.15, np.nan, np.nan]
        windows = build_windows(d, "bay_A", 4, horizon=3, normal_days=2)
        by_date = windows.set_index("issue_date")
        self.assertEqual(by_date.loc["2026-01-04", "reason"],
                         "future_low_coverage")
        self.assertTrue(np.isnan(by_date.loc["2026-01-04", "label"]))
        self.assertEqual(by_date.loc["2026-01-05", "reason"],
                         "future_low_coverage")
        self.assertEqual(by_date.loc["2026-01-06", "reason"],
                         "already_below_threshold")
        self.assertEqual(by_date.loc["2026-01-11", "reason"],
                         "incomplete_future_window")

    def test_onset_requires_preceding_normal_history(self):
        d = daily_signal(n=16)
        windows = build_windows(d, "bay_A", 4, horizon=3, normal_days=2)
        by_date = windows.set_index("issue_date")
        self.assertEqual(by_date.loc["2026-01-04", "label"], 1)
        self.assertEqual(by_date.loc["2026-01-04", "event_date"], "2026-01-06")
        self.assertEqual(by_date.loc["2026-01-06", "reason"],
                         "already_below_threshold")
        self.assertEqual(by_date.loc["2026-01-08", "reason"],
                         "already_below_threshold")
        self.assertEqual(by_date.loc["2026-01-09", "label"], 0)

    def test_threshold_2_respects_its_own_at_risk_condition(self):
        d = daily_signal(n=45)
        # 4mg/L depleted but not 2mg/L is still at risk for severe onset
        d.loc[5:8, "hours_lt2"] = 0.0
        severe = build_windows(d, "bay_A", 2, horizon=3)
        moderate = build_windows(d, "bay_A", 4, horizon=3)
        self.assertEqual(severe.set_index("issue_date").loc[
            "2026-01-07", "reason"], "evaluable")
        self.assertEqual(moderate.set_index("issue_date").loc[
            "2026-01-07", "reason"], "already_below_threshold")

    def test_train_windows_end_before_issue_and_future_mutation_invariance(self):
        d = daily_signal()
        kwargs = dict(thresholds=(4,), horizons=(3,), min_train=15,
                      min_events=2, months=("2026-06",))
        original, _, _ = backtest(d, **kwargs)
        self.assertFalse(original.empty)
        # Alter future targets only; earlier forecast origins must not change.
        pivot = pd.Timestamp("2026-06-20")
        changed = d.copy()
        mask = pd.to_datetime(changed["date"]) >= pivot
        changed.loc[mask, "hours_lt4"] = 0.0
        changed.loc[mask, "hours_lt2"] = 0.0
        altered, _, _ = backtest(changed, **kwargs)
        a = original.loc[pd.to_datetime(original["issue_date"]) <
                         pivot - pd.Timedelta(days=3)]
        b = altered.loc[pd.to_datetime(altered["issue_date"]) <
                        pivot - pd.Timedelta(days=3)]
        joined = a.merge(b, on=["node", "issue_date", "horizon_days",
                               "threshold_mgL", "model"], suffixes=("_a", "_b"))
        self.assertGreater(len(joined), 0)
        self.assertTrue(np.allclose(joined["predicted_probability_a"],
                                    joined["predicted_probability_b"]))

    def test_logistic_fit_and_paired_brier(self):
        d = daily_signal()
        p, _, _ = backtest(
            d, thresholds=(4,), horizons=(1,), min_train=20,
            min_events=2, months=("2026-06",))
        self.assertIn("history_rate", set(p["model"]))
        self.assertIn("logit_do", set(p["model"]))
        self.assertTrue(p["predicted_probability"].between(0, 1).all())
        summary = summarize(p)
        self.assertTrue((summary["n_cases"] > 0).all())
        paired = paired_risk_scores(p)
        self.assertFalse(paired.empty)
        self.assertTrue((paired["n_paired"] > 0).all())

    def test_sparse_event_conditions_do_not_invent_predictions(self):
        d = daily_signal(n=80)
        d["hours_lt2"] = 0.0
        predictions, excluded, _ = backtest(
            d, thresholds=(2,), horizons=(1,), min_train=10,
            min_events=3)
        self.assertTrue(predictions["model"].isin(["history_rate"]).all())
        self.assertTrue((excluded.loc[
            excluded["model"].eq("logit_do") &
            excluded["reason"].eq("insufficient_event_classes"),
        ].shape[0] > 0))

    def test_average_precision_with_ties_and_no_positives(self):
        self.assertTrue(np.isnan(average_precision([0, 0], [.1, .2])))
        self.assertAlmostEqual(average_precision([1, 0], [.1, .1]), .5)
        self.assertAlmostEqual(average_precision([0, 1], [.1, .9]), 1.0)

    def test_logistic_model_rejects_unobserved_inputs(self):
        d = pd.DataFrame({
            "odo_mean": [5.0] * 6, "odo_min": [4.0] * 6,
            "label": [0, 1, 0, 1, 0, 1],
        })
        with self.assertRaises(ValueError):
            fit_logistic_ridge(d, pd.Series({
                "odo_mean": np.nan, "odo_min": 4.0}),
                ("odo_mean", "odo_min"))


if __name__ == "__main__":
    unittest.main()
