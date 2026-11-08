"""Synthetic causal-timing and fair-evaluation regression checks for FCM v2."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fcm_v2.model import (
    DO_CONCEPTS, PHYSICAL_CONCEPTS, BOUNDS,
    fit_map, prepare_targets, edge_rows, sensitivity_rows,
)
from fcm_v2.evaluate import (
    evaluate, summarize, compare_on_common_cases,
)
from trustworthy.backtest import evaluate as baselines, prepare


def make_daily(n=190, missing_physical=False):
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    rng = np.random.default_rng(13)
    temp = 28 + 1.5 * np.sin(np.arange(n) / 14)
    sal = 25 + 3.0 * np.cos(np.arange(n) / 20)
    h = np.clip(1.5 + 0.4 * np.roll(temp - 28, 1)
                - 0.1 * (sal - 25)
                + rng.normal(0, 0.3, n), 0, 24)
    data = pd.DataFrame({
        "node": "bay_a", "date": dates,
        "hours_lt4": h,
        "odo_mean": 6.2 - h / 3,
        "odo_min": 5.5 - h / 2.5,
        "odo_range": 1 + h / 5,
        "predawn_min": 5.0 - h / 3,
        "temp_mean": temp,
        "sal_mean": sal,
        "eligible": True,
    })
    if missing_physical:
        data[["temp_mean", "sal_mean"]] = np.nan
    return data


class PhysicalFCMV2Tests(unittest.TestCase):
    def test_fit_scaling_is_train_only(self):
        d = make_daily()
        frame = prepare_targets(
            prepare(d, "bay_a", 3), PHYSICAL_CONCEPTS, 3)
        issue = pd.Timestamp("2026-05-10")
        model = fit_map(frame, PHYSICAL_CONCEPTS, 3, issue,
                        min_train=20, activation="linear")
        self.assertIsNotNone(model)
        pred = model.predict(frame.loc[issue, list(PHYSICAL_CONCEPTS)])
        w0, center0, intercept0 = (
            model.weights.copy(), model.center.copy(),
            model.intercept.copy())
        # Perturb values in the future; both fitted weights and train-only
        # standardization must remain exactly the same.
        d.loc[d["date"] >= pd.Timestamp("2026-05-12"),
              ["hours_lt4", "odo_mean", "temp_mean", "sal_mean"]] = [
                  23., 1., 42., 4.]
        later_frame = prepare_targets(
            prepare(d, "bay_a", 3), PHYSICAL_CONCEPTS, 3)
        later = fit_map(later_frame, PHYSICAL_CONCEPTS, 3, issue,
                        min_train=20, activation="linear")
        self.assertTrue(np.allclose(w0, later.weights, equal_nan=True))
        self.assertTrue(np.allclose(center0, later.center))
        self.assertTrue(np.allclose(intercept0, later.intercept, equal_nan=True))
        self.assertTrue(np.allclose(
            pred, later.predict(later_frame.loc[issue, list(PHYSICAL_CONCEPTS)]),
            equal_nan=True))

    def test_sigmoid_fits_same_link_function_it_uses(self):
        d = make_daily()
        f = prepare_targets(prepare(d, "bay_a", 1), DO_CONCEPTS, 1)
        model = fit_map(f, DO_CONCEPTS, 1, pd.Timestamp("2026-06-05"),
                        activation="sigmoid", min_train=20)
        self.assertIsNotNone(model)
        for date in (pd.Timestamp("2026-05-15"), pd.Timestamp("2026-06-01")):
            values = model.predict(f.loc[date, list(DO_CONCEPTS)])
            self.assertTrue(np.isfinite(values).all())
            for i, concept in enumerate(DO_CONCEPTS):
                self.assertGreaterEqual(values[i], BOUNDS[concept][0])
                self.assertLessEqual(values[i], BOUNDS[concept][1])

    def test_forecasts_require_observed_target_and_issue(self):
        d = make_daily(missing_physical=True)
        missing = pd.Timestamp("2026-06-06")
        d.loc[d["date"] == missing, "hours_lt4"] = np.nan
        p, exclusions, _, _ = evaluate(
            d, horizons=(1,), months=("2026-06",),
            activation="linear", min_train=20, include_edges=False)
        self.assertFalse(p.empty)
        self.assertEqual(set(p["model"]), {"fcm_linear_do"})
        self.assertFalse((p["target_date"] == "2026-06-06").any())
        miss = exclusions[exclusions["target_date"] == "2026-06-06"]
        self.assertTrue((miss["reason"] == "missing_target").all())
        self.assertTrue((exclusions[
            exclusions["model"] == "fcm_linear_physical"]["reason"]
            == "missing_model_features").any())

    def test_pairwise_persistence_and_common_case(self):
        d = make_daily()
        p, _, _, _ = evaluate(
            d, horizons=(1,), months=("2026-06",),
            activation="linear", min_train=20, include_edges=False)
        summary = summarize(p)
        for _, entry in summary.iterrows():
            group = p[p["model"] == entry["model"]]
            y = group["actual_hours_lt4"].to_numpy(float)
            b = group["persistence_prediction"].to_numpy(float)
            expected = np.sqrt(np.mean((y - b)**2))
            self.assertAlmostEqual(entry["paired_persistence_rmse"], expected)
        b = baselines(d, horizons=(1,), test_months=("2026-06",),
                      min_train=20)
        common = compare_on_common_cases(p, b)
        self.assertFalse(common.empty)
        counts = common.groupby(["node", "test_month", "horizon_days"])["n_test"]
        self.assertTrue(all(s.nunique() == 1 for _, s in counts))

    def test_future_target_dates_stay_strictly_before_fit(self):
        d = make_daily()
        h, issue = 7, pd.Timestamp("2026-05-03")
        f = prepare_targets(prepare(d, "bay_a", h), DO_CONCEPTS, h)
        valid = (f["target_date"] < issue) & f["target_hours_lt4"].notna()
        self.assertTrue((f.loc[valid, "target_date"] < issue).all())
        m = fit_map(f, DO_CONCEPTS, h, issue, min_train=20)
        self.assertIsNotNone(m)
        self.assertLessEqual(m.n_target_pairs[0], valid.sum())

    def test_edges_and_sensitivities_are_labeled_noncausal(self):
        d = make_daily()
        f = prepare_targets(
            prepare(d, "bay_a", 1), PHYSICAL_CONCEPTS, 1)
        issue = pd.Timestamp("2026-05-10")
        m = fit_map(f, PHYSICAL_CONCEPTS, 1, issue,
                    activation="linear", min_train=20)
        self.assertIsNotNone(m)
        edges = edge_rows(m, "bay_a", issue)
        self.assertTrue(edges)
        self.assertTrue(all(e["interpretation"] ==
                            "lagged_association_not_causal" for e in edges))
        changes = sensitivity_rows(
            m, "bay_a", issue,
            f.loc[issue, list(PHYSICAL_CONCEPTS)].to_numpy(float))
        self.assertEqual(len(changes), 2)
        self.assertTrue(all(
            x["interpretation"] == "model_input_sensitivity_not_causal"
            for x in changes))


if __name__ == "__main__":
    unittest.main()
