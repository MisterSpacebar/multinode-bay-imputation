"""Synthetic tests for paired uncertainty and coefficient diagnostics."""
import unittest
import numpy as np
import pandas as pd

from fcm_v2.diagnostics import (
    compare_models, edge_stability, load_predictions, summarize_stable_edges,
    leave_one_station_out, sensitivity_audit,
)


def synthetic_predictions():
    rows = []
    for node in ("canal", "bay"):
        for i, target in enumerate(pd.date_range("2026-05-01", periods=64)):
            issue = target - pd.Timedelta(days=7)
            y = float(i % 11)
            for model, offset in (
                ("fcm_linear_physical", 0.7),
                ("fcm_linear_do", 1.1),
                ("ridge_physical", 0.65),
                ("persistence", 1.3),
            ):
                rows.append({
                    "node": node, "issue_date": str(issue.date()),
                    "target_date": str(target.date()), "horizon_days": 7,
                    "model": model, "actual_hours_lt4": y,
                    "predicted_hours_lt4": y + offset,
                })
    df = pd.DataFrame(rows)
    return df[df.model.str.startswith("fcm_")], df[
        df.model.isin(["ridge_physical", "persistence"])
    ]


def synthetic_edges():
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    rows = []
    for node in ("canal", "bay"):
        for n, date in enumerate(dates):
            for source, value in (
                ("temp_mean", 0.4 + 0.01 * np.sin(n / 10)),
                ("sal_mean", (0.3 if n < 50 else -0.3)),
            ):
                rows.append({
                    "node": node, "horizon_days": 7, "activation": "linear",
                    "concept_set": "physical", "source": source,
                    "target": "hours_lt4", "asof_date": str(date.date()),
                    "weight": value,
                })
    return pd.DataFrame(rows)


class EdgeDiagnosticsTests(unittest.TestCase):
    def test_paired_forecast_error_and_clusters(self):
        fcm, base = synthetic_predictions()
        combined = load_predictions(fcm, base)
        out = compare_models(combined, n_boot=100, seed=3)
        self.assertEqual(len(out), 3)
        row = out.loc[out["reference"] == "persistence"].iloc[0]
        self.assertEqual(row["n_cases"], 128)
        self.assertEqual(row["n_stations"], 2)
        self.assertEqual(row["n_station_month_blocks"], 6)
        self.assertAlmostEqual(row["candidate_rmse_hours"], 0.7)
        self.assertAlmostEqual(row["reference_rmse_hours"], 1.3)
        self.assertLess(row["delta_rmse_hours"], 0)
        self.assertLess(row["delta_rmse_ci_high_approx"], 0)

    def test_duplicate_predictions_are_rejected(self):
        fcm, base = synthetic_predictions()
        with self.assertRaises(ValueError):
            load_predictions(pd.concat([fcm, fcm.iloc[[0]]]), base)

    def test_inconsistent_ground_truth_rejected(self):
        fcm, base = synthetic_predictions()
        base = base.copy()
        base.loc[base.index[0], "actual_hours_lt4"] = 20.0
        with self.assertRaises(ValueError):
            load_predictions(fcm, base)

    def test_distinguishes_stable_sign_from_switching_edge(self):
        rows = edge_stability(synthetic_edges())
        stable = rows[rows["source"] == "temp_mean"]
        flipping = rows[rows["source"] == "sal_mean"]
        self.assertTrue(stable["stable_exploratory"].all())
        self.assertFalse(flipping["stable_exploratory"].any())
        self.assertTrue((stable["n_origin_months"] >= 3).all())
        self.assertTrue((flipping["n_significant_sign_flips"] >= 1).all())
        summary = summarize_stable_edges(rows)
        temp = summary.loc[summary["source"] == "temp_mean"].iloc[0]
        self.assertEqual(temp["n_stable_series"], 2)
        self.assertEqual(temp["n_station_series"], 2)

    def test_duplicate_edges_fail(self):
        e = synthetic_edges()
        with self.assertRaises(ValueError):
            edge_stability(pd.concat([e, e.iloc[[0]]]))

    def test_leave_one_station_out_reports_matched_counts(self):
        fcm, base = synthetic_predictions()
        results = leave_one_station_out(load_predictions(fcm, base))
        self.assertEqual(len(results), 6)
        self.assertTrue((results["remaining_stations"] == 1).all())
        self.assertTrue((results["remaining_cases"] == 64).all())
        self.assertTrue((results["delta_rmse_hours"] < 0).all())

    def test_large_salinity_response_is_reviewed(self):
        records = pd.DataFrame({
            "node": ["bay", "bay", "canal", "canal"],
            "horizon_days": [7] * 4, "activation": ["linear"] * 4,
            "perturbed_concept": ["sal_mean"] * 4,
            "actual_delta": [1.] * 4,
            "predicted_difference_hours": [.2, -.1, 10., 11.],
        })
        audit = sensitivity_audit(records, threshold_hours=4)
        self.assertFalse(bool(audit.set_index("node").loc[
            "bay", "review_large_response"]))
        self.assertTrue(bool(audit.set_index("node").loc[
            "canal", "review_large_response"]))

    def test_empty_cross_model_overlap_not_misleading(self):
        fcm, base = synthetic_predictions()
        base = base.copy()
        base["issue_date"] = "2025-01-01"
        combined = load_predictions(fcm, base)
        res = compare_models(combined, n_boot=10)
        self.assertTrue(res.empty)


if __name__ == "__main__":
    unittest.main()
