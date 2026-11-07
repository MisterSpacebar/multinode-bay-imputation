"""Focused checks for presentation-only FCM edge filtering and inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from fcm_presentation import load_weights, select_edges


class PresentationTests(unittest.TestCase):
    def test_filter_preserves_direction_sign_and_exact_threshold(self):
        weights = pd.DataFrame(
            [[0.9, -0.2, 0.1999], [0.4, 0.8, 0], [0, 0.3, 0.7]],
            index=["a", "b", "c"], columns=["a", "b", "c"],
        )
        original = weights.copy()
        self.assertEqual(select_edges(weights, list(weights.index), 0.2),
                         [("a", "b", -0.2), ("b", "a", 0.4), ("c", "b", 0.3)])
        pd.testing.assert_frame_equal(weights, original)
        self.assertEqual(select_edges(weights, ["a", "b"], 0.2),
                         [("a", "b", -0.2), ("b", "a", 0.4)])

    def test_invalid_selections_fail_explicitly(self):
        weights = pd.DataFrame([[0]], index=["a"], columns=["a"])
        for threshold in [0, -1, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                select_edges(weights, ["a"], threshold)
        for concepts in [["missing"], ["a", "a"]]:
            with self.assertRaises(ValueError):
                select_edges(weights, concepts, 0.2)

    def test_saved_matrices_are_valid_and_filter_counts_match(self):
        root = Path(__file__).resolve().parent
        for model, size, threshold, negative, count, negatives in [
            ("physical", 11, 0.25, 0.10, 12, 4), ("nutrient", 16, 0.42, 0.22, 22, 5),
        ]:
            weights = load_weights(root / "analysis" / f"fcm_weights_{model}.csv")
            self.assertEqual(weights.shape, (size, size))
            self.assertTrue((weights.iloc[:, :3] == 0).all().all())
            edges = select_edges(weights, list(weights.index), threshold, negative)
            self.assertEqual(len(edges), count)
            self.assertEqual(sum(weight < 0 for *_, weight in edges), negatives)

    def test_negative_threshold_only_relaxes_negative_edges(self):
        weights = pd.DataFrame(
            [[0, -0.15, 0.3], [0.2, 0, -0.05], [0, 0, 0]],
            index=["a", "b", "c"], columns=["a", "b", "c"],
        )
        self.assertEqual(select_edges(weights, list(weights.index), 0.3, 0.1),
                         [("a", "b", -0.15), ("a", "c", 0.3)])
        for negative in [0, 0.5]:
            with self.assertRaises(ValueError):
                select_edges(weights, list(weights.index), 0.3, negative)

    def test_invalid_weight_files_fail_explicitly(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.csv"
            for value in [np.nan, np.inf, 1.1]:
                pd.DataFrame([[value]], index=["Water Temp"],
                             columns=["Water Temp"]).to_csv(path)
                with self.assertRaises(ValueError):
                    load_weights(path)
            pd.DataFrame([[0]], index=["Water Temp"],
                         columns=["Salinity"]).to_csv(path)
            with self.assertRaises(ValueError):
                load_weights(path)


if __name__ == "__main__":
    unittest.main()
