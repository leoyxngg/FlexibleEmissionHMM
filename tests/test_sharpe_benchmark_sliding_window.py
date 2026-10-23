"""Check sliding-window boundaries, causal isolation, and equal-window summaries."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/sp500"))
import sharpe_benchmark_sliding_window as sliding  # noqa: E402


class SlidingWindowTests(unittest.TestCase):
    """Verify actual-data windows without expensive HMM estimation."""

    @classmethod
    def setUpClass(cls):
        """Prepare actual prices and one deterministic Gaussian filter."""
        cls.prices = sliding.load_prices()
        cls.windows = sliding.available_windows(cls.prices)
        cls.bounds = cls.windows.iloc[0]
        cls.model = GaussianHMM(n_components=3, covariance_type="full")
        cls.model.n_features = 3
        cls.model.startprob_ = np.array([0.3, 0.4, 0.3])
        cls.model.transmat_ = np.full((3, 3), 0.025) + np.eye(3) * 0.925
        cls.model.means_ = np.array([
            [-0.002, 0.020, -0.001], [0, 0.010, 0], [0.002, 0.006, 0.001],
        ])
        cls.model.covars_ = np.array([
            np.diag([0.0004, 0.00001, 0.00001]),
            np.diag([0.0001, 0.00001, 0.00001]),
            np.diag([0.00005, 0.00001, 0.00001]),
        ])
        with patch.object(sliding.matrix, "fit_hmmlearn", return_value=cls.model):
            cls.result = sliding.run_window(cls.prices, cls.bounds)

    def test_actual_boundaries_and_warmup(self):
        """Retain all 37 full spans and the first training day's features."""
        self.assertEqual(len(self.windows), 37)
        self.assertEqual(list(self.windows.index[[0, -1]]), ["1963", "1999"])
        self.assertEqual(self.bounds["test_end"], "1989-12-31")
        _, emissions, features = sliding.matrix.build_features(self.prices)
        index = self.prices.loc["1963":"1982"].index
        self.assertTrue(index.isin(emissions.index).all())
        self.assertTrue(index.isin(features.index).all())
        lengths = {"training": 20, "validation": 3, "gap": 1, "test": 3}
        for split, length in lengths.items():
            self.assertEqual(self.result["splits"][split].index.year.nunique(), length)

    def test_incomplete_boundaries_and_missing_year(self):
        """Reject truncated end years, missing interior years, and cold training."""
        self.assertEqual(len(sliding.available_windows(self.prices.loc[:"2025-06"])), 36)
        cold = sliding.available_windows(self.prices.loc["1963":])
        self.assertEqual(cold.index[0], "1964")
        broken = self.prices[self.prices.index.year != 1980]
        self.assertEqual(sliding.available_windows(broken).index[0], "1981")

    def test_gap_and_test_cannot_change_fits_or_selection(self):
        """Perturb gap/test prices while retaining identical training and selection."""
        altered = self.prices.copy()
        tail = altered.loc[self.bounds["gap_start"]:]
        altered.loc[tail.index] *= np.exp(np.linspace(0.2, -0.2, len(tail)))
        with patch.object(sliding.matrix, "fit_hmmlearn", return_value=self.model) as fit:
            result = sliding.run_window(altered, self.bounds)
        _, emissions, _ = sliding.matrix.build_features(self.prices)
        np.testing.assert_allclose(fit.call_args.args[0], emissions.loc[
            self.bounds["training_start"]:self.bounds["training_end"]].to_numpy())
        pd.testing.assert_frame_equal(
            result["validation_table"], self.result["validation_table"],
        )
        self.assertEqual(result["rotation"], self.result["rotation"])
        self.assertEqual(result["best_single"], self.result["best_single"])
        self.assertEqual(sliding.matrix.TRAIN_START, "1988-01-01")
        self.assertEqual(sliding.matrix.TRAIN_END, "1997-12-31")

    def test_equal_window_numeric_aggregation(self):
        """Include every metric and finite counts without pooling daily returns."""
        second = dict(self.result)
        second["summary"] = self.result["summary"].copy()
        second["summary"].loc["buy_and_hold", "sharpe"] = np.inf
        original = self.result["summary"].loc["buy_and_hold", "total_return"]
        second["summary"].loc["buy_and_hold", "total_return"] = original + 2
        report = sliding.aggregate_results({"a": self.result, "b": second})
        self.assertEqual(set(report["summary"].columns.get_level_values(0)),
                         set(self.result["summary"].columns))
        self.assertEqual(report["summary"].loc["buy_and_hold", ("sharpe", "count")], 1)
        self.assertTrue(report["summary"]["years"]["count"].eq(2).all())
        self.assertTrue(report["selection_counts"].sum().eq(2).all())
        self.assertTrue(report["selection_frequencies"].sum().eq(1).all())
        self.assertEqual(report["test_table"].shape, (10, 12))
        stats = report["summary"].loc["buy_and_hold", "total_return"]
        self.assertAlmostEqual(stats["mean"], original + 1)
        self.assertAlmostEqual(stats["std"], np.sqrt(2))

    def test_same_day_test_close_is_not_a_signal(self):
        """Keep first test exposure unchanged when its own close changes."""
        altered = self.prices.copy()
        first = self.result["test_weights"].index[0]
        altered.loc[first] *= 1.25
        with patch.object(sliding.matrix, "fit_hmmlearn", return_value=self.model):
            result = sliding.run_window(altered, self.bounds)
        pd.testing.assert_series_equal(result["test_weights"].iloc[0],
                                       self.result["test_weights"].iloc[0])


if __name__ == "__main__":
    unittest.main()
