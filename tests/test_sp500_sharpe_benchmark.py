"""Split isolation and portfolio accounting checks for regime co-rotation."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

SCRIPT = Path(__file__).resolve().parents[1] / "notebooks/sp500/sp500_sharpe_benchmark.py"
SPEC = importlib.util.spec_from_file_location("sharpe_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class CoRotationTests(unittest.TestCase):
    """Check development selection, temporal isolation, and composite turnover."""

    @classmethod
    def setUpClass(cls):
        """Run actual-data predictions with fixed HMM parameters for fast checks."""
        data = pd.read_csv(benchmark.DATA_PATH, parse_dates=["YYYYMMDD"])
        cls.prices = data.set_index("YYYYMMDD")["DlyPrcInd"].sort_index()
        cls.model = GaussianHMM(n_components=3, covariance_type="full")
        cls.model.n_features = 3
        cls.model.startprob_ = np.array([0.3, 0.4, 0.3])
        cls.model.transmat_ = np.full((3, 3), 0.025) + np.eye(3) * 0.925
        cls.model.means_ = np.array([
            [-0.002, 0.020, -0.001],
            [0.000, 0.010, 0.000],
            [0.002, 0.006, 0.001],
        ])
        cls.model.covars_ = np.array([
            np.diag([0.0004, 0.00001, 0.00001]),
            np.diag([0.0001, 0.00001, 0.00001]),
            np.diag([0.00005, 0.00001, 0.00001]),
        ])
        with patch.object(benchmark, "fit_hmmlearn", return_value=cls.model):
            cls.result = benchmark.run_benchmark(cls.prices)

    def test_disjoint_complete_splits_and_test_only_outputs(self):
        """Keep all reported test portfolios on exactly the held-out dates."""
        splits = self.result["splits"]
        self.assertEqual(splits["training"].index.year.unique().tolist(),
                         list(range(1988, 1998)))
        self.assertEqual(splits["development"].index.year.unique().tolist(),
                         list(range(1998, 2002)))
        self.assertEqual(splits["test"].index.year.unique().tolist(),
                         list(range(2002, 2005)))
        self.assertLess(splits["training"].index[-1], splits["development"].index[0])
        self.assertLess(splits["development"].index[-1], splits["test"].index[0])
        for key in ["test_weights", "test_pnl", "test_labels"]:
            self.assertTrue(self.result[key].index.equals(splits["test"].index))
        self.assertEqual(len(self.result["test_weights"].columns), 10)
        self.assertEqual(list(self.result["test_table"]),
                         benchmark.REGIMES + ["test_all"])
        self.assertEqual(list(self.result["development_table"]),
                         benchmark.REGIMES + ["dev_all"])

    def test_selection_fallback_ties_and_best_single(self):
        """Select only finite development scores with deterministic fallbacks."""
        table = pd.DataFrame({
            "bear": [0.1, 0.8, 0.8],
            "sideways": [0.0, 1.0, 2.0],
            "bull": [np.nan, np.inf, np.nan],
            "dev_all": [0.1, 0.2, 0.3],
        }, index=["buy_and_hold", "ridge_return", "ma_50_200"])
        counts = pd.Series({"bear": 100, "sideways": 19, "bull": 100})
        rotation, best = benchmark.select_rotation(table, counts)
        self.assertEqual(rotation, {
            "bear": "ridge_return", "sideways": "buy_and_hold",
            "bull": "buy_and_hold",
        })
        self.assertEqual(best, "ma_50_200")

    def test_test_prices_cannot_change_fit_or_development_selection(self):
        """Perturb every test price without changing fitted inputs or selection."""
        changed = self.prices.copy()
        tail = changed.loc[benchmark.TEST_START:]
        changed.loc[benchmark.TEST_START:] *= np.exp(np.linspace(0.3, -0.3, len(tail)))
        with patch.object(benchmark, "fit_hmmlearn", return_value=self.model) as fit:
            altered = benchmark.run_benchmark(changed)
        _, emissions, _ = benchmark.build_features(self.prices)
        np.testing.assert_allclose(
            fit.call_args.args[0],
            emissions.loc[benchmark.TRAIN_START:benchmark.TRAIN_END].to_numpy(),
        )
        pd.testing.assert_frame_equal(
            altered["development_table"], self.result["development_table"],
        )
        self.assertEqual(altered["rotation"], self.result["rotation"])
        self.assertEqual(altered["best_single"], self.result["best_single"])
        # The first test day's decisions must not see even its own close.
        pd.testing.assert_series_equal(
            altered["test_weights"].iloc[0], self.result["test_weights"].iloc[0],
        )

    def test_composite_costs_charge_actual_exposure_changes_once(self):
        """Charge entry and long-to-short flips, but not same-exposure switches."""
        index = pd.bdate_range("2002-01-01", periods=4)
        weights = pd.DataFrame({"a": [1.0] * 4, "b": [-1.0] * 4,
                                "c": [-1.0] * 4}, index=index)
        labels = pd.Series(["bear", "sideways", "bull", "bear"], index=index)
        rotation = {"bear": "a", "sideways": "b", "bull": "c"}
        composite = benchmark.co_rotation_weights(weights, labels, rotation)
        np.testing.assert_allclose(composite, [1, -1, -1, 1])
        _, _, pnl = benchmark.sharpe_matrix(
            pd.Series(0.0, index=index), composite.to_frame(), labels,
        )
        np.testing.assert_allclose(
            pnl["co_rotation"], -benchmark.COST_BPS / 1e4 * np.array([1, 2, 0, 2]),
        )

    def test_best_single_and_rotation_follow_frozen_mapping(self):
        """Match the selected candidate weights and reproduce the single baseline."""
        weights = self.result["test_weights"]
        best = self.result["best_single"]
        np.testing.assert_allclose(weights["dev_best_single"], weights[best])
        np.testing.assert_allclose(self.result["test_pnl"]["dev_best_single"],
                                   self.result["test_pnl"][best])
        for regime, strategy in self.result["rotation"].items():
            dates = self.result["test_labels"] == regime
            np.testing.assert_allclose(weights.loc[dates, "co_rotation"],
                                       weights.loc[dates, strategy])

    def test_reject_unknown_regime_and_misaligned_positions(self):
        """Reject missing mappings and date mismatches rather than trade silently."""
        weights = self.result["test_weights"]
        labels = self.result["test_labels"]
        with self.assertRaises(ValueError):
            benchmark.co_rotation_weights(weights, labels, {})
        with self.assertRaises(ValueError):
            benchmark.co_rotation_weights(weights.iloc[1:], labels,
                                          self.result["rotation"])

    def test_missing_test_year_is_rejected(self):
        """Require all requested test years from the existing dataset."""
        returns, _, _ = benchmark.build_features(self.prices.loc[:"2003"])
        with self.assertRaises(ValueError):
            benchmark.split_returns(returns)

    def test_max_drawdown_counts_loss_from_initial_dollar(self):
        """A first-day loss from $1 is a drawdown, not a zero-peak baseline."""
        value = benchmark.max_drawdown(pd.Series([-0.10, 0.01]))
        self.assertAlmostEqual(value, np.exp(-0.10) - 1.0)
        self.assertAlmostEqual(
            benchmark.max_drawdown(pd.Series([0.01, -0.02, 0.05])),
            np.exp(-0.02) - 1.0,
        )

    def test_plots_render_without_interactive_windows(self):
        """Render both benchmark figures with the noninteractive backend."""
        benchmark.plot_benchmark(self.result)
        for number in plt.get_fignums():
            plt.figure(number).canvas.draw()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
