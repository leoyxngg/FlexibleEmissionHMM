"""Behavioral leakage checks for the S&P 500 regime benchmark."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "notebooks/sp500/sp500_regime_algorithm_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("regime_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class RegimeBenchmarkLeakageTests(unittest.TestCase):
    """Verify observable time boundaries using actual prices and frozen models."""

    @classmethod
    def setUpClass(cls):
        """Load local prices and fit supervised models once for the test class."""
        data = pd.read_csv(benchmark.DATA_PATH, parse_dates=["YYYYMMDD"])
        cls.prices = data.set_index("YYYYMMDD")["DlyPrcInd"].sort_index()
        cls.returns, cls.emissions, cls.features = benchmark.build_features(cls.prices)
        cls.index = cls.returns.loc[benchmark.EVAL_START:benchmark.EVAL_END].index
        cls.ml = benchmark.fit_ml(cls.features, cls.returns)
        # Fixed, valid parameters isolate inference timing from EM randomness.
        cls.hmm = GaussianHMM(n_components=3, covariance_type="full")
        cls.hmm.n_features = 3
        cls.hmm.startprob_ = np.array([0.3, 0.4, 0.3])
        cls.hmm.transmat_ = np.full((3, 3), 0.025) + np.eye(3) * 0.925
        cls.hmm.means_ = np.array([
            [-0.002, 0.020, -0.001],
            [0.000, 0.010, 0.000],
            [0.002, 0.006, 0.001],
        ])
        cls.hmm.covars_ = np.array([
            np.diag([0.0004, 0.00001, 0.00001]),
            np.diag([0.0001, 0.00001, 0.00001]),
            np.diag([0.00005, 0.00001, 0.00001]),
        ])
        cls.names, _ = benchmark.name_regimes(cls.hmm)
        cls.weights = benchmark.strategy_weights(cls.prices, cls.index, cls.ml)
        cls.labels, cls.probabilities = benchmark.causal_regimes(
            cls.hmm, cls.emissions, cls.index, cls.names,
        )

    def test_features_are_lagged_before_splitting(self):
        """Check every ML row, including both split boundaries, against t-1."""
        expected = self.emissions.shift(1).dropna()
        pd.testing.assert_frame_equal(self.features, expected)
        for day in [self.index[0], self.returns.loc["1988"].index[0]]:
            previous_day = self.emissions.index[
                self.emissions.index.get_loc(day) - 1
            ]
            np.testing.assert_allclose(
                self.features.loc[day], self.emissions.loc[previous_day],
            )

    def test_current_and_future_prices_cannot_change_signals(self):
        """Perturb raw prices at t onward and preserve all decisions through t."""
        for offset in [0, 317, len(self.index) - 1]:
            day = self.index[offset]
            changed = self.prices.copy()
            tail = changed.loc[day:]
            changed.loc[day:] *= np.exp(np.linspace(0.2, -0.2, len(tail)))
            _, emissions, _ = benchmark.build_features(changed)
            weights = benchmark.strategy_weights(changed, self.index, self.ml)
            labels, probabilities = benchmark.causal_regimes(
                self.hmm, emissions, self.index, self.names,
            )
            pd.testing.assert_frame_equal(
                weights.loc[:day], self.weights.loc[:day],
            )
            pd.testing.assert_frame_equal(
                probabilities.loc[:day], self.probabilities.loc[:day],
            )
            pd.testing.assert_series_equal(labels.loc[:day], self.labels.loc[:day])
            if offset < len(self.index) - 1:
                # The perturbation is meaningful: later ML inputs do change.
                self.assertFalse(weights.iloc[offset + 1:].equals(
                    self.weights.iloc[offset + 1:],
                ))
                self.assertFalse(probabilities.iloc[offset + 1:].equals(
                    self.probabilities.iloc[offset + 1:],
                ))

    def test_hmm_boundary_carries_training_posterior(self):
        """Carry the final training posterior forward instead of resetting."""
        training = self.emissions.loc[
            benchmark.TRAIN_START:benchmark.TRAIN_END
        ].to_numpy()
        # At the last row only, smoothing equals filtering: no future exists.
        posterior = self.hmm.predict_proba(training)[-1]
        expected = posterior @ self.hmm.transmat_
        np.testing.assert_allclose(self.probabilities.iloc[0], expected)
        self.assertFalse(np.allclose(expected, self.hmm.startprob_))

    def test_prefix_matches_full_history(self):
        """Appending future observations must preserve all earlier decisions."""
        prefix_index = self.index[:400]
        prefix_prices = self.prices.loc[:prefix_index[-1]]
        _, emissions, _ = benchmark.build_features(prefix_prices)
        weights = benchmark.strategy_weights(prefix_prices, prefix_index, self.ml)
        labels, probabilities = benchmark.causal_regimes(
            self.hmm, emissions, prefix_index, self.names,
        )
        pd.testing.assert_frame_equal(weights, self.weights.loc[prefix_index])
        pd.testing.assert_frame_equal(
            probabilities, self.probabilities.loc[prefix_index],
        )
        pd.testing.assert_series_equal(labels, self.labels.loc[prefix_index])

    def test_evaluation_changes_cannot_change_training(self):
        """Refit after replacing evaluation prices and compare frozen models."""
        changed = self.prices.copy()
        changed.loc[benchmark.EVAL_START:] *= 1.7
        returns, emissions, features = benchmark.build_features(changed)
        pd.testing.assert_frame_equal(
            emissions.loc[benchmark.TRAIN_START:benchmark.TRAIN_END],
            self.emissions.loc[benchmark.TRAIN_START:benchmark.TRAIN_END],
        )
        fitted = benchmark.fit_ml(features, returns)
        for key in ["mu", "sd", "logistic", "ridge", "return_sd"]:
            np.testing.assert_allclose(fitted[key], self.ml[key], rtol=0, atol=0)
        # Compare predictions on the same unmodified evaluation design matrix.
        pd.testing.assert_frame_equal(
            benchmark.strategy_weights(self.prices, self.index, fitted), self.weights,
        )
        training = self.features.loc[benchmark.TRAIN_START:benchmark.TRAIN_END]
        pd.testing.assert_series_equal(self.ml["mu"], training.mean())
        self.assertLess(training.index[-1], self.index[0])

    def test_boundary_predictions_are_not_shifted_twice(self):
        """Compare first-day positions with direct predictions from t-1 inputs."""
        day = self.index[0]
        scaled = (self.features.loc[day] - self.ml["mu"]) / self.ml["sd"]
        design = np.r_[1.0, scaled]
        expected_logistic = 2 * benchmark.expit(design @ self.ml["logistic"]) - 1
        self.assertAlmostEqual(self.weights.loc[day, "logistic_up"], expected_logistic)
        expected_ridge = np.clip(
            design @ self.ml["ridge"] / self.ml["return_sd"], -1, 1.5,
        )
        self.assertAlmostEqual(self.weights.loc[day, "ridge_return"], expected_ridge)
        probability = self.ml["forest"].predict_proba(
            scaled.to_numpy().reshape(1, -1),
        )[0, 1]
        self.assertAlmostEqual(
            self.weights.loc[day, "random_forest"], 2 * probability - 1,
        )
        self.assertGreater(self.weights.loc[day, "vol_target_60"], 0)
        self.assertEqual(
            self.weights.loc[day, "ma_50_200"],
            benchmark.moving_average_timing(self.prices).loc[day],
        )

    def test_regime_naming_is_unique_when_extremes_overlap(self):
        """Map every state even when highest return and volatility coincide."""
        model = GaussianHMM(n_components=3, covariance_type="full")
        model.n_features = 3
        model.means_ = self.hmm.means_.copy()
        model.means_[0, 0] = 0.01
        model.covars_ = self.hmm.covars_.copy()
        names, _ = benchmark.name_regimes(model)
        self.assertEqual(set(names), {0, 1, 2})
        self.assertEqual(set(names.values()), set(benchmark.REGIMES))

    def test_costs_are_applied_before_regime_grouping(self):
        """Keep actual daily turnover when neighboring dates have other labels."""
        index = self.index[:60]
        weights = pd.DataFrame({"test": np.arange(60) % 2}, index=index)
        labels = pd.Series(np.resize(benchmark.REGIMES, 60), index=index)
        returns = self.returns.loc[index]
        table, counts, pnl = benchmark.sharpe_matrix(returns, weights, labels)
        expected = benchmark.apply_weights(returns, weights["test"], cost_bps=5)
        np.testing.assert_allclose(pnl["test"], expected)
        self.assertEqual(list(table.columns), benchmark.REGIMES + ["test_all"])
        self.assertEqual(counts["test_all"], 60)
        self.assertAlmostEqual(table.loc["test", "test_all"], benchmark.sharpe(expected))
        for regime in benchmark.REGIMES:
            self.assertEqual(counts[regime], 20)
            self.assertAlmostEqual(
                table.loc["test", regime],
                benchmark.sharpe(expected.loc[labels == regime]),
            )
        self.assertTrue(np.isnan(benchmark.sharpe(pd.Series(np.zeros(25)))))


if __name__ == "__main__":
    unittest.main()
