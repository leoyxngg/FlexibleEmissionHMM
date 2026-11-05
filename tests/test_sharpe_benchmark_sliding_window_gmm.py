"""Check GMM-HMM fitting, regime naming, and isolation from the Gaussian run."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/sp500"))
import sharpe_benchmark_sliding_window as sliding  # noqa: E402
import sharpe_benchmark_sliding_window_gmm as gmm  # noqa: E402


def fixed_gmm():
    """Build a deterministic three-state, two-component GMM-HMM."""
    model = GMMHMM(n_components=3, n_mix=2, covariance_type="full")
    model.n_features = 3
    model.startprob_ = np.array([0.3, 0.4, 0.3])
    model.transmat_ = np.full((3, 3), 0.025) + np.eye(3) * 0.925
    model.weights_ = np.array([[0.7, 0.3], [0.5, 0.5], [0.9, 0.1]])
    returns = np.array([[[-0.002, 0.02, 0], [-0.01, 0.03, 0]],
                        [[0.000, 0.01, 0], [0.001, 0.011, 0]],
                        [[0.002, 0.006, 0], [0.003, 0.007, 0]]])
    model.means_ = returns
    sd = np.array([[4e-4, 1e-3], [1e-4, 1e-4], [5e-5, 8e-5]])
    covars = np.zeros((3, 2, 3, 3))
    for state in range(3):
        for mix in range(2):
            covars[state, mix] = np.diag([sd[state, mix], 1e-5, 1e-5])
    model.covars_ = covars
    return model


class GmmSlidingWindowTests(unittest.TestCase):
    """Verify the GMM variant changes only the emission model."""

    def test_state_moments_use_mixture_variance(self):
        """Include between-component spread, not only within-component variance."""
        moments = gmm.state_moments(fixed_gmm())
        weights = np.array([0.7, 0.3])
        means = np.array([-0.002, -0.01])
        mean = (weights * means).sum()
        variance = (weights * (np.array([4e-4, 1e-3]) + means**2)).sum() - mean**2
        self.assertAlmostEqual(moments.loc[0, "mean_r_%day"], 100 * mean)
        self.assertAlmostEqual(moments.loc[0, "sd_r_%day"], 100 * np.sqrt(variance))
        self.assertGreater(moments.loc[0, "sd_r_%day"],
                           100 * np.sqrt((weights * np.array([4e-4, 1e-3])).sum()))

    def test_names_are_distinct_and_follow_gaussian_rule(self):
        """Name the widest state bear and the best remaining mean bull."""
        names, table = gmm.name_gmm_regimes(fixed_gmm())
        self.assertEqual(sorted(names.values()), ["bear", "bull", "sideways"])
        self.assertEqual(names[int(table["sd_r_%day"].idxmax())], "bear")
        with self.assertRaises(ValueError):
            two = GMMHMM(n_components=2, n_mix=2)
            two.weights_ = np.full((2, 2), 0.5)
            two.means_ = np.zeros((2, 2, 3))
            two.covars_ = np.ones((2, 2, 3, 3))
            gmm.name_gmm_regimes(two)

    def test_fit_matches_gaussian_benchmark_settings(self):
        """Reuse restarts, iteration cap, tolerance, floor, and full covariance."""
        features = np.random.default_rng(0).normal(size=(300, 3)) * 0.01
        calls = []
        real = GMMHMM

        def spy(**kwargs):
            calls.append(kwargs)
            return real(**kwargs)

        with patch.object(gmm, "GMMHMM", side_effect=spy):
            model = gmm.fit_gmm_hmm(features)
        self.assertEqual(len(calls), gmm.matrix.N_RESTARTS)
        first = calls[0]
        self.assertEqual(first["n_components"], gmm.matrix.N_STATES)
        self.assertEqual(first["n_mix"], gmm.N_MIX)
        self.assertEqual(first["covariance_type"], "full")
        self.assertEqual(first["min_covar"], gmm.matrix.MIN_COVAR)
        self.assertEqual(first["n_iter"], gmm.matrix.MAX_ITER)
        self.assertEqual(first["tol"], gmm.matrix.TOL)
        self.assertEqual(model.weights_.shape, (3, gmm.N_MIX))

    def test_window_runs_with_fixed_model_and_isolates_test(self):
        """Run one actual window and keep fits and selection free of test data."""
        prices = sliding.load_prices()
        bounds = sliding.available_windows(prices).iloc[0]
        model = fixed_gmm()
        result = sliding.run_window(
            prices, bounds, lambda features: model, gmm.name_gmm_regimes,
        )
        self.assertEqual(len(result["summary"]), 10)
        altered = prices.copy()
        tail = altered.loc[bounds["gap_start"]:]
        altered.loc[tail.index] *= np.exp(np.linspace(0.2, -0.2, len(tail)))
        again = sliding.run_window(
            altered, bounds, lambda features: model, gmm.name_gmm_regimes,
        )
        pd.testing.assert_frame_equal(
            result["validation_table"], again["validation_table"],
        )
        self.assertEqual(result["rotation"], again["rotation"])

    def test_default_run_window_is_still_gaussian(self):
        """Keep the original benchmark path unchanged when no hooks are passed."""
        prices = sliding.load_prices()
        bounds = sliding.available_windows(prices).iloc[0]
        with patch.object(sliding.matrix, "fit_hmmlearn") as fit:
            fit.side_effect = RuntimeError("original path reached")
            with self.assertRaisesRegex(RuntimeError, "original path reached"):
                sliding.run_window(prices, bounds)


if __name__ == "__main__":
    unittest.main()
