"""Numerical and temporal checks for the paper-inspired regime comparison."""

import unittest

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t

from notebooks.sp500.paper_regime_experiment import (
    MultivariateStudentTEmission,
    evaluate_signals,
    fit_day,
)


class RegimeExperimentTests(unittest.TestCase):
    """Verify emissions and next-day 0/1 strategy accounting."""

    def test_multivariate_t_density_matches_scipy(self):
        X = np.array([[0.4, -0.7], [-1.1, 1.8], [3.1, -0.4]])
        emission = MultivariateStudentTEmission()
        emission.initialize(X)
        emission.means[:] = [0.25, -0.5]
        emission.covariances[:] = [[1.2, 0.25], [0.25, 2.1]]
        emission.dof[:] = 7.0
        expected = multivariate_t.logpdf(X, loc=emission.means[0],
                                         shape=emission.covariances[0], df=7)
        np.testing.assert_allclose(emission.log_prob(X)[:, 0], expected, atol=1e-12)

    def test_wealth_uses_return_after_decision(self):
        dates = pd.date_range("2020-01-01", periods=5)
        prices = pd.Series([100, 110, 99, 108.9, 98.01], index=dates)
        rows = [{"date": date, "model": "Gaussian", "pca": False,
                 "bull_raw": raw} for date, raw in zip(dates[:4], [1, 1, 0, 0])]
        fits = pd.DataFrame(rows)
        pnl, weights = evaluate_signals(fits, prices, median_window=1)
        np.testing.assert_allclose(np.expm1(pnl["Gaussian / no PCA"]),
                                   [0.1, -0.1, 0, 0])
        np.testing.assert_allclose(np.expm1(pnl["Buy and hold"]),
                                   [0.1, -0.1, 0.1, -0.1])
        self.assertEqual(weights["Gaussian / no PCA"].tolist(), [1, 1, 0, 0])

    def test_even_median_tie_preserves_previous_position(self):
        dates = pd.date_range("2020-01-01", periods=4)
        prices = pd.Series([100, 101, 102, 103], index=dates)
        rows = [{"date": date, "model": "Gaussian", "pca": False,
                 "bull_raw": raw} for date, raw in zip(dates[:3], [1, 0, 0])]
        _, weights = evaluate_signals(pd.DataFrame(rows), prices, median_window=2)
        self.assertEqual(weights["Gaussian / no PCA"].tolist(), [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
