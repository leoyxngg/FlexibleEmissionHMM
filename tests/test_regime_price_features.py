"""Causality and shape checks for price-derived regime features."""

import unittest

import numpy as np
import pandas as pd

from flexible_emission_hmm.feature import regime_price_features


class RegimePriceFeaturesTests(unittest.TestCase):
    """Check warm-up, formulas, and no use of future prices."""

    def setUp(self):
        self.prices = pd.Series(100 * np.cumprod(1 + 0.01 * np.sin(np.arange(100))),
                                index=pd.date_range("2000-01-01", periods=100))

    def test_prefix_is_unchanged_by_future_prices(self):
        full = regime_price_features(self.prices, windows=(5, 10), half_lives=(5,))
        prefix = regime_price_features(self.prices.iloc[:60], windows=(5, 10),
                                       half_lives=(5,))
        pd.testing.assert_frame_equal(full.iloc[:60], prefix)

    def test_shape_warmup_and_return(self):
        features = regime_price_features(self.prices, windows=(5, 10),
                                         half_lives=(5,))
        self.assertEqual(features.shape, (100, 10))
        self.assertTrue(features.iloc[0].isna().all())
        self.assertTrue(np.isfinite(features.iloc[-1]).all())
        self.assertAlmostEqual(features["return"].iloc[10],
                               self.prices.iloc[10] / self.prices.iloc[9] - 1)

    def test_downside_uses_zero_contributions(self):
        prices = pd.Series([100, 90, 99, 89.1, 98.01, 88.209])
        features = regime_price_features(prices, windows=(2,), half_lives=(2,))
        self.assertAlmostEqual(features["downside_2"].iloc[2], 0.05)
        self.assertAlmostEqual(features["sortino_2"].iloc[2], 0.0)

    def test_reject_invalid_horizons(self):
        with self.assertRaises(ValueError):
            regime_price_features(self.prices, windows=(1,))
        with self.assertRaises(ValueError):
            regime_price_features(self.prices, half_lives=(5, 5))


if __name__ == "__main__":
    unittest.main()
