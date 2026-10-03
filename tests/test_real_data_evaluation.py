import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from flexible_emission_hmm import ExactInference, StateParameters
from model_evaluation.real_data import (
    backtest, ewma_lagged_signal_variance, lagged_state_forecasts,
    mixture_moments, portfolio_summary,
)


class FilteringTests(unittest.TestCase):
    def setUp(self):
        self.states = StateParameters(
            np.array([0.6, 0.4]), np.array([[0.9, 0.1], [0.2, 0.8]])
        )
        self.log_emissions = np.log(np.array([
            [0.4, 0.2], [0.1, 0.7], [0.8, 0.3], [0.3, 0.5]
        ]))
        self.inference = ExactInference()

    def test_filter_matches_prefix_posterior_and_likelihood(self):
        predicted, filtered, scores = self.inference.filter(self.states, self.log_emissions)
        for length in range(1, len(scores) + 1):
            prefix = self.inference.infer(self.states, self.log_emissions[:length])
            assert_allclose(filtered[length - 1], prefix.responsibilities[-1])
            assert_allclose(scores[:length].sum(), prefix.log_likelihood)
        assert_allclose(predicted.sum(axis=1), 1)
        assert_allclose(filtered.sum(axis=1), 1)

    def test_future_changes_do_not_rewrite_forecasts(self):
        original = self.inference.filter(self.states, self.log_emissions)
        changed = self.log_emissions.copy()
        changed[-1] = np.log([0.99, 0.01])
        revised = self.inference.filter(self.states, changed)
        for before, after in zip(original, revised):
            assert_allclose(before[:-1], after[:-1])

    def test_training_boundary_preserves_state(self):
        _, train_filtered, _ = self.inference.filter(self.states, self.log_emissions[:2])
        prior = train_filtered[-1] @ self.states.transitions
        predicted, filtered, scores = self.inference.filter(
            self.states, self.log_emissions[2:], prior
        )
        full = self.inference.filter(self.states, self.log_emissions)
        assert_allclose(predicted, full[0][2:])
        assert_allclose(filtered, full[1][2:])
        assert_allclose(scores, full[2][2:])


class BacktestTests(unittest.TestCase):
    def test_two_close_signal_uses_training_history_at_boundary(self):
        train = np.array([[1.0, 0.0], [0.0, 1.0]])
        evaluate = np.array([[0.25, 0.75], [0.8, 0.2], [0.1, 0.9]])
        transitions = np.array([[0.9, 0.1], [0.2, 0.8]])
        signals = lagged_state_forecasts(train, evaluate, transitions)
        assert_allclose(signals, np.vstack([train, evaluate[:1]]) @ transitions @ transitions)

    def test_mixture_variance_includes_between_state_means(self):
        mean, variance = mixture_moments(
            np.array([[0.5, 0.5]]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])
        )
        assert_allclose(mean, [0])
        assert_allclose(variance, [2])

    def test_ewma_signal_uses_two_closes_of_lag(self):
        train = np.array([0.01, -0.02, 0.03])
        evaluate = np.array([0.04, 0.05])
        signals = ewma_lagged_signal_variance(train, evaluate)
        initial = np.var(train)
        first = 0.94 * initial + 0.06 * train[0] ** 2
        second = 0.94 * first + 0.06 * train[1] ** 2
        assert_allclose(signals, [second, 0.94 * second + 0.06 * train[2] ** 2])

    def test_turnover_uses_drifted_weight_and_charges_cost(self):
        frame = pd.DataFrame({"target_weight": [0.5, 0.5],
                              "simple_return": [0.1, -0.1]})
        result = backtest(frame, cash_return=0, cost_bps=100)
        first_cost = 0.01 * 0.5 / 1.005
        drifted_weight = 0.55 / 1.05
        second_cost = 0.01 * (drifted_weight - 0.5) / 0.995
        assert_allclose(result.turnover, [0.5 * (1 - first_cost),
                                          abs(0.5 * (1 - second_cost) - drifted_weight)])
        assert_allclose(result.gross_return, [0.05, -0.05])
        assert_allclose(result.cost, result.turnover * 0.01)
        assert_allclose(result.net_return, (1 - result.cost) * (1 + result.gross_return) - 1)

    def test_cash_accrues_on_uninvested_allocation(self):
        frame = pd.DataFrame({"target_weight": [0.25], "simple_return": [0.08]})
        result = backtest(frame, cash_return=0.01, cost_bps=0)
        assert_allclose(result.gross_return, [0.25 * 0.08 + 0.75 * 0.01])
        assert_allclose(result.net_return, result.gross_return)

    def test_summary_uses_stitched_returns(self):
        returns = np.array([0.01, -0.02, 0.03, 0.00])
        summary = portfolio_summary(returns, np.zeros(4))
        expected = np.sqrt(252) * returns.mean() / returns.std(ddof=1)
        assert_allclose(summary["sharpe"], expected)
        self.assertEqual(summary["observations"], 4)


if __name__ == "__main__":
    unittest.main()
