"""Benchmark HMM regime switches against market strategies.

Causality protocol: a weight for day t may use data through t-1 only. Strategy
builders return fully causal weights (they shift internally); regime signals
must come from the one-step-ahead filter in ``signals`` (never smoothed
``predict_proba``). PnL is ``apply_weights(returns, weights)``.
"""

from .evaluation import (
    compare_strategies,
    drawdown,
    performance_summary,
    plot_equity_curves,
    plot_regime_overlay,
)
from .models import (
    cash_proxy,
    credit_commodities_proxy,
    equal_weight_proxy,
    long_treasuries_proxy,
    market_proxy,
)
from .signals import (
    forward_predictive_probabilities,
    high_volatility_state,
    hmm_causal_probabilities,
    hmmlearn_causal_probabilities,
)
from .strategies import (
    apply_weights,
    build_rotation_map,
    buy_and_hold,
    constant_exposure,
    moving_average_timing,
    regime_model_rotation,
    regime_probability_scaled,
    regime_sharpe_table,
    regime_threshold,
    volatility_target,
)

__all__ = [
    "apply_weights",
    "build_rotation_map",
    "buy_and_hold",
    "cash_proxy",
    "compare_strategies",
    "constant_exposure",
    "credit_commodities_proxy",
    "drawdown",
    "equal_weight_proxy",
    "forward_predictive_probabilities",
    "high_volatility_state",
    "hmm_causal_probabilities",
    "hmmlearn_causal_probabilities",
    "long_treasuries_proxy",
    "market_proxy",
    "moving_average_timing",
    "performance_summary",
    "plot_equity_curves",
    "plot_regime_overlay",
    "regime_model_rotation",
    "regime_probability_scaled",
    "regime_sharpe_table",
    "regime_threshold",
    "volatility_target",
]
