# Market Strategy Benchmarks for HMM Regimes

Benchmark what a fitted regime model is *worth*: can regimes beat strategies
that use no regime model at all? Every strategy here is causal - a weight for
day `t` is decided with data through `t-1`.

## The causality protocol (the whole game)

- `strategies.*` return weights where `w[t]` is the position **held on day t**,
  decided by day `t-1`. They shift internally; do not shift again.
- Regime inputs must be **one-step-ahead** probabilities `P(S_t | data < t)`:
  - local package: `signals.hmm_causal_probabilities(model, X)` (wraps
    `HMM.filter`, whose first output is causal by construction)
  - hmmlearn: `signals.hmmlearn_causal_probabilities(model, X)`
- Never feed `predict_proba` (smoothed, forward-backward) into a strategy: it
  peeks at future returns and inflates every metric.

## Standard benchmark table

Baselines (zero model signals) vs HMM-signal strategies:

| family | builder | what it tests |
|---|---|---|
| baseline | `buy_and_hold` | do regimes beat doing nothing |
| baseline | `volatility_target` | divide-by-recent-vol; the strongest no-regime baseline |
| baseline | `moving_average_timing` | trend following, the classic regime-free timing rule |
| HMM | `regime_threshold` | binary exposure cut on causal P(high vol) > trigger |
| HMM | `regime_probability_scaled` | exposure blended continuously in the probability |
| HMM | `regime_model_rotation` | Wang et al. (2020): hold the model best suited to the current regime |

Rotation universe (`models.py`), each docstring records the regime it is
expected to win: `market_proxy` (bull), `equal_weight_proxy` (sideways /
post-crash recovery), `long_treasuries_proxy` (bear, flight to quality),
`credit_commodities_proxy` (late-bull reflation), `cash_proxy` (fallback,
never wins). Rotation map from training data:
`regime_sharpe_table` (per-regime Sharpe, the Wang Tables 5-7 pattern) ->
`build_rotation_map` (argmax guarded by a Sharpe margin and a minimum day
count, fallback = cash) -> `regime_model_rotation(causal_labels, ...)` with
costs charged per switch.

Metrics (`compare_strategies`): annualized return/vol, Sharpe, Sortino, max
drawdown, Calmar, hit rate, exposure, turnover/year. The usual verdict for
volatility regimes: regimes win on **drawdown and Sharpe**, not raw return.
Pass `cost_bps` to `apply_weights` to charge turnover.

## Recipe

```python
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from flexible_emission_hmm import HMM, GaussianEmission, BaumWelchTrainer
from flexible_emission_hmm.feature import compute_features
from model_evaluation.market_strategy import (
    apply_weights, buy_and_hold, volatility_target, moving_average_timing,
    regime_threshold, regime_probability_scaled, compare_strategies,
    high_volatility_state, hmm_causal_probabilities,
    plot_equity_curves, plot_regime_overlay,
)

# data / features
data = pd.read_csv("data/processed/sp500_index_price.csv",
                   parse_dates=["YYYYMMDD"]).sort_values("YYYYMMDD")
prices = data.set_index("YYYYMMDD")["DlyPrcInd"]
returns = np.log(prices).diff().dropna()
transformed = compute_features(prices, volatility_days=60, momentum_days=20)

# fit on the training window only
train = returns.loc["1988":"1997"]
X_train = transformed.loc[train.index].to_numpy()
model = HMM(GaussianEmission(2, min_variance=1e-5, random_state=0),
            trainer=BaumWelchTrainer(max_iter=120, tol=1e-3))
model.fit(X_train)
high = high_volatility_state(model)                      # rank states by fitted sd(r)

# causal signal over the evaluation window (fit params frozen)
evaluation = returns.loc["1998":"2001"]
X_eval = transformed.loc[evaluation.index].to_numpy()
p_high = pd.Series(hmm_causal_probabilities(model, X_eval)[:, high],
                   index=evaluation.index)

# strategies: baselines vs HMM signals
weights = {
    "buy_and_hold": buy_and_hold(evaluation),
    "vol_target": volatility_target(evaluation, days=60, target_volatility=0.15),
    "ma_timing": moving_average_timing(prices.loc[evaluation.index], fast=50, slow=200),
    "regime_cut": regime_threshold(p_high, high_exposure=0.3, low_exposure=1.0),
    "regime_scaled": regime_probability_scaled(p_high, high_exposure=0.3, low_exposure=1.0),
}
results = {name: apply_weights(evaluation, w, cost_bps=5) for name, w in weights.items()}
print(compare_strategies(results, weights).round(3))
plot_equity_curves(results)
plot_regime_overlay(prices.loc[evaluation.index], weights["regime_cut"], p_high)
```

## Caveats

- Metrics are rf=0 on log PnL; turnover is charged only when `cost_bps > 0`.
- One evaluation window is a small sample. For an honest verdict, loop the
  recipe over evaluation years, refitting on expanding data (see
  `model_evaluation/real_data.py` for the walk-forward splitter), and pool the
  daily PnL into one table.
- Regime labels are post-hoc: `high_volatility_state` ranks states by fitted
  sd of the return feature (column 0); the model itself has no notion of
  "high volatility".
