"""Causal market-strategy weight builders.

Convention: ``weights[t]`` is the position held during day ``t``, chosen using
data through day ``t-1`` only. PnL is then simply ``weights * returns`` with no
further shifting (see ``apply_weights``). Functions that use prices/returns
shift internally, so callers cannot introduce look-ahead by accident;
regime-probability inputs must already be causal (see ``signals`` module).
"""

import numpy as np
import pandas as pd


def _as_returns(returns: pd.Series) -> pd.Series:
    returns = returns.astype(float)
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns must be sorted chronologically")
    return returns


def buy_and_hold(returns: pd.Series) -> pd.Series:
    """Fully invested: the strategy every benchmark must beat."""
    return pd.Series(1.0, index=_as_returns(returns).index, name="buy_and_hold")


def constant_exposure(returns: pd.Series, exposure: float) -> pd.Series:
    """Fixed fraction of capital invested every day (e.g. 0.5)."""
    if not 0.0 <= exposure <= 5.0:
        raise ValueError("exposure must be between 0 and 5")
    return pd.Series(float(exposure), index=_as_returns(returns).index, name="constant_exposure")


def volatility_target(
    returns: pd.Series,
    days: int = 60,
    target_volatility: float = 0.15,
    max_weight: float = 1.5,
    periods_per_year: int = 252,
) -> pd.Series:
    """Naive vol targeting: w_t = min(target / trailing vol, max_weight).

    The trailing volatility at day t-1 sizes day t's position - the strongest
    zero-signal baseline: if an HMM cannot beat divide-by-recent-vol, the
    regimes add nothing.
    """
    returns = _as_returns(returns)
    if days < 2:
        raise ValueError("days must be at least 2")
    trailing_vol = returns.rolling(days, min_periods=days).std(ddof=1) * np.sqrt(periods_per_year)
    weights = (target_volatility / trailing_vol).clip(upper=max_weight)
    return weights.shift(1).fillna(0.0).rename("volatility_target")


def moving_average_timing(
    prices: pd.Series,
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """Time-series momentum: long after the fast SMA crosses above the slow SMA.

    Signal from closes through t-1 applied to day t. The classic
    no-regime-signal trend baseline (Moskowitz-Ooi-Pedersen style, long/flat).
    """
    if not 1 <= fast < slow:
        raise ValueError("require 1 <= fast < slow")
    prices = _as_returns(prices.astype(float))
    fast_ma = prices.rolling(fast, min_periods=fast).mean()
    slow_ma = prices.rolling(slow, min_periods=slow).mean()
    weights = (fast_ma > slow_ma).astype(float)
    return weights.shift(1).fillna(0.0).rename("moving_average_timing")


def regime_threshold(
    causal_probabilities: pd.Series,
    high_exposure: float = 0.3,
    low_exposure: float = 1.0,
    trigger: float = 0.5,
) -> pd.Series:
    """Binary regime switch: cut exposure when causal P(high-vol) > trigger.

    ``causal_probabilities[t]`` must be P(S_t | data < t) - the first output of
    ``signals.hmm_causal_probabilities`` (or ``filter``), never a smoothed
    posterior. No extra shift: the probability is already one-step ahead.
    """
    probabilities = _as_returns(causal_probabilities)
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("causal_probabilities must lie in [0, 1]")
    weights = np.where(probabilities > trigger, high_exposure, low_exposure)
    return pd.Series(weights, index=probabilities.index, name="regime_threshold")


def regime_probability_scaled(
    causal_probabilities: pd.Series,
    high_exposure: float = 0.3,
    low_exposure: float = 1.0,
) -> pd.Series:
    """Blend exposure linearly in the regime probability: w = low + (high-low) * p.

    Continuous version of ``regime_threshold``; trades more smoothly and lets
    the benchmark test how much of the value is in the *magnitude* of the
    probability, not just the 0.5 crossing.
    """
    probabilities = _as_returns(causal_probabilities)
    weights = low_exposure + (high_exposure - low_exposure) * probabilities
    return pd.Series(weights, index=probabilities.index, name="regime_probability_scaled")


def apply_weights(
    returns: pd.Series,
    weights: pd.Series,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily log PnL of a causal weight series: pnl_t = w_t * r_t - costs.

    ``cost_bps`` is a per-unit-turnover transaction cost in basis points,
    charged on |w_t - w_{t-1}|. Weights are assumed causal by convention
    (position for day t decided by t-1), so no shift is applied here.
    """
    returns = _as_returns(returns)
    weights = weights.reindex(returns.index)
    if weights.isna().any():
        raise ValueError("weights must cover every return date (reindex before applying)")
    pnl = weights * returns
    if cost_bps:
        turnover = weights.diff().abs().fillna(weights.abs())
        pnl = pnl - turnover * (cost_bps / 1e4)
    return pnl.rename("pnl")


def regime_sharpe_table(
    labels: pd.Series,
    model_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """Annualized Sharpe of every model within every regime (the per-regime tables
    of Wang et al. 2020 / Ang-Bekaert 2002), plus the per-regime day count.

    ``labels``: regime index per day - for TRAINING use smoothed labels
    (they describe the past); rotation on EVALUATION must use causal labels
    via ``regime_model_rotation``. ``model_returns``: daily log returns, one
    column per model. Counts matter: regimes with a few hundred (autocorrelated)
    days are anecdotal - see ``build_rotation_map``'s min_count guard.
    """
    labels = _as_returns(labels)
    aligned = model_returns.reindex(labels.index)
    if aligned.isna().any().any():
        raise ValueError("every model must cover every labeled date")
    grouped = aligned.groupby(labels.values)
    sharpe = (grouped.mean() * periods_per_year) / (
        grouped.std(ddof=1) * np.sqrt(periods_per_year)
    )
    counts = grouped.count().min(axis=1)
    return sharpe, counts.rename("min_days")


def build_rotation_map(
    sharpe_table: pd.Series,
    counts: pd.Series,
    fallback: str = "cash",
    margin: float = 0.25,
    min_days: int = 250,
) -> dict:
    """Regime -> winning model, with the two guards the naive argmax skips.

    Wang et al. (2020) take the raw argmax of per-regime Sharpe; that is fragile
    when the winner leads the runner-up by noise or the regime barely appears in
    training. This builder requires the winner to beat the runner-up by
    ``margin`` annualized Sharpe and the regime to have at least ``min_days``
    observations, and falls back (e.g. to the cash leg, whose regime
    expectations are in ``models`` docstrings) otherwise.
    """
    rotation = {}
    for regime in sharpe_table.index:
        row = sharpe_table.loc[regime].dropna().sort_values(ascending=False)
        if len(row) == 0 or counts.get(regime, 0) < min_days:
            rotation[regime] = fallback
            continue
        if len(row) > 1 and row.iloc[0] - row.iloc[1] < margin:
            rotation[regime] = fallback
            continue
        rotation[regime] = row.index[0]
    return rotation


def regime_model_rotation(
    causal_labels: pd.Series,
    model_returns: pd.DataFrame,
    rotation_map: dict,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Wang et al. (2020) rotation: each day, hold 100% of the model chosen for
    that regime.

    ``causal_labels[t]``: regime for day t from the argmax of the one-step-ahead
    filter (``signals.hmm_causal_probabilities``), never a smoothed posterior -
    Wang et al.'s same-day PDF detector leaks the day being traded.
    ``model_returns``: daily log returns per model (see ``models`` for the
    standard universe and which regime each leg should win).
    ``rotation_map``: regime -> model name (build with ``build_rotation_map``
    on TRAINING data only). Costs are charged per full switch
    (100% -> 100% turnover), so the OOS verdict prices the switching.
    """
    causal_labels = _as_returns(causal_labels).astype(int)
    aligned = model_returns.reindex(causal_labels.index)
    missing = set(causal_labels.unique()) - set(rotation_map)
    if missing:
        raise ValueError(f"rotation_map has no model for regimes {sorted(missing)}")
    missing_models = {rotation_map[r] for r in causal_labels.unique()} - set(aligned.columns)
    if missing_models:
        raise ValueError(f"rotation_map names unknown models: {sorted(missing_models)}")
    pnl = pd.Series(0.0, index=causal_labels.index)
    for regime, model in rotation_map.items():
        pnl[causal_labels.values == regime] = aligned[model].to_numpy()[causal_labels.values == regime]
    if cost_bps:
        switches = (causal_labels.diff() != 0) & causal_labels.diff().notna()
        pnl = pnl - switches.astype(float) * (2 * cost_bps / 1e4)
    return pnl.rename("rotation_pnl")
