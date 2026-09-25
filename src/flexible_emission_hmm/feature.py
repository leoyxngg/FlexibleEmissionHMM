"""Daily return features for price series ordered from oldest to newest."""

from numbers import Integral

import numpy as np
import pandas as pd


def daily_log_return(prices: pd.Series) -> pd.Series:
    """Return r_t = log(P_t) - log(P_{t-1}), preserving the input index.

    Prices must be positive and finite (missing observations are allowed).
    The first observation and returns adjacent to missing prices are NaN.
    """
    prices = prices.astype(float)
    observed = prices.dropna()
    if (observed <= 0).any() or not np.isfinite(observed).all():
        raise ValueError("Prices must be positive and finite")
    return np.log(prices).diff().rename("r_t")


def _validate_window(days: int, minimum: int) -> None:
    if isinstance(days, bool) or not isinstance(days, Integral) or days < minimum:
        raise ValueError(f"days must be an integer of at least {minimum}")


def rolling_volatility(returns: pd.Series, days: int = 60) -> pd.Series:
    """Return sigma_t: trailing sample standard deviation of log returns.

    Includes the current return, requires `days` nonmissing observations,
    and uses ddof=1. Values are not annualized; days must be at least two.
    """
    _validate_window(days, minimum=2)
    return returns.rolling(window=days, min_periods=days).std(ddof=1).rename("sigma_t")


def rolling_mean_return(returns: pd.Series, days: int = 20) -> pd.Series:
    """Return m_t: trailing mean log return, including the current return.

    Requires `days` nonmissing observations; days must be at least one.
    """
    _validate_window(days, minimum=1)
    return returns.rolling(window=days, min_periods=days).mean().rename("m_t")


def compute_features(
    prices: pd.Series,
    volatility_days: int = 60,
    momentum_days: int = 20,
) -> pd.DataFrame:
    """Produce r_t, sigma_t, and m_t with configurable rolling windows.

    Each row represents one trading day. Supply prices in chronological
    order. The input index and warm-up NaNs are preserved.

    Example:
        features = compute_features(df["close"], volatility_days=60, momentum_days=20)
    """
    returns = daily_log_return(prices)
    return pd.concat(
        [
            returns,
            rolling_volatility(returns, days=volatility_days),
            rolling_mean_return(returns, days=momentum_days),
        ],
        axis=1,
    )
