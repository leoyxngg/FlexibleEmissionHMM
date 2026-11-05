"""Daily return features for price series ordered from oldest to newest."""

from numbers import Integral
from collections.abc import Sequence

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


## Paper-inspired regime features


def regime_price_features(
    prices: pd.Series,
    windows: Sequence[int] = (5, 10, 60),
    half_lives: Sequence[int] = (5, 10, 60),
    downside_target: float = 0.0,
    downside_floor: float = 1e-6,
) -> pd.DataFrame:
    """Build causal return, downside, and Sortino-style features from prices.

    This is a documented approximation of King et al. (2025), which does not
    specify its final horizons or feature list. Each EWMA has a full rolling
    warm-up; a row uses prices no later than its own date. The downside
    deviation is the population standard deviation of negative
    target-relative returns after setting nonnegative values to zero, as
    described in the paper. Sortino uses the
    trailing mean excess return over this downside deviation.

    Args:
        prices: Chronologically ordered positive daily index prices.
        windows: Trailing trading-day windows for downside and Sortino.
        half_lives: EWMA half-lives applied to every base feature.
        downside_target: Daily target simple return (zero by default).
        downside_floor: Denominator floor to keep Sortino finite.

    Returns:
        Features with warm-up NaNs and the original index preserved.

    Raises:
        ValueError: Horizons, target, floor, or price index is invalid.
    """
    for values in (windows, half_lives):
        if not values or len(set(values)) != len(values):
            raise ValueError("Horizons must be nonempty and unique")
        for value in values:
            _validate_window(value, minimum=2)
    if not np.isfinite(downside_target):
        raise ValueError("downside_target must be finite")
    if not np.isfinite(downside_floor) or downside_floor <= 0:
        raise ValueError("downside_floor must be positive and finite")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price index must be unique and chronologically ordered")

    daily_log_return(prices)  # Validate all observed prices before division.
    simple_return = prices.astype(float).pct_change(fill_method=None)
    excess = simple_return - downside_target
    negative = excess.clip(upper=0)
    base = {"return": simple_return}
    for window in windows:
        # Keep nonnegative observations as zero before computing deviation.
        downside = negative.rolling(window, min_periods=window).std(ddof=0)
        average = excess.rolling(window, min_periods=window).mean()
        base[f"downside_{window}"] = downside
        base[f"sortino_{window}"] = average / downside.clip(lower=downside_floor)
    features = pd.DataFrame(base, index=prices.index)
    # A full EWMA warm-up excludes the first poorly supported observations.
    for name, series in base.items():
        for half_life in half_lives:
            features[f"{name}_ewm_{half_life}"] = series.ewm(
                halflife=half_life, adjust=False, min_periods=half_life,
            ).mean()
    features.attrs["return_definition"] = "simple price return"
    return features
