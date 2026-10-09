"""Rotation universe: the candidate "models" a regime strategy rotates between.

Each builder is data-agnostic: it takes one daily log-return Series (a proxy
asset or factor) and returns it validated and named, so a rotation map can
reference it by name. The docstrings record the regime each model is EXPECTED
to win, following Wang-Lin-Mikhelson (2020, jrfm-13-00311) and
Kritzman-Page-Turkington (2012) - use the per-regime Sharpe table
(``strategies.regime_sharpe_table``) on YOUR training window to confirm before
trusting the priors.
"""

import numpy as np
import pandas as pd


def _proxy(returns: pd.Series, name: str) -> pd.Series:
    returns = returns.astype(float)
    if not returns.index.is_monotonic_increasing:
        raise ValueError(f"{name} proxy returns must be sorted chronologically")
    if returns.isna().any():
        raise ValueError(f"{name} proxy returns contain NaNs")
    return returns.rename(name)


def market_proxy(returns: pd.Series) -> pd.Series:
    """Capitalization-weighted large-cap equity (e.g. SPY / S&P 500 total return).

    Regime expectation: strongest in BULL regimes (full beta earns the equity
    risk premium); weakest in BEAR regimes. Wang et al. (2020): SPY Sharpe was
    negative across their high-vol OOS window; Kritzman et al. (2012):
    cap-weighted equity leads calm bull regimes and falls last in crashes.
    """
    return _proxy(returns, "market")


def equal_weight_proxy(returns: pd.Series) -> pd.Series:
    """Equal-weighted broad equity (small/mid tilt, breadth beta).

    Regime expectation: best in SIDEWAYS / early-recovery regimes where gains
    rotate across many stocks (breadth expands) and in the post-crash snapback
    where small caps outperform; degrades in late bull (concentration) and
    suffers most in BEAR regimes on liquidity flight. Kritzman et al. (2012)
    rotate into equal-weight in their mid-cycle regime.
    """
    return _proxy(returns, "equal_weight")


def long_treasuries_proxy(returns: pd.Series) -> pd.Series:
    """Long-duration US Treasuries (flight-to-quality bond leg).

    Regime expectation: strongest in BEAR / high-volatility regimes (negative
    correlation with equities when panic reprices rates down; Wang et al.'s
    best bear-regime performers all carried crash-protective exposure);
    weakest in reflationary BULL regimes with rising rates. This is the
    defense leg Kritzman et al. (2012) rotate into in their panic regime.
    """
    return _proxy(returns, "long_treasuries")


def credit_commodities_proxy(returns: pd.Series) -> pd.Series:
    """Credit + commodities composite (reflation / late-cycle risk leg).

    Regime expectation: best in late-BULL regimes where growth and
    inflation risk premia coincide (commodities rally, spreads tighten);
    among the WORST performers in BEAR regimes (spreads widen, commodities
    gap down with demand). A Wang et al.-style momentum/value rotation leans
    on this leg for bull-regime excess return.
    """
    return _proxy(returns, "credit_commodities")


def cash_proxy(index: pd.DatetimeIndex) -> pd.Series:
    """Cash (zero return) - the neutral fallback leg.

    Regime expectation: never wins; it is the benchmark every risk model
    must beat within a regime and the standard fallback when a regime's
    training sample is too small to pick a winner. Kritzman et al. (2012)
    rotate fully into cash in their deepest panic regime.
    """
    return pd.Series(0.0, index=index, name="cash")
