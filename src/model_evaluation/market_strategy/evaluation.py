"""Performance summaries and plots for HMM regime-switch benchmarks.

Inputs are daily *log* PnL series from ``strategies.apply_weights`` (one row
per trading day, causally weighted). All metrics are rf=0.
"""

import numpy as np
import pandas as pd


def drawdown(log_pnl: pd.Series) -> pd.Series:
    """Peak-to-trough drawdown series of a log PnL stream."""
    equity = np.exp(log_pnl.astype(float).cumsum())
    return equity / equity.cummax() - 1.0


def performance_summary(
    log_pnl: pd.Series,
    weights: pd.Series | None = None,
    periods_per_year: int = 252,
) -> pd.Series:
    """Annualized performance statistics of one strategy.

    Reports return, risk, Sharpe/Sortino, drawdown, Calmar, hit rate, plus
    exposure and annual turnover (one-way) when weights are supplied.
    """
    pnl = log_pnl.astype(float).dropna()
    if len(pnl) < 2:
        raise ValueError("need at least two PnL observations")
    years = len(pnl) / periods_per_year
    equity = np.exp(pnl.cumsum())
    total_return = equity.iloc[-1] - 1.0
    annual_return = pnl.mean() * periods_per_year
    annual_volatility = pnl.std(ddof=1) * np.sqrt(periods_per_year)
    downside = pnl[pnl < 0]
    downside_volatility = downside.std(ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    max_drawdown = drawdown(pnl).min()

    summary = pd.Series(
        {
            "years": years,
            "total_return": total_return,
            "ann_return": annual_return,
            "ann_volatility": annual_volatility,
            "sharpe": annual_return / annual_volatility if annual_volatility > 0 else np.nan,
            "sortino": annual_return / downside_volatility if downside_volatility and downside_volatility > 0 else np.nan,
            "max_drawdown": max_drawdown,
            "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
            "hit_rate": float((pnl > 0).mean()),
        }
    )
    if weights is not None:
        aligned = weights.reindex(pnl.index)
        turnover = aligned.diff().abs().sum() / years
        summary["exposure"] = float(aligned.abs().mean())
        summary["turnover_per_year"] = float(turnover)
    return summary


def compare_strategies(
    results: dict[str, pd.Series],
    weights: dict[str, pd.Series] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """One row per strategy, sorted by Sharpe.

    ``results`` maps strategy name to log PnL (from ``apply_weights``);
    ``weights`` optionally maps the same names to the causal weight series.
    """
    if not results:
        raise ValueError("results must contain at least one strategy")
    rows = {
        name: performance_summary(pnl, None if weights is None else weights.get(name), periods_per_year)
        for name, pnl in results.items()
    }
    table = pd.DataFrame(rows).T
    return table.sort_values("sharpe", ascending=False)


def plot_equity_curves(results: dict[str, pd.Series], title: str = "Equity curves") -> "object":
    """Cumulative growth of $1 for each strategy on one chart (date axes when available)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, pnl in results.items():
        equity = np.exp(pnl.astype(float).cumsum())
        ax.plot(equity.index, equity.to_numpy(), label=name)
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set(title=title, ylabel="growth of $1")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_regime_overlay(
    prices: pd.Series,
    weights: pd.Series,
    causal_probabilities: pd.Series,
    title: str = "HMM regime strategy",
    shading_trigger: float = 0.5,
) -> "object":
    """Three panels: price with regime shading, strategy exposure, regime probability."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    axes[0].plot(prices.index, prices.to_numpy() / prices.iloc[0], color="black", label="index")
    axes[0].fill_between(
        causal_probabilities.index,
        0,
        1,
        where=(causal_probabilities > shading_trigger),
        transform=axes[0].get_xaxis_transform(),
        color="steelblue",
        alpha=0.2,
        label=f"P(high vol) > {shading_trigger}",
    )
    axes[0].set(title=title)
    axes[0].legend(loc="upper left")
    axes[1].fill_between(weights.index, weights, color="darkseagreen", alpha=0.6)
    axes[1].set(ylabel="exposure", ylim=(0, max(1.05, float(weights.max()) * 1.05)))
    axes[2].plot(causal_probabilities.index, causal_probabilities, color="steelblue")
    axes[2].axhline(shading_trigger, color="gray", ls="--", lw=0.8)
    axes[2].set(ylabel="P(high vol)", ylim=(0, 1))
    fig.tight_layout()
    return fig
