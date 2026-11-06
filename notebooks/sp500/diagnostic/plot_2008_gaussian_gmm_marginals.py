"""Gaussian and GMM return-feature marginals for 2008 HMM diagnostics."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest_experiment import build_observations, fit_one, label_bull  # noqa: E402
from run_backtest import HALF_LIVES, WINDOWS  # noqa: E402


PLOT_DATES = pd.to_datetime(["2008-09-02", "2008-12-31"])
FAMILIES = ("Gaussian", "GMM (M=2)", "GMM (M=3)")


def parse_family(name):
    """Return the ``fit_one`` emission name and GMM component count for a label.

    Args:
        name: Branch label such as ``"Gaussian"`` or ``"GMM (M=3)"``.

    Returns:
        Tuple of emission name and components per state.
    """
    match = re.fullmatch(r"GMM \(M=(\d+)\)", name)
    if match:
        return "GMM", int(match.group(1))
    return name, 1


def replay_marginals(saved_rows, prices, start="2004-01-02"):
    """Replay Gaussian/GMM fits and validate historical bull forecasts.

    Args:
        saved_rows: Decision rows; families and PCA settings come from the
            Gaussian and ``GMM (M=k)`` rows, so merged PCA-only frames
            replay the PCA branches only.
        prices: S&P 500 price-index series.
        start: First experiment date, fixing the initial training window.

    Returns:
        Component rows (date, model, pca, state, component, weight,
        location, scale, forecast_error) for the two plot dates.
    """
    saved = saved_rows.loc[saved_rows.model.isin(FAMILIES)].copy()
    saved["date"] = pd.to_datetime(saved["date"])
    families = [name for name in FAMILIES if saved.model.eq(name).any()]
    pca_settings = tuple(sorted(map(bool, saved["pca"].unique()), reverse=True))
    expected = saved.set_index(["date", "model", "pca"])
    decision_dates = pd.DatetimeIndex(saved.date.unique()).sort_values()
    refresh_dates = pd.DatetimeIndex(saved.loc[saved.refresh, "date"].unique())
    prior_refresh = refresh_dates[refresh_dates <= PLOT_DATES[0]][-1]
    features = build_observations(prices, windows=WINDOWS,
                                  half_lives=HALF_LIVES).dropna()
    returns = prices.pct_change(fill_method=None)
    initial = features.loc[
        (features.index > pd.Timestamp(start) - pd.DateOffset(years=10))
        & (features.index <= pd.Timestamp(start))
    ].index
    return_feature = features.columns.get_loc("return")
    models = {}
    rows = []
    for date in decision_dates[(decision_dates >= prior_refresh)
                               & (decision_dates <= PLOT_DATES[-1])]:
        history = features.loc[initial[0]:date].index
        refresh = date in refresh_dates
        if refresh:
            scaler = StandardScaler().fit(features.loc[history])
            standardized = scaler.transform(features.loc[history])
            pca = PCA(n_components=0.95, svd_solver="full").fit(standardized)
            models = {}
            for use_pca in pca_settings:
                x = pca.transform(standardized) if use_pca else standardized
                for name in families:
                    emission, components = parse_family(name)
                    models[name, use_pca] = fit_one(
                        x, returns.loc[history].to_numpy(), emission,
                        seed=0, max_iter=100, gmm_components=components,
                    )[0]
        standardized = scaler.transform(features.loc[history])
        for (name, use_pca), model in models.items():
            x = pca.transform(standardized) if use_pca else standardized
            if not refresh:
                model.continue_fit(x, max_iter=3, tol=len(x) * 1e-5)
            bull = label_bull(model, x, returns.loc[history].to_numpy())
            _, filtered, _ = model.filter(x)
            forecast = float((filtered[-1] @ model.states.transitions)[bull])
            error = abs(forecast - expected.loc[
                (date, name, use_pca), "bull_probability"
            ])
            if error >= 1e-8:
                raise ValueError(f"Replay forecast mismatch: {date}, {name}, "
                                 f"PCA={use_pca}, error={error}")
            if date not in PLOT_DATES:
                continue
            direction = (pca.components_[:, return_feature] if use_pca
                         else np.eye(x.shape[1])[return_feature])
            intercept = float(pca.mean_[return_feature]) if use_pca else 0.0
            for state, index in (("bull", bull), ("bear", 1 - bull)):
                emission = model.emission
                # Gaussian has one component; GMM has M components per state.
                if name == "Gaussian":
                    means = emission.means[index][None, :]
                    covariances = emission.covariances[index][None, :, :]
                    weights = np.ones(1)
                else:
                    means = emission.means[index]
                    covariances = emission.covariances[index]
                    weights = emission.mixture_weights[index].copy()
                for component, (weight, mean, covariance) in enumerate(
                    zip(weights, means, covariances), start=1
                ):
                    location = float(intercept + direction @ mean)
                    scale = float(np.sqrt(direction @ covariance @ direction))
                    if scale <= 0:
                        raise ValueError("Non-positive marginal scale")
                    rows.append({
                        "date": date, "model": name, "pca": use_pca,
                        "state": state, "component": component,
                        "weight": float(weight), "location": location,
                        "scale": scale, "forecast_error": error,
                    })
    result = pd.DataFrame(rows)
    counts = result.groupby(["date", "model", "pca", "state"]).size()
    for date in PLOT_DATES:
        for name in families:
            for use_pca in pca_settings:
                expected_parts = parse_family(name)[1]
                for state in ("bull", "bear"):
                    if counts.loc[date, name, use_pca, state] != expected_parts:
                        raise ValueError("Missing marginal components")
    return result


def draw_marginals(marginals, name, pca=True):
    """Display one branch's two-date bull/bear marginals in a single row.

    Args:
        marginals: Component rows from ``replay_marginals``.
        name: Family label, e.g. ``"GMM (M=3)"``.
        pca: PCA setting of the branch to plot.
    """
    if name not in FAMILIES:
        raise ValueError(f"Expected one of {FAMILIES}")
    selected = marginals.loc[marginals.model.eq(name) & marginals.pca.eq(pca)]
    if selected.empty:
        raise ValueError(f"No marginals for {name} / PCA={pca}")
    expected_parts = parse_family(name)[1]
    xs = np.linspace(-5, 5, 1300)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for column, date in enumerate(PLOT_DATES):
        ax = axes[column]
        for state, color in (("bull", "#1769aa"), ("bear", "#c15b30")):
            parts = selected.loc[
                selected.date.eq(date) & selected.state.eq(state)
            ]
            if len(parts) != expected_parts:
                raise ValueError("Component count mismatch for one panel")
            density = sum(
                part.weight * norm.pdf(xs, loc=part.location, scale=part.scale)
                for part in parts.itertuples()
            )
            ax.plot(xs, density, color=color, lw=2.3, label=state.title())
            ax.fill_between(xs, density, color=color, alpha=0.08)
        ax.set_title(f"{date:%b %d, %Y}  |  {name}"
                        , fontsize=12)
        ax.set_ylabel("Marginal density")
        ax.set_xlim(-5, 5)
        ax.set_xticks(np.arange(-5, 6, 1))
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("Standardized return feature")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.16, wspace=0.12)
    plt.show()
