"""Gaussian and GMM return-feature marginals for 2008 HMM diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from paper_regime_experiment import build_observations, fit_one, label_bull
from run_paper_regime import HALF_LIVES, WINDOWS


PLOT_DATES = pd.to_datetime(["2008-09-02", "2008-12-31"])


def replay_marginals(saved_rows, prices, start="2004-01-02"):
    """Replay Gaussian and GMM fits and validate historical bull forecasts."""
    saved = saved_rows.loc[saved_rows.model.isin(("Gaussian", "GMM"))].copy()
    saved["date"] = pd.to_datetime(saved["date"])
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
            for use_pca in (False, True):
                x = pca.transform(standardized) if use_pca else standardized
                for name in ("Gaussian", "GMM"):
                    models[name, use_pca] = fit_one(
                        x, returns.loc[history].to_numpy(), name,
                        seed=0, max_iter=100,
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
                # Gaussian has one component; GMM has two per state.
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
        for name in ("Gaussian", "GMM"):
            for use_pca in (False, True):
                for state in ("bull", "bear"):
                    if counts.loc[date, name, use_pca, state] != (
                        1 if name == "Gaussian" else 2
                    ):
                        raise ValueError("Missing marginal components")
    return result


def draw_marginals(marginals, name):
    """Display two-date bull/bear return marginals for one emission family."""
    if name not in ("Gaussian", "GMM"):
        raise ValueError("Expected Gaussian or GMM")
    selected = marginals.loc[marginals.model.eq(name)]
    xs = np.linspace(-5, 5, 1300)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for row, date in enumerate(PLOT_DATES):
        for column, use_pca in enumerate((False, True)):
            ax = axes[row, column]
            for state, color in (("bull", "#1769aa"), ("bear", "#c15b30")):
                parts = selected.loc[
                    selected.date.eq(date) & selected.pca.eq(use_pca)
                    & selected.state.eq(state)
                ]
                if len(parts) != (1 if name == "Gaussian" else 2):
                    raise ValueError("Expected one Gaussian or two GMM components")
                density = sum(
                    part.weight * norm.pdf(xs, loc=part.location,
                                           scale=part.scale)
                    for part in parts.itertuples()
                )
                ax.plot(xs, density, color=color, lw=2.3,
                        label=state.title())
                ax.fill_between(xs, density, color=color, alpha=0.08)
            branch = "PCA" if use_pca else "no PCA"
            ax.set_title(f"{date:%b %d, %Y}  |  {name} / {branch}", fontsize=12)
            ax.set_ylabel("Marginal density")
            ax.set_xlim(-5, 5)
            ax.set_xticks(np.arange(-5, 6, 1))
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right", fontsize=9)
            if row == 1:
                ax.set_xlabel("Standardized return feature (training scaler units)")
    fig.suptitle(f"{name} emission marginals: start of September vs. year end\n"
                 "Bull and bear in each panel", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.15,
                        hspace=0.22, wspace=0.14)
    fig.text(0.5, 0.065,
             "Sep 1 was a market holiday; Sep 2 is the first September decision.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.043,
             "PCA marginal reconstructs standardized return from retained PCs; "
             "neither curve is a raw-return density.", ha="center", fontsize=9)
    note = ("Each GMM state curve combines its two Gaussian components with "
            "fitted within-state weights." if name == "GMM" else
            "Each curve is a one-dimensional Gaussian marginal.")
    fig.text(0.5, 0.021, note, ha="center", fontsize=9)
    plt.show()
