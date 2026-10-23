"""Historical Student-t emission summaries for notebook diagnostics."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from paper_regime_experiment import build_observations, fit_one, label_bull
from run_paper_regime import HALF_LIVES, WINDOWS


COLORS = {"Student-t / no PCA": "#c15b30", "Student-t / PCA": "#1769aa"}


def draw_market(ax, prices, plot_start, plot_end):
    """Draw daily S&P 500 index log returns on the supplied axis."""
    log_returns = np.log(prices).diff().loc[plot_start:plot_end] * 100
    ax.bar(log_returns.index, log_returns, width=0.85,
           color=np.where(log_returns >= 0, "#389269", "#bb5660"))
    ax.axhline(0, color="#444", lw=0.8)
    ax.set(ylabel="S&P 500 daily\nlog return (%)",
           xlabel="2008 trading date", xlim=(plot_start, plot_end))
    ax.grid(axis="y", alpha=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def draw_figures(data, prices):
    """Display the yearly df and location/dispersion comparison figures."""
    plot_start, plot_end = data.date.min(), data.date.max()
    fig, (ax, market) = plt.subplots(
        2, 1, figsize=(15, 8.5), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for name, color in COLORS.items():
        group = data.loc[data.branch == name]
        ax.plot(group.date, group.bull_df, color=color, lw=1.7,
                label=f"{name} / bull (df)")
        ax.plot(group.date, group.bear_df, color=color, lw=1.3, ls="--",
                label=f"{name} / bear (df)")
    ax.axhline(2.1, color="gray", lw=0.8, ls=":",
               label="Lower bound: df = 2.1")
    for date in data.loc[data.refresh, "date"].unique():
        ax.axvline(date, color="#aaa", lw=0.7, alpha=0.6)
    ax.plot([], [], color="#aaa", lw=0.7, label="Full refit")
    ax.set(title="Student-t emission degrees of freedom · 2008",
           ylabel="Degrees of freedom (df)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, loc="upper left", fontsize=8)
    draw_market(market, prices, plot_start, plot_end)
    fig.suptitle("Historical replay: Student-t tail parameters and index returns",
                 fontsize=15)
    plt.show()

    fig, (means, spread, market) = plt.subplots(
        3, 1, figsize=(15, 10), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [1.25, 1.25, 1]},
    )
    for name, color in COLORS.items():
        group = data.loc[data.branch == name]
        for state, style in (("bull", "-"), ("bear", "--")):
            label = f"{name} / {state}"
            means.plot(group.date, group[f"{state}_mean_norm"],
                       color=color, lw=1.7, ls=style, label=label)
            spread.plot(group.date, group[f"{state}_median_std"],
                        color=color, lw=1.7, ls=style, label=label)
    for axis in (means, spread):
        for date in data.loc[data.refresh, "date"].unique():
            axis.axvline(date, color="#aaa", lw=0.7, alpha=0.6)
        axis.grid(alpha=0.2)
    means.set(title="Fitted Student-t location and dispersion · 2008",
              ylabel="Mean-vector norm")
    means.legend(ncol=2, fontsize=8, loc="upper left")
    spread.set(ylabel="Median marginal std")
    draw_market(market, prices, plot_start, plot_end)
    fig.suptitle("Historical replay: Student-t emission summaries and index returns",
                 fontsize=15)
    plt.show()


def draw_return_marginals(marginals):
    """Display bull and bear Student-t return marginals on two decision dates."""
    dates = pd.to_datetime(["2008-09-02", "2008-12-31"])
    branches = ("Student-t / no PCA", "Student-t / PCA")
    selected = marginals.loc[marginals.date.isin(dates)]
    if (len(selected) != 8 or selected.duplicated(
        ["date", "branch", "state"]
    ).any()):
        raise ValueError("Expected bull and bear marginals for both dates/branches")
    xs = np.linspace(-5, 5, 1300)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for row, date in enumerate(dates):
        for column, branch in enumerate(branches):
            ax = axes[row, column]
            for state, color in (("bull", "#1769aa"), ("bear", "#c15b30")):
                item = selected.loc[
                    selected.date.eq(date) & selected.branch.eq(branch)
                    & selected.state.eq(state)
                ].iloc[0]
                density = t.pdf(xs, item.df, loc=item.location,
                                scale=item.scale)
                std = item.scale * np.sqrt(item.df / (item.df - 2))
                ax.plot(xs, density, color=color, lw=2.3,
                        label=f"{state.title()} · df={item.df:.2f} · std={std:.2f}")
                ax.fill_between(xs, density, color=color, alpha=0.08)
            ax.set_title(f"{date:%b %d, %Y}  |  {branch}", fontsize=12)
            ax.set_ylabel("Marginal density")
            ax.set_xlim(-5, 5)
            ax.set_xticks(np.arange(-5, 6, 1))
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right", fontsize=9)
            if row == 1:
                ax.set_xlabel("Standardized return feature (training scaler units)")
    fig.suptitle("Student-t emission marginals: start of September vs. year end\n"
                 "Bull and bear in each panel", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.15,
                        hspace=0.22, wspace=0.14)
    fig.text(0.5, 0.065,
             "Sep 1 was a market holiday; Sep 2 is the first September decision.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.043,
             "PCA marginal reconstructs standardized return from retained PCs; "
             "neither curve is a raw-return density.", ha="center", fontsize=9)
    fig.text(0.5, 0.021,
             "Each panel shows a one-dimensional marginal, not the full "
             "multivariate emission density.", ha="center", fontsize=9)
    plt.show()


def replay_emissions(saved_rows, prices, start="2004-01-02", year=2008):
    """Reconstruct Student-t emissions from a trusted checkpoint's decision rows."""
    plot_start = pd.Timestamp(year=year, month=1, day=1)
    plot_end = pd.Timestamp(year=year, month=12, day=31)
    saved = saved_rows.loc[saved_rows.model.eq("Student-t")].copy()
    saved["date"] = pd.to_datetime(saved["date"])
    decision_dates = pd.DatetimeIndex(saved.date.unique()).sort_values()
    refresh_dates = pd.DatetimeIndex(saved.loc[saved.refresh, "date"].unique())
    prior_refresh = refresh_dates[refresh_dates <= plot_start][-1]
    features = build_observations(prices, windows=WINDOWS,
                                  half_lives=HALF_LIVES).dropna()
    returns = prices.pct_change(fill_method=None)
    initial = features.loc[
        (features.index > pd.Timestamp(start) - pd.DateOffset(years=10))
        & (features.index <= pd.Timestamp(start))
    ].index
    expected = saved.set_index(["date", "pca"])
    models = {}
    rows = []
    coordinates = []
    return_marginals = []
    snapshots = pd.to_datetime([
        "2008-09-02", "2008-10-03", "2008-11-26", "2008-11-28",
        "2008-12-01", "2008-12-31",
    ])
    return_feature = features.columns.get_loc("return")
    for date in decision_dates[(decision_dates >= prior_refresh)
                               & (decision_dates <= plot_end)]:
        history = features.loc[initial[0]:date].index
        refresh = date in refresh_dates
        if refresh:
            scaler = StandardScaler().fit(features.loc[history])
            standardized = scaler.transform(features.loc[history])
            pca = PCA(n_components=0.95, svd_solver="full").fit(standardized)
            models = {}
            for use_pca in (False, True):
                x = pca.transform(standardized) if use_pca else standardized
                models[use_pca] = fit_one(
                    x, returns.loc[history].to_numpy(), "Student-t",
                    seed=0, max_iter=100,
                )[0]
        standardized = scaler.transform(features.loc[history])
        for use_pca, model in models.items():
            x = pca.transform(standardized) if use_pca else standardized
            if not refresh:
                model.continue_fit(x, max_iter=3, tol=len(x) * 1e-5)
            bull = label_bull(model, x, returns.loc[history].to_numpy())
            _, filtered, _ = model.filter(x)
            forecast = float((filtered[-1] @ model.states.transitions)[bull])
            error = abs(forecast - expected.loc[(date, use_pca), "bull_probability"])
            assert error < 1e-8, (date, use_pca, error)
            if date < plot_start:
                continue
            row = {"date": date, "branch": ("Student-t / PCA" if use_pca
                                              else "Student-t / no PCA"),
                   "refresh": refresh, "components": x.shape[1],
                   "max_forecast_error": error}
            for state, index in (("bull", bull), ("bear", 1 - bull)):
                df = float(model.emission.dof[index])
                marginal_std = np.sqrt(
                    np.diag(model.emission.covariances[index]) * df / (df - 2)
                )
                row[f"{state}_df"] = df
                row[f"{state}_mean_norm"] = np.linalg.norm(
                    model.emission.means[index]
                )
                row[f"{state}_median_std"] = np.median(marginal_std)
                if date in snapshots:
                    direction = (pca.components_[:, return_feature] if use_pca
                                 else np.eye(x.shape[1])[return_feature])
                    location = float(direction @ model.emission.means[index])
                    if use_pca:
                        location += float(pca.mean_[return_feature])
                    scale = float(np.sqrt(
                        direction @ model.emission.covariances[index] @ direction
                    ))
                    return_marginals.append({
                        "date": date, "branch": row["branch"],
                        "state": state, "df": df,
                        "location": location, "scale": scale,
                    })
                    for column, mean, std in zip(features.columns if not use_pca
                                                 else (f"PC{i + 1}" for i in
                                                       range(x.shape[1])),
                                                 model.emission.means[index],
                                                 marginal_std):
                        coordinates.append({
                            "date": date.date().isoformat(),
                            "branch": row["branch"], "state": state,
                            "coordinate": column, "df": df,
                            "mean": mean, "marginal_std": std,
                        })
            rows.append(row)
    data = pd.DataFrame(rows)
    year_dates = decision_dates[(decision_dates >= plot_start)
                                & (decision_dates <= plot_end)]
    assert len(data) == 2 * len(year_dates)
    return data, pd.DataFrame(coordinates), pd.DataFrame(return_marginals)
