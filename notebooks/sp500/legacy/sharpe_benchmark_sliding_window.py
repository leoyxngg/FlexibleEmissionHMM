"""Refit S&P 500 benchmarks in annual 20/3/1/3 calendar-year windows."""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR))

import sp500_regime_algorithm_matrix as matrix  # noqa: E402
import sp500_sharpe_benchmark as benchmark  # noqa: E402
from model_evaluation.market_strategy import (  # noqa: E402
    compare_strategies,
    hmmlearn_causal_probabilities,
    plot_equity_curves,
)

TRAIN_YEARS, VALIDATION_YEARS, GAP_YEARS, TEST_YEARS = 20, 3, 1, 3
TOTAL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + GAP_YEARS + TEST_YEARS


def load_prices():
    """Load all available local S&P 500 closing prices."""
    data = pd.read_csv(matrix.DATA_PATH, parse_dates=["YYYYMMDD"])
    return data.set_index("YYYYMMDD")["DlyPrcInd"].sort_index()


def available_windows(prices):
    """Enumerate complete calendar spans with pretraining emission/ML warmup.

    Require observations in the first and last seven calendar days of every
    year and all twelve months. This boundary-coverage rule tolerates exchange
    holidays; it is not an exchange-calendar audit of missing interior sessions.
    Feature rows must cover every training return, including the first day.
    """
    returns, emissions, features = matrix.build_features(prices)
    complete = set()
    for year, group in prices.groupby(prices.index.year):
        if (group.index[0] <= pd.Timestamp(year, 1, 7)
                and group.index[-1] >= pd.Timestamp(year, 12, 25)
                and group.index.month.nunique() == 12):
            complete.add(year)
    rows = []
    for year in sorted(complete):
        if not set(range(year, year + TOTAL_YEARS)).issubset(complete):
            continue
        train_index = prices.loc[f"{year}":f"{year + 19}"].index
        if not all(train_index.isin(index).all() for index in (
                returns.index, emissions.index, features.index)):
            continue
        row = {"window": str(year)}
        for name, offset, length in (
                ("training", 0, 20), ("validation", 20, 3),
                ("gap", 23, 1), ("test", 24, 3)):
            row[f"{name}_start"] = f"{year + offset}-01-01"
            row[f"{name}_end"] = f"{year + offset + length - 1}-12-31"
        rows.append(row)
    if not rows:
        raise ValueError("No complete 27-year windows with training feature warmup")
    return pd.DataFrame(rows).set_index("window")


def fit_ml(features, returns, start, end):
    """Fit the original supervised models on explicitly bounded target dates.

    Match the original scaling, optimizer, ridge penalty, forest, and seeds.
    Keep this small date-parameterized copy local rather than changing globals.
    """
    features = features.loc[start:end]
    targets = returns.loc[features.index]
    if features.empty or (targets > 0).nunique() != 2:
        raise ValueError("Training requires observations from both classes")
    mu, sd = features.mean(), features.std(ddof=0) + 1e-12
    scaled = ((features - mu) / sd).to_numpy()
    design = np.column_stack([np.ones(len(scaled)), scaled])
    y_up = (targets.to_numpy() > 0).astype(int)
    coefficients = np.zeros(design.shape[1])
    for _ in range(3000):
        probability = expit(design @ coefficients)
        gradient = design.T @ (probability - y_up) / len(y_up)
        gradient += coefficients / len(y_up)
        coefficients -= 0.05 * gradient
    ridge = np.linalg.solve(
        design.T @ design + 10.0 * np.eye(design.shape[1]),
        design.T @ targets.to_numpy(),
    )
    forest = RandomForestClassifier(
        n_estimators=300, max_depth=3, min_samples_leaf=50,
        random_state=0, n_jobs=-1,
    ).fit(scaled, y_up)
    return {
        "mu": mu, "sd": sd, "logistic": coefficients, "ridge": ridge,
        "return_sd": targets.std(ddof=1), "forest": forest,
        "train_accuracy": forest.score(scaled, y_up),
    }


def causal_regimes(model, emissions, index, names, training_start):
    """Filter from this window's training start without smoothing or refitting."""
    history = emissions.loc[training_start:index[-1]]
    probabilities = pd.DataFrame(
        hmmlearn_causal_probabilities(model, history.to_numpy()),
        index=history.index,
    ).loc[index]
    return probabilities.idxmax(axis=1).map(names).rename("regime")


def run_window(prices, bounds, fit_model=None, name_model=None):
    """Refit independently, select on validation, and freeze through gap/test.

    Args:
        prices: Closes through at least the window's test end.
        bounds: One row of ``available_windows``.
        fit_model: Maps training emissions to a fitted hmmlearn HMM; ``None``
            uses the original Gaussian restarts.
        name_model: Maps a fitted model to regime names and moments; ``None``
            uses the original heuristic naming.
    """
    fit_model = fit_model or matrix.fit_hmmlearn
    name_model = name_model or matrix.name_regimes
    prices = prices.loc[:bounds["test_end"]]
    returns, emissions, features = matrix.build_features(prices)
    splits = {
        name: returns.loc[bounds[f"{name}_start"]:bounds[f"{name}_end"]]
        for name in ("training", "validation", "gap", "test")
    }
    start, end = bounds["training_start"], bounds["training_end"]
    model = fit_model(emissions.loc[start:end].to_numpy())
    names, moments = name_model(model)
    history = list(model.monitor_.history)
    fit_diagnostics = {
        "iterations": model.monitor_.iter,
        "max_iter_reached": model.monitor_.iter >= model.n_iter,
        "last_log_likelihood_delta": (
            history[-1] - history[-2] if len(history) >= 2 else np.nan
        ),
    }
    fitted_ml = fit_ml(features, returns, start, end)
    validation = splits["validation"]
    validation_labels = causal_regimes(
        model, emissions, validation.index, names, start,
    )
    validation_weights = matrix.strategy_weights(
        prices, validation.index, fitted_ml,
    )
    table, counts, _ = matrix.sharpe_matrix(
        validation, validation_weights, validation_labels,
    )
    table = table.rename(columns={"test_all": "dev_all"})
    counts = counts.rename(index={"test_all": "dev_all"})
    rotation, best_single = benchmark.select_rotation(table, counts)
    test = splits["test"]
    # This forward pass includes the gap, but no gap target enters either fit
    # or selection. Strategy histories also include observed gap prices.
    labels = causal_regimes(model, emissions, test.index, names, start)
    weights = matrix.strategy_weights(prices, test.index, fitted_ml)
    weights["dev_best_single"] = weights[best_single]
    weights["co_rotation"] = benchmark.co_rotation_weights(
        weights, labels, rotation,
    )
    test_table, test_counts, pnl = matrix.sharpe_matrix(test, weights, labels)
    summary = compare_strategies(
        {name: pnl[name] for name in pnl},
        {name: weights[name] for name in weights},
    )
    # Include the initial cash peak and entry in display metrics as in costs.
    summary["max_drawdown"] = [benchmark.max_drawdown(pnl[n]) for n in summary.index]
    summary["calmar"] = summary["ann_return"] / summary["max_drawdown"].abs()
    summary.loc[summary["max_drawdown"] >= 0, "calmar"] = np.nan
    summary["turnover_per_year"] = (
        weights.diff().abs().fillna(weights.abs()).sum() / (len(test) / 252)
    )
    return {
        "bounds": dict(bounds), "prices": prices, "splits": splits,
        "model": model, "moments": moments, "transitions": model.transmat_.copy(),
        "fit_diagnostics": fit_diagnostics,
        "train_accuracy": fitted_ml["train_accuracy"],
        "validation_table": table, "validation_counts": counts,
        "validation_labels": validation_labels, "rotation": rotation,
        "best_single": best_single, "test_labels": labels,
        "test_weights": weights, "test_pnl": pnl, "test_table": test_table,
        "test_counts": test_counts, "summary": summary,
    }


def aggregate_results(results):
    """Summarize every numeric metric equally across windows, excluding nonfinite."""
    if not results:
        raise ValueError("At least one window result is required")
    aggregated = {}
    for key in ("summary", "validation_table", "test_table"):
        per_window = pd.concat(
            {name: result[key] for name, result in results.items()},
            names=["window", "strategy"],
        ).select_dtypes(include="number")
        aggregated[f"{key}_per_window"] = per_window
        aggregated[key] = per_window.replace([np.inf, -np.inf], np.nan).groupby(
            level="strategy", sort=False,
        ).agg(["mean", "std", "count"])
    selections = pd.DataFrame({
        name: {**result["rotation"], "best_single": result["best_single"]}
        for name, result in results.items()
    }).T.rename_axis("window")
    candidates = next(iter(results.values()))["validation_table"].index
    frequencies = pd.DataFrame({
        column: selections[column].value_counts().reindex(candidates, fill_value=0)
        for column in selections
    }).rename_axis("strategy")
    aggregated.update({
        "selections": selections, "selection_counts": frequencies,
        "selection_frequencies": frequencies / len(results),
    })
    return aggregated


def run_benchmark(prices, progress=True, fit_model=None, name_model=None):
    """Run all eligible annual windows without reusing any fitted model."""
    windows = available_windows(prices)
    results = {}
    for number, (name, bounds) in enumerate(windows.iterrows(), 1):
        if progress:
            print(f"Window {number}/{len(windows)}: {name}", flush=True)
        results[name] = run_window(prices, bounds, fit_model, name_model)
    return {"windows": windows, "results": results, **aggregate_results(results)}


def plot_window(result):
    """Plot validation/test regimes and separate test portfolio equity curves."""
    bounds = result["bounds"]
    for split in ("validation", "test"):
        benchmark.plot_regime_shading(
            result["prices"], result[f"{split}_labels"],
            f"{split.title()} regimes: {bounds[f'{split}_start']} — "
            f"{bounds[f'{split}_end']}",
        )
    plot_equity_curves(
        {name: result["test_pnl"][name] for name in result["test_pnl"]},
        title=f"Separate test portfolios: {bounds['test_start']} — "
        f"{bounds['test_end']} ({matrix.COST_BPS:g} bps turnover costs)",
    )


def plot_aggregate(report):
    """Plot descriptive window Sharpes, mean/std, and selection frequencies."""
    sharpes = report["summary_per_window"]["sharpe"].unstack("strategy")
    sharpes.plot(figsize=(12, 5), title="Overlapping test-window Sharpes")
    plt.gca().set(xlabel="Training start year", ylabel="Annualized Sharpe")
    plt.tight_layout()
    stats = report["summary"]["sharpe"]
    stats["mean"].plot.barh(
        xerr=stats["std"], figsize=(10, 5),
        title="Equal-window mean Sharpe ± sample std (not standard errors)",
    )
    plt.tight_layout()
    report["selection_frequencies"].plot.bar(
        figsize=(12, 5), title="Validation selection frequencies across windows",
    )
    plt.tight_layout()


def main(show=True, list_windows=False):
    """Print all per-window tables and aggregate metrics, optionally plotting."""
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    prices = load_prices()
    windows = available_windows(prices)
    print(f"{len(windows)} eligible windows\n{windows.to_string()}")
    if list_windows:
        return windows
    report = run_benchmark(prices)
    for name, result in report["results"].items():
        print(f"\nWINDOW {name}: {result['bounds']}")
        for key in ("moments", "transitions", "fit_diagnostics", "train_accuracy",
                    "validation_counts",
                    "validation_table", "rotation", "best_single", "test_counts",
                    "test_table", "summary"):
            value = result[key]
            print(f"\n{key}\n{value.to_string() if hasattr(value, 'to_string') else value}")
    for key in ("summary", "validation_table", "test_table", "selection_counts",
                "selection_frequencies"):
        print(f"\n{key}\n{report[key].to_string()}")
    print("Overlapping tests are NOT independent; means are NOT a pooled portfolio.")
    if show:
        plot_aggregate(report)
        plt.show()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--list-windows", action="store_true")
    args, _ = parser.parse_known_args()
    main(show=not args.no_show, list_windows=args.list_windows)
