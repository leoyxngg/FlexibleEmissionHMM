"""Annual 20/3/1/3 sliding-window S&P 500 benchmark with GMM-HMM emissions."""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM

NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR))

import sharpe_benchmark_sliding_window as sliding  # noqa: E402

matrix = sliding.matrix

N_MIX = 3
"""Gaussian components per hidden state, matching the synthetic GMM study."""


## Emission fitting and regime naming


def fit_gmm_hmm(features, n_states=matrix.N_STATES, n_mix=N_MIX):
    """Fit restarts of a full-covariance GMM-HMM and keep the best likelihood.

    Use the same restarts, iteration cap, tolerance, and covariance floor as
    the Gaussian benchmark so the emission is the only difference.

    Args:
        features: Training-only matrix in return, volatility, mean order.
        n_states: Number of hidden states.
        n_mix: Gaussian components per state.

    Returns:
        The highest-likelihood finite fitted model.

    Raises:
        RuntimeError: Every restart failed.
    """
    best, best_score = None, -np.inf
    for seed in range(matrix.N_RESTARTS):
        model = GMMHMM(
            n_components=n_states, n_mix=n_mix, covariance_type="full",
            min_covar=matrix.MIN_COVAR, n_iter=matrix.MAX_ITER,
            tol=matrix.TOL, random_state=seed,
        )
        try:
            model.fit(features)
            score = model.score(features)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(score) and score > best_score:
            best, best_score = model, score
    if best is None:
        raise RuntimeError(f"All {matrix.N_RESTARTS} restarts failed")
    return best


def state_moments(model):
    """Collapse each state's mixture to its overall mean and spread.

    Args:
        model: Fitted ``GMMHMM`` with a return, volatility, mean feature order.

    Returns:
        Per-state return mean and standard deviation, mean volatility, and
        largest component weight, all in percent per day except the weight.
    """
    weights, means = model.weights_, model.means_
    covars = model.covars_[..., 0, 0]
    mean_r = (weights * means[..., 0]).sum(axis=1)
    variance_r = (weights * (covars + means[..., 0] ** 2)).sum(axis=1) - mean_r**2
    mean_sigma = (weights * means[..., 1]).sum(axis=1)
    return pd.DataFrame({
        "mean_r_%day": 100 * mean_r,
        "sd_r_%day": 100 * np.sqrt(variance_r),
        "mean_sigma_%day": 100 * mean_sigma,
        "max_component_weight": weights.max(axis=1),
    })


def name_gmm_regimes(model):
    """Name states with the Gaussian benchmark's rule on mixture moments.

    The highest-volatility state is bear; the highest-return remaining state is
    bull; the leftover state is sideways. These are heuristic names only.

    Args:
        model: Fitted three-state ``GMMHMM``.

    Returns:
        State-to-name map and the moment table with a ``regime`` column.
    """
    if model.n_components != 3:
        raise ValueError("Regime naming requires exactly three states")
    fitted = state_moments(model)
    bear = int(fitted["sd_r_%day"].idxmax())
    bull = int(fitted.drop(index=bear)["mean_r_%day"].idxmax())
    sideways = next(state for state in fitted.index if state not in (bear, bull))
    names = {bear: "bear", sideways: "sideways", bull: "bull"}
    fitted["regime"] = pd.Series(names)
    return names, fitted


## Benchmark


def run_benchmark(prices, progress=True):
    """Run every eligible annual window with GMM-HMM regime emissions."""
    return sliding.run_benchmark(
        prices, progress, fit_model=fit_gmm_hmm, name_model=name_gmm_regimes,
    )


def main(show=True, list_windows=False):
    """Print all per-window tables and aggregate metrics, optionally plotting."""
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    prices = sliding.load_prices()
    windows = sliding.available_windows(prices)
    print(f"{len(windows)} eligible windows\n{windows.to_string()}")
    if list_windows:
        return windows
    report = run_benchmark(prices)
    for name, result in report["results"].items():
        print(f"\nWINDOW {name}: {result['bounds']}")
        for key in ("moments", "transitions", "fit_diagnostics", "train_accuracy",
                    "validation_counts", "validation_table", "rotation",
                    "best_single", "test_counts", "test_table", "summary"):
            value = result[key]
            print(f"\n{key}\n{value.to_string() if hasattr(value, 'to_string') else value}")
    for key in ("summary", "validation_table", "test_table", "selection_counts",
                "selection_frequencies"):
        print(f"\n{key}\n{report[key].to_string()}")
    print("Overlapping tests are NOT independent; means are NOT a pooled portfolio.")
    if show:
        sliding.plot_aggregate(report)
        plt.show()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--list-windows", action="store_true")
    args, _ = parser.parse_known_args()
    main(show=not args.no_show, list_windows=args.list_windows)
