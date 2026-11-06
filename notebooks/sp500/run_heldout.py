"""Resumable expanding-window held-out HMM predictive scoring.

Fit before each chronological block of ``test_days`` observations, keep the
parameters fixed, and score the block with one-day-ahead filtered predictive
log densities, starting from the next-step prior implied by the training history.
Scaler and 95% PCA are fitted on the training history only.

Engines (``--engine``):
    classic  Gaussian, GMM and Student-t HMMs trained by EM.
    flow     Spline-flow HMM, best of several seeds (needs the ``flows`` extra).
"""

import argparse
import hashlib
import json
import pickle
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_backtest import (  # noqa: E402
    DEFAULT_MODELS, FLOW_DEFAULTS, HALF_LIVES, MODEL_KEYS, WINDOWS, _atomic_bytes,
    _prepare, build_observations, parse_specs,
)
from backtest_experiment import fit_one, label_bull  # noqa: E402

VERSION = 2


class ClassicEngine:
    """Gaussian, GMM and Student-t HMMs trained by EM, refit each fold."""

    def __init__(self, models=DEFAULT_MODELS, seed=0, max_iter=100):
        self.specs = parse_specs(models)
        self.n_branches = len(self.specs)
        self.seed, self.max_iter = seed, max_iter

    def settings(self):
        """Return the engine settings that define a compatible checkpoint."""
        return dict(engine="classic", seed=self.seed, max_iter=self.max_iter,
                    models=[spec.key for spec in self.specs])

    def evaluate(self, features, returns, train_dates, test_dates, fold):
        """Fit every branch and return daily rows, fold rows, extras and models."""
        window = features.loc[train_dates]
        scaler = StandardScaler().fit(window)
        standardized = scaler.transform(window)
        pca = PCA(n_components=0.95, svd_solver="full").fit(standardized)
        daily, folds, branches = [], [], {}
        for spec in self.specs:
            use = pca if spec.use_pca else None
            training = pca.transform(standardized) if spec.use_pca else standardized
            model, _ = fit_one(training, returns.loc[train_dates].to_numpy(),
                               spec.emission, seed=self.seed,
                               max_iter=self.max_iter,
                               gmm_components=spec.gmm_components)
            branches[spec.key] = (model, scaler, use)
            test = _prepare(features, test_dates, scaler, use)
            _, filtered, train_scores = model.filter(training)
            prior = filtered[-1] @ model.states.transitions
            _, _, test_scores = model.filter(test, initial_prior=prior)
            meta = dict(fold=fold, spec=spec.key, model=spec.emission,
                        pca=spec.use_pca, gmm_components=spec.gmm_components,
                        train_end=train_dates[-1], training_days=len(training),
                        n_features=training.shape[1])
            folds.append(dict(
                **meta, test_start=test_dates[0], test_end=test_dates[-1],
                test_days=len(test), train_mean_log_score=float(train_scores.mean()),
                test_mean_log_score=float(test_scores.mean()),
                test_total_log_score=float(test_scores.sum()),
                converged=bool(model.trainer.converged),
                iterations=int(model.trainer.n_iter)))
            daily.extend(dict(**meta, date=d, log_predictive=float(s))
                         for d, s in zip(test_dates, test_scores))
            print(f"  {spec.key}: test mean log score {test_scores.mean():.6f}",
                  flush=True)
        return daily, folds, {}, branches


class FlowEngine:
    """Spline-flow HMM: best of several seeds by training likelihood per fold."""

    n_branches = 1
    spec = "flow_pca"

    def __init__(self, seeds=(0, 1, 2, 3, 4), max_iter=100, n_states=2,
                 device=None, **flow_options):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.seeds, self.max_iter, self.n_states = list(seeds), max_iter, n_states
        self.options = {**FLOW_DEFAULTS, **flow_options}

    def settings(self):
        """Return the engine settings that define a compatible checkpoint."""
        return dict(engine="flow", seeds=self.seeds, max_iter=self.max_iter,
                    n_states=self.n_states, device=self.device, flow=self.options)

    def _fit(self, X):
        """Fit one FlowHMM per seed; return the best model, record and attempts."""
        from hmmlearn_flow_hmm import FlowHMM
        best, attempts = None, []
        for seed in self.seeds:
            model = FlowHMM(
                n_components=self.n_states, device=self.device,
                dtype="float32" if self.device == "mps" else "float64",
                random_state=int(seed), n_iter=self.max_iter, tol=len(X) * 1e-6,
                convergence_patience=5, **self.options)
            record = dict(seed=int(seed), train_log_likelihood=np.nan,
                          converged=False, iterations=0, error=None)
            try:
                model.fit(X)
                record.update(train_log_likelihood=float(model.score(X)),
                              converged=bool(model.converged_),
                              iterations=int(model.monitor_.iter))
            except (ValueError, RuntimeError, FloatingPointError,
                    np.linalg.LinAlgError) as error:
                record["error"] = repr(error)
            attempts.append(record)
            if np.isfinite(record["train_log_likelihood"]) and (
                    best is None or record["train_log_likelihood"] > best[0]):
                best = (record["train_log_likelihood"], model, record)
        if best is None:
            raise RuntimeError(f"All flow fits failed: {attempts}")
        return best[1], best[2], attempts

    def evaluate(self, features, returns, train_dates, test_dates, fold):
        """Fit the best seed and return daily rows, fold rows and extra tables."""
        began = time.time()
        window = features.loc[train_dates]
        scaler = StandardScaler().fit(window)
        pca = PCA(n_components=0.95, svd_solver="full").fit(
            scaler.transform(window))
        training = pca.transform(scaler.transform(window))
        test = pca.transform(scaler.transform(features.loc[test_dates]))
        model, chosen, attempts = self._fit(training)
        _, filtered, train_scores = model.filter(training)
        prior = filtered[-1] @ model.transmat_
        _, _, test_scores = model.filter(test, initial_prior=prior)
        meta = dict(fold=fold, spec=self.spec, model="Flow", pca=True,
                    train_end=train_dates[-1], training_days=len(training),
                    n_features=training.shape[1])
        folds = [dict(
            **meta, test_start=test_dates[0], test_end=test_dates[-1],
            test_days=len(test), train_mean_log_score=float(train_scores.mean()),
            test_mean_log_score=float(test_scores.mean()),
            test_total_log_score=float(test_scores.sum()),
            converged=chosen["converged"], iterations=chosen["iterations"],
            selected_seed=chosen["seed"], seconds=time.time() - began)]
        daily = [dict(**meta, date=d, log_predictive=float(s))
                 for d, s in zip(test_dates, test_scores)]
        mass = filtered.sum(axis=0)
        r = returns.loc[train_dates].to_numpy()
        mean = filtered.T @ r / mass
        variance = filtered.T @ (r ** 2) / mass - mean ** 2
        regimes = [dict(fold=fold, state=k, occupancy=float(mass[k] / mass.sum()),
                        mean_return=float(mean[k]),
                        volatility=float(np.sqrt(max(variance[k], 0.0))),
                        persistence=float(model.transmat_[k, k]))
                   for k in range(len(mass))]
        extras = dict(regimes=regimes,
                      seed_attempts=[dict(fold=fold, **row) for row in attempts])
        print(f"  {self.spec}: {training.shape[1]} PCs, test mean log score "
              f"{test_scores.mean():.6f}, seed {chosen['seed']}", flush=True)
        return daily, folds, extras, None


def run(prices, output_dir, start, end, years=10, seed=0, max_iter=100,
        test_days=63, models=DEFAULT_MODELS, command=None, engine=None):
    """Fit before each chronological block and save fixed-parameter test scores.

    Args:
        prices: Positive price series indexed by trading date.
        output_dir: Dedicated result folder; resume only trusted checkpoints.
        start: First held-out observation date, excluded from initial training.
        end: Last held-out date, included in the experiment fingerprint.
        years: Initial trailing calendar years; history then expands.
        seed: Fitting seed for the default classic engine.
        max_iter: Maximum EM iterations for the default classic engine.
        test_days: Number of observations per nonoverlapping test block.
        models: Branch keys for the default classic engine.
        command: Optional command-line provenance.
        engine: Optional engine; defaults to ``ClassicEngine``.

    Returns:
        Daily predictive scores, per-block scores and summary tables.

    Raises:
        ValueError: Data, dates, settings, or resumed checkpoint are incompatible.
    """
    if engine is None:
        engine = ClassicEngine(models, seed, max_iter)
    if any(type(v) is not int or v < 1 for v in (years, test_days)):
        raise ValueError("years and test_days must be positive integers")
    prices = prices.astype(float).sort_index()
    if (not isinstance(prices.index, pd.DatetimeIndex)
            or not prices.index.is_unique or prices.index.hasnans
            or not np.isfinite(prices).all() or (prices <= 0).any()):
        raise ValueError("Prices must be positive and finite with unique dates")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end < start or end > prices.index[-1]:
        raise ValueError("Invalid held-out date range")
    features = build_observations(
        prices, windows=WINDOWS, half_lives=HALF_LIVES).dropna()
    initial = features.loc[(features.index >= start - pd.DateOffset(years=years))
                           & (features.index < start)].index
    dates = features.loc[start:end].index
    if len(initial) < years * 200 or len(dates) == 0 or dates[0] != start:
        raise ValueError("Insufficient training data or invalid test start")
    returns = prices.pct_change(fill_method=None)
    settings = dict(version=VERSION, start=str(start), end=str(end), years=years,
                    test_days=test_days, windows=WINDOWS, half_lives=HALF_LIVES,
                    **engine.settings())
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(np.asarray(prices.index.asi8, dtype="<i8").tobytes())
    digest.update(np.asarray(prices, dtype="<f8").tobytes())
    fingerprint = digest.hexdigest()
    output = Path(output_dir)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:  # Trusted: written by this script.
            state = pickle.load(handle)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint settings/data differ; use a new folder")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use an empty output folder or a compatible checkpoint")
        state = dict(fingerprint=fingerprint, completed=0, daily=[], folds=[],
                     extras={})
    completed = state["completed"]
    if (not 0 <= completed <= len(dates)
            or (completed < len(dates) and completed % test_days)
            or len(state["daily"]) != completed * engine.n_branches):
        raise ValueError("Incomplete checkpoint")
    if command is not None:
        _atomic_bytes(output / "command.txt", command.encode())
    _atomic_bytes(output / "settings.json", json.dumps(settings, indent=2).encode())

    for offset in range(completed, len(dates), test_days):
        test_dates = dates[offset:offset + test_days]
        train_dates = features.loc[(features.index >= initial[0])
                                   & (features.index < test_dates[0])].index
        print(f"Fit before {test_dates[0].date()}: {len(train_dates)} train, "
              f"{len(test_dates)} test observations", flush=True)
        daily, folds, extras, branches = engine.evaluate(
            features, returns, train_dates, test_dates, offset // test_days)
        state.update(completed=offset + len(test_dates))
        if branches is not None:
            state["branches"] = branches
        state["daily"].extend(daily)
        state["folds"].extend(folds)
        for name, rows in extras.items():
            state["extras"].setdefault(name, []).extend(rows)
        _atomic_bytes(checkpoint, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        _save_tables(output, state)
    # Regenerate CSV views even when resuming an already completed experiment.
    return _save_tables(output, state)


def _save_tables(output, state):
    """Write the CSV views of a checkpoint state and return the main tables."""
    daily, folds = pd.DataFrame(state["daily"]), pd.DataFrame(state["folds"])
    summary = daily.groupby(["spec", "pca"])["log_predictive"].agg(
        observations="size", mean_log_score="mean", total_log_score="sum"
    ).reset_index()
    tables = [("daily_scores", daily), ("fold_scores", folds), ("summary", summary)]
    tables += [(name, pd.DataFrame(rows)) for name, rows in state["extras"].items()]
    for name, frame in tables:
        _atomic_bytes(output / f"{name}.csv", frame.to_csv(index=False).encode())
    return daily, folds, summary


def main():
    """Run chronological held-out scoring on the existing SPX price data."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", choices=["classic", "flow"], default="classic")
    parser.add_argument("--start", default="2004-01-02")
    parser.add_argument("--end", default="2025-12-30")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="classic only")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="flow only")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS,
                        default=list(DEFAULT_MODELS), help="classic only")
    parser.add_argument("--states", type=int, default=2, help="flow only")
    parser.add_argument("--device", default=None, help="flow only")
    parser.add_argument("--base", choices=["normal", "student_t"],
                        default=FLOW_DEFAULTS["base_distribution"], help="flow only")
    parser.add_argument("--count-bins", type=int,
                        default=FLOW_DEFAULTS["count_bins"], help="flow only")
    parser.add_argument("--num-layers", type=int,
                        default=FLOW_DEFAULTS["num_layers"], help="flow only")
    args = parser.parse_args()
    if args.engine == "flow":
        engine = FlowEngine(args.seeds, args.max_iter, args.states, args.device,
                            base_distribution=args.base,
                            count_bins=args.count_bins, num_layers=args.num_layers)
    else:
        engine = ClassicEngine(args.models, args.seed, args.max_iter)
    source = ROOT / "data/processed/sp500_index_price.csv"
    prices = pd.read_csv(source, parse_dates=["YYYYMMDD"]).set_index("YYYYMMDD")
    summary = run(prices["DlyPrcInd"], args.output, args.start, args.end,
                  years=args.years, test_days=args.test_days, engine=engine,
                  command=shlex.join([sys.executable, *sys.argv]) + "\n")[2]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
