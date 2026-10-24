"""Resumable expanding-window HMM experiment with checkpointed daily decisions."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from .paper_regime_experiment import build_observations, fit_one, label_bull
except ImportError:
    from paper_regime_experiment import build_observations, fit_one, label_bull


MODELS = ("Gaussian", "GMM", "Student-t")
WINDOWS = (5, 10, 60)
HALF_LIVES = (5, 10, 60)
VERSION = 2


@dataclass(frozen=True)
class ModelSpec:
    """One emission model, its GMM component count, and its PCA setting."""

    key: str
    emission: str
    gmm_components: int
    use_pca: bool


MODEL_SPECS = (
    ModelSpec("gaussian_pca", "Gaussian", 1, True),
    ModelSpec("gaussian_no_pca", "Gaussian", 1, False),
    ModelSpec("gmm2_pca", "GMM", 2, True),
    ModelSpec("gmm2_no_pca", "GMM", 2, False),
    ModelSpec("gmm3_pca", "GMM", 3, True),
    ModelSpec("gmm3_no_pca", "GMM", 3, False),
    ModelSpec("gmm4_pca", "GMM", 4, True),
    ModelSpec("gmm4_no_pca", "GMM", 4, False),
    ModelSpec("gmm5_pca", "GMM", 5, True),
    ModelSpec("gmm5_no_pca", "GMM", 5, False),
    ModelSpec("student_t_pca", "Student-t", 1, True),
    ModelSpec("student_t_no_pca", "Student-t", 1, False),
)
MODEL_KEYS = tuple(spec.key for spec in MODEL_SPECS)
ORIGINAL_SIX = ("gaussian_pca", "gaussian_no_pca", "gmm2_pca", "gmm2_no_pca",
                "student_t_pca", "student_t_no_pca")
DEFAULT_MODELS = ORIGINAL_SIX
_SPECS_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def parse_specs(keys):
    """Resolve requested model keys into unique model specifications.

    Args:
        keys: Keys such as ``("gmm3_pca", "gmm3_no_pca")``.

    Returns:
        Tuple of ``ModelSpec`` objects in the requested order.

    Raises:
        ValueError: A key is unknown, repeated, or the list is empty.
    """
    keys = tuple(keys)
    if not keys:
        raise ValueError("Select at least one model")
    for key in keys:
        if key not in _SPECS_BY_KEY:
            raise ValueError(f"Unknown model {key!r}; choose from {MODEL_KEYS}")
    if len(set(keys)) != len(keys):
        raise ValueError("Model keys must be unique")
    return tuple(_SPECS_BY_KEY[key] for key in keys)


def _atomic_bytes(path, payload):
    """Replace one output atomically without leaving a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.resolve().parent != path.resolve().parent:
        raise ValueError("Temporary and final output must share a directory")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fingerprint(prices, start, years, seed, max_iter, update_iter, refresh_days,
                 specs):
    """Bind a checkpoint to its data and immutable experiment settings."""
    digest = hashlib.sha256()
    digest.update(np.asarray(prices.index.asi8, dtype="<i8").tobytes())
    digest.update(np.asarray(prices.to_numpy(), dtype="<f8").tobytes())
    settings = dict(version=VERSION, start=start, years=years, seed=seed,
                    max_iter=max_iter, update_iter=update_iter,
                    refresh_days=refresh_days, windows=WINDOWS,
                    half_lives=HALF_LIVES, models=[spec.key for spec in specs])
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def _prepare(features, dates, scaler, pca):
    """Transform historical features using the last refresh's training map."""
    X = scaler.transform(features.loc[dates])
    return pca.transform(X) if pca is not None else X


def _refresh(features, returns, dates, seed, max_iter, specs):
    """Fit every requested branch on one preprocessing map per PCA setting."""
    window = features.loc[dates]
    scaler = StandardScaler().fit(window)
    standardized = scaler.transform(window)
    preprocessors = {
        use_pca: (PCA(n_components=0.95, svd_solver="full").fit(standardized)
                  if use_pca else None)
        for use_pca in (True, False)
    }
    branches = {}
    for spec in specs:
        pca = preprocessors[spec.use_pca]
        X = pca.transform(standardized) if pca is not None else standardized
        model, _ = fit_one(X, returns.loc[dates].to_numpy(), spec.emission,
                           seed=seed, max_iter=max_iter,
                           gmm_components=spec.gmm_components)
        branches[spec.key] = (model, scaler, pca)
    return branches


def run(prices, output_dir, start, end, years=10, seed=0, max_iter=100,
        update_iter=3, refresh_days=63, command=None, models=DEFAULT_MODELS):
    """Run and checkpoint daily decisions, resuming after the last complete day.

    The first date fits on its trailing calendar years. Later dates retain
    that history and append one observation. Every refresh_days decisions,
    models and preprocessing are rebuilt on the expanded history; otherwise
    each model receives up to update_iter EM steps on all history. The
    checkpoint is trusted pickle: only load files generated by this script.
    If supplied by the CLI, command records the invocation after validation.
    """
    counts = (years, max_iter, update_iter, refresh_days)
    if any(not isinstance(value, int) or value < 1 for value in counts):
        raise ValueError("Years and iteration/refresh counts must be positive integers")
    specs = parse_specs(models)
    n_branches = len(specs)
    output_dir = Path(output_dir)
    prices = prices.astype(float).sort_index()
    if not prices.index.is_unique or not np.isfinite(prices).all():
        raise ValueError("Prices must be finite with unique dates")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end < start or end >= prices.index[-1]:
        raise ValueError("End must follow start and precede the last price date")
    features = build_observations(prices, windows=WINDOWS,
                                  half_lives=HALF_LIVES).dropna()
    returns = prices.pct_change(fill_method=None)
    initial_dates = features.loc[(features.index > start - pd.DateOffset(years=years))
                                 & (features.index <= start)].index
    if len(initial_dates) < years * 200 or returns.reindex(initial_dates).isna().any():
        raise ValueError("Insufficient complete initial training observations")
    decisions = features.loc[start:end].index
    if len(decisions) == 0 or decisions[0] != start:
        raise ValueError("Start must be a trading day with complete features")
    fingerprint = _fingerprint(prices, start.isoformat(), years, seed,
                               max_iter, update_iter, refresh_days, specs)
    checkpoint_path = output_dir / "checkpoint.pkl"
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            state = pickle.load(handle)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint data or settings differ; use a new output folder")
        if state["last_date"] not in decisions:
            raise ValueError("Checkpoint date is outside requested evaluation range")
        completed = decisions.get_loc(state["last_date"]) + 1
        if len(state["rows"]) != completed * n_branches:
            raise ValueError("Checkpoint has incomplete or duplicated decisions")
        print(f"Resuming after {state['last_date'].date()} ({completed}/{len(decisions)})",
              flush=True)
    else:
        state = dict(fingerprint=fingerprint, last_date=None, rows=[], branches=None)
        completed = 0
    if command is not None:
        _atomic_bytes(output_dir / "command.txt", command.encode("utf-8"))
    started = time.monotonic()
    for offset, date in enumerate(decisions[completed:], start=completed):
        if offset == 0:
            history = initial_dates
        else:
            history = features.loc[initial_dates[0]:date].index
        is_refresh = offset % refresh_days == 0
        if is_refresh:
            print(f"Refitting {date.date()}: {n_branches} models, "
                  f"{len(history)} training days", flush=True)
            refresh_started = time.monotonic()
            state["branches"] = _refresh(features, returns, history, seed,
                                         max_iter, specs)
            iterations = ", ".join(
                f"{key}:{model.trainer.n_iter}"
                for key, (model, _, _) in state["branches"].items()
            )
            print(f"Refit done in {time.monotonic() - refresh_started:.1f}s "
                  f"(EM iterations: {iterations})", flush=True)
        rows = []
        for spec in specs:
            model, scaler, pca = state["branches"][spec.key]
            X = _prepare(features, history, scaler, pca)
            if not is_refresh:
                model.continue_fit(X, max_iter=update_iter,
                                   tol=len(X) * 1e-5)
            bull = label_bull(model, X, returns.loc[history].to_numpy())
            _, filtered, _ = model.filter(X)
            next_prior = filtered[-1] @ model.states.transitions
            rows.append(dict(date=date, model=spec.emission, pca=spec.use_pca,
                             spec=spec.key,
                             bull_probability=float(next_prior[bull]),
                             bull_raw=int(next_prior[bull] >= 0.5),
                             log_likelihood=float(model.history[-1]),
                             n_components=X.shape[1], converged=model.trainer.converged,
                             iterations=model.trainer.n_iter,
                             gmm_components=(spec.gmm_components
                                             if spec.emission == "GMM" else 0),
                             selected_seed=seed, refresh=is_refresh,
                             training_days=len(history)))
        state["rows"].extend(rows)
        state["last_date"] = date
        _atomic_bytes(checkpoint_path, pickle.dumps(state,
                                                    protocol=pickle.HIGHEST_PROTOCOL))
        # The CSV is a regenerable view; the atomic pickle is authoritative.
        table = pd.DataFrame(state["rows"])
        _atomic_bytes(output_dir / "signals.csv", table.to_csv(index=False).encode())
        if is_refresh or offset == completed or offset == len(decisions) - 1:
            elapsed = time.monotonic() - started
            done = offset - completed + 1
            eta = elapsed / done * (len(decisions) - offset - 1)
            print(f"{offset + 1}/{len(decisions)} {date.date()} "
                  f"{'REFRESH' if is_refresh else 'update'} "
                  f"elapsed={elapsed:.1f}s ETA={eta / 3600:.1f}h", flush=True)
    return pd.DataFrame(state["rows"])


def main():
    """Run the command-line experiment from a repository checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2004-01-02")
    parser.add_argument("--end", default="2025-12-30")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--update-iter", type=int, default=3)
    parser.add_argument("--refresh-days", type=int, default=63)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        choices=list(MODEL_KEYS), metavar="MODEL",
                        help="space-separated branches to run; options: "
                             + " ".join(MODEL_KEYS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "data/processed/sp500_index_price.csv"
    prices = pd.read_csv(source, parse_dates=["YYYYMMDD"])
    prices = prices.set_index("YYYYMMDD")["DlyPrcInd"]
    # Reconstruct the invocation; original shell whitespace is unavailable.
    prefix = (f"PYTHONPATH={shlex.quote(os.environ['PYTHONPATH'])} "
              if "PYTHONPATH" in os.environ else "")
    command = prefix + shlex.join([sys.executable, *sys.argv]) + "\n"
    run(prices, args.output, args.start, args.end, years=args.years,
        seed=args.seed, max_iter=args.max_iter, update_iter=args.update_iter,
        refresh_days=args.refresh_days, models=args.models, command=command)


if __name__ == "__main__":
    main()
