"""Resumable expanding-window backtest with checkpointed daily regime decisions.

The first decision fits on the trailing ``years`` calendar years of price-only
features and the history then expands one day at a time. Every ``refresh_days``
decisions the scaler, 95% PCA and models are rebuilt on all history; on other days
each model takes a few warm-start steps on all history. Each decision is the
predicted next-day probability of the bull state.

Engines (``--engine``):
    classic  Gaussian, GMM and Student-t HMMs trained by EM.
    flow     Spline-flow HMM (needs the ``flows`` extra).
    mixed    One Gaussian and one Student-t state trained by gradient descent.
"""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_experiment import build_observations, fit_one, label_bull  # noqa: E402

MODELS = ("Gaussian", "GMM", "Student-t", "Mixed t-Gaussian")
WINDOWS = (5, 10, 60)
HALF_LIVES = (5, 10, 60)
VERSION = 3


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
    # PCA by default. State 0 Gaussian, state 1 Student-t; per-state EM M-step.
    ModelSpec("mixed_t_gaussian", "Mixed t-Gaussian", 1, True),
)
MODEL_KEYS = tuple(spec.key for spec in MODEL_SPECS)
ORIGINAL_SIX = ("gaussian_pca", "gaussian_no_pca", "gmm2_pca", "gmm2_no_pca",
                "student_t_pca", "student_t_no_pca")
DEFAULT_MODELS = ORIGINAL_SIX
_SPECS_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}

FLOW_DEFAULTS = dict(
    base_distribution="student_t", base_df=5.0, bound=5.0, count_bins=8,
    num_layers=1, m_step_steps=5, batch_size=2048, learning_rate=1e-3,
)
MIXED_LEARNING_RATE = 0.02


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


def _prepare(features, dates, scaler, pca):
    """Transform historical features using the last refresh's training map."""
    X = scaler.transform(features.loc[dates])
    return pca.transform(X) if pca is not None else X


def _fit_preprocessing(features, history):
    """Fit the scaler and 95% PCA on history; return both and the PCA scores."""
    window = features.loc[history]
    scaler = StandardScaler().fit(window)
    standardized = scaler.transform(window)
    pca = PCA(n_components=0.95, svd_solver="full").fit(standardized)
    return scaler, pca, standardized


def bull_state(filtered, returns):
    """Return the state with the larger posterior-weighted training return."""
    mass = filtered.sum(axis=0)
    return int(np.argmax((filtered.T @ np.asarray(returns)) / mass))


class ClassicEngine:
    """Gaussian, GMM and Student-t HMMs trained by Baum-Welch EM."""

    def __init__(self, models=DEFAULT_MODELS, seed=0, max_iter=100, update_iter=3):
        self.specs = parse_specs(models)
        self.n_branches = len(self.specs)
        self.seed, self.max_iter, self.update_iter = seed, max_iter, update_iter

    def settings(self):
        """Return the engine settings that define a compatible checkpoint."""
        return dict(engine="classic", seed=self.seed, max_iter=self.max_iter,
                    update_iter=self.update_iter,
                    models=[spec.key for spec in self.specs])

    def refresh(self, features, returns, history):
        """Fit every requested branch on one preprocessing map per PCA setting."""
        window = features.loc[history]
        scaler = StandardScaler().fit(window)
        standardized = scaler.transform(window)
        preprocessors = {
            use_pca: (PCA(n_components=0.95, svd_solver="full").fit(standardized)
                      if use_pca else None)
            for use_pca in (True, False)
        }
        branches = {}
        for spec in self.specs:
            pca = preprocessors[spec.use_pca]
            X = pca.transform(standardized) if pca is not None else standardized
            model, _ = fit_one(X, returns.loc[history].to_numpy(), spec.emission,
                               seed=self.seed, max_iter=self.max_iter,
                               gmm_components=spec.gmm_components)
            branches[spec.key] = (model, scaler, pca)
        return branches

    def describe(self, branches):
        """Return a short log string of per-branch EM iteration counts."""
        return ", ".join(f"{key}:{model.trainer.n_iter}"
                         for key, (model, _, _) in branches.items())

    def decide(self, branches, features, returns, history, date, is_refresh):
        """Return one signal row per branch for ``date``."""
        rows = []
        for spec in self.specs:
            model, scaler, pca = branches[spec.key]
            X = _prepare(features, history, scaler, pca)
            if not is_refresh:
                model.continue_fit(X, max_iter=self.update_iter,
                                   tol=len(X) * 1e-5)
            bull = label_bull(model, X, returns.loc[history].to_numpy())
            _, filtered, _ = model.filter(X)
            next_prior = filtered[-1] @ model.states.transitions
            rows.append(dict(date=date, model=spec.emission, pca=spec.use_pca,
                             spec=spec.key,
                             bull_probability=float(next_prior[bull]),
                             bull_raw=int(next_prior[bull] >= 0.5),
                             log_likelihood=float(model.history[-1]),
                             n_components=X.shape[1],
                             converged=model.trainer.converged,
                             iterations=model.trainer.n_iter,
                             gmm_components=(spec.gmm_components
                                             if spec.emission == "GMM" else 0),
                             selected_seed=self.seed, refresh=is_refresh,
                             training_days=len(history)))
        return rows


def _flow_names(options):
    """Return the spec key and display label for a flow base distribution."""
    if options["base_distribution"] == "normal":
        return "flow_normal", "Flow (normal)"
    df = options["base_df"]
    if df == FLOW_DEFAULTS["base_df"]:
        # Keep the original names so earlier t5 results stay comparable.
        return "flow_pca", "Flow"
    return f"flow_t{df:g}", f"Flow (t, df={df:g})"


class FlowEngine:
    """Spline-flow HMM trained by generalized EM (``hmmlearn_flow_hmm``)."""

    n_branches = 1

    def __init__(self, seed=0, max_iter=100, update_iter=3, n_states=2,
                 device=None, **flow_options):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.seed, self.max_iter, self.update_iter = seed, max_iter, update_iter
        self.n_states = n_states
        self.options = {**FLOW_DEFAULTS, **flow_options}
        self.spec, self.label = _flow_names(self.options)

    def settings(self):
        """Return the engine settings that define a compatible checkpoint."""
        return dict(engine="flow", seed=self.seed, max_iter=self.max_iter,
                    update_iter=self.update_iter, n_states=self.n_states,
                    device=self.device, flow=self.options)

    def refresh(self, features, returns, history):
        """Fit scaler, PCA and a fresh FlowHMM on all history."""
        from hmmlearn_flow_hmm import FlowHMM
        scaler, pca, standardized = _fit_preprocessing(features, history)
        X = pca.transform(standardized)
        model = FlowHMM(
            n_components=self.n_states, device=self.device,
            dtype="float32" if self.device == "mps" else "float64",
            random_state=int(self.seed), n_iter=self.max_iter,
            tol=len(X) * 1e-5, convergence_patience=5, **self.options,
        ).fit(X)
        return {self.spec: (model, scaler, pca)}

    def describe(self, branches):
        """Return a short log string of the EM iteration count."""
        return f"iters={branches[self.spec][0].monitor_.iter}"

    def decide(self, branches, features, returns, history, date, is_refresh):
        """Return the flow signal row for ``date``."""
        model, scaler, pca = branches[self.spec]
        X = pca.transform(scaler.transform(features.loc[history]))
        if not is_refresh:
            model.set_params(init_params="", n_iter=self.update_iter,
                             tol=len(X) * 1e-5)
            model.fit(X)
        _, filtered, _ = model.filter(X)
        bull = bull_state(filtered, returns.loc[history].to_numpy())
        next_prior = filtered[-1] @ model.transmat_
        return [dict(
            date=date, model=self.label, pca=True, spec=self.spec,
            bull_probability=float(next_prior[bull]),
            bull_raw=int(next_prior[bull] >= 0.5),
            log_likelihood=float(model.history_[-1]), n_components=X.shape[1],
            converged=bool(model.converged_), iterations=int(model.monitor_.iter),
            gmm_components=0, selected_seed=self.seed, refresh=is_refresh,
            training_days=len(history))]


class MixedEngine:
    """One emission family per state, trained by gradient descent (torch)."""

    n_branches = 1

    def __init__(self, emissions=("gaussian", "student_t"), seed=0,
                 max_iter=400, update_iter=3):
        from flexible_emission_hmm.torch_hmm import (
            GaussianStateEmission, StudentTStateEmission)
        self.families = {"gaussian": GaussianStateEmission,
                         "student_t": StudentTStateEmission}
        self.emissions = tuple(emissions)
        if any(name not in self.families for name in self.emissions):
            raise ValueError(f"Unknown emission; choose from {sorted(self.families)}")
        self.seed, self.max_iter, self.update_iter = seed, max_iter, update_iter
        self.spec = "_".join(self.emissions) + "_pca"

    def settings(self):
        """Return the engine settings that define a compatible checkpoint."""
        return dict(engine="mixed", seed=self.seed, max_iter=self.max_iter,
                    update_iter=self.update_iter, emissions=list(self.emissions),
                    lr=MIXED_LEARNING_RATE)

    def refresh(self, features, returns, history):
        """Fit scaler, PCA and a fresh EmissionHMM on all history."""
        import torch
        from flexible_emission_hmm.torch_hmm import EmissionHMM
        torch.set_num_threads(min(4, torch.get_num_threads()))
        scaler, pca, standardized = _fit_preprocessing(features, history)
        X = pca.transform(standardized)
        torch.manual_seed(self.seed)
        model = EmissionHMM([self.families[name](X.shape[1])
                             for name in self.emissions], seed=self.seed)
        model.init_from_data(X)
        model.fit(X, self.max_iter, lr=MIXED_LEARNING_RATE)
        return {self.spec: (model, scaler, pca)}

    def describe(self, branches):
        """Return a short log string for a refit."""
        return f"steps={self.max_iter}"

    def decide(self, branches, features, returns, history, date, is_refresh):
        """Return the mixed-model signal row for ``date``."""
        model, scaler, pca = branches[self.spec]
        X = pca.transform(scaler.transform(features.loc[history]))
        last_ll = None
        if not is_refresh:
            last_ll = model.fit(X, self.update_iter, lr=MIXED_LEARNING_RATE)[-1]
        scores, filtered, next_prior = model.filter(X)
        bull = bull_state(filtered, returns.loc[history].to_numpy())
        return [dict(
            date=date, model="Mixed", pca=True, spec=self.spec,
            bull_probability=float(next_prior[bull]),
            bull_raw=int(next_prior[bull] >= 0.5),
            log_likelihood=float(scores.sum() if last_ll is None else last_ll),
            n_components=X.shape[1], converged=True,
            iterations=self.max_iter if is_refresh else self.update_iter,
            gmm_components=0, selected_seed=self.seed, refresh=is_refresh,
            training_days=len(history))]


def _fingerprint(prices, start, years, refresh_days, engine):
    """Bind a checkpoint to its data and immutable experiment settings."""
    digest = hashlib.sha256()
    digest.update(np.asarray(prices.index.asi8, dtype="<i8").tobytes())
    digest.update(np.asarray(prices.to_numpy(), dtype="<f8").tobytes())
    settings = dict(version=VERSION, start=start, years=years,
                    refresh_days=refresh_days, windows=WINDOWS,
                    half_lives=HALF_LIVES, **engine.settings())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest(), settings


def _schedule(prices, start, end, years):
    """Validate the inputs; return features, returns and the decision dates."""
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
    return prices, start, features, returns, initial_dates, decisions


def run(prices, output_dir, start, end, years=10, seed=0, max_iter=100,
        update_iter=3, refresh_days=63, command=None, models=DEFAULT_MODELS,
        engine=None):
    """Run and checkpoint daily decisions, resuming after the last complete day.

    The first date fits on its trailing calendar years. Later dates retain
    that history and append one observation. Every refresh_days decisions,
    models and preprocessing are rebuilt on the expanded history; otherwise
    each model takes ``update_iter`` warm-start steps on all history. Without
    ``engine``, a ``ClassicEngine`` is built from seed, max_iter, update_iter and
    models. The checkpoint is trusted pickle: only load files generated by this
    script. If supplied by the CLI, command records the invocation.
    """
    counts = (years, refresh_days)
    if engine is None:
        counts += (max_iter, update_iter)
        engine = ClassicEngine(models, seed, max_iter, update_iter)
    if any(not isinstance(value, int) or value < 1 for value in counts):
        raise ValueError("Years and iteration/refresh counts must be positive integers")
    n_branches = engine.n_branches
    output_dir = Path(output_dir)
    prices, start, features, returns, initial_dates, decisions = _schedule(
        prices, start, end, years)
    fingerprint, settings = _fingerprint(prices, start.isoformat(), years,
                                         refresh_days, engine)
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
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("Use an empty output folder or a compatible checkpoint")
        state = dict(fingerprint=fingerprint, last_date=None, rows=[], branches=None)
        completed = 0
    if command is not None:
        _atomic_bytes(output_dir / "command.txt", command.encode("utf-8"))
    _atomic_bytes(output_dir / "settings.json", json.dumps(settings, indent=2).encode())
    started = time.monotonic()
    for offset, date in enumerate(decisions[completed:], start=completed):
        history = (initial_dates if offset == 0
                   else features.loc[initial_dates[0]:date].index)
        is_refresh = offset % refresh_days == 0
        if is_refresh:
            print(f"Refitting {date.date()}: {n_branches} models, "
                  f"{len(history)} training days", flush=True)
            refresh_started = time.monotonic()
            state["branches"] = engine.refresh(features, returns, history)
            print(f"Refit done in {time.monotonic() - refresh_started:.1f}s "
                  f"({engine.describe(state['branches'])})", flush=True)
        state["rows"].extend(engine.decide(
            state["branches"], features, returns, history, date, is_refresh))
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


def _run_block(args, prices, block, threads):
    """Run one refresh block: a fresh refit, then daily warm-start updates.

    Every refit reseeds and rebuilds the scaler, PCA and model, so a block
    depends only on its own dates and matches the sequential run.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(threads):
        try:
            import torch
            torch.set_num_threads(threads)
        except ImportError:
            pass
        engine = build_engine(args)
        _, _, features, returns, initial_dates, decisions = _schedule(
            prices, args.start, args.end, args.years)
        first = block * args.refresh_days
        last = min(first + args.refresh_days, len(decisions))
        started = time.monotonic()
        rows, branches = [], None
        for offset in range(first, last):
            date = decisions[offset]
            history = (initial_dates if offset == 0
                       else features.loc[initial_dates[0]:date].index)
            is_refresh = branches is None
            if is_refresh:
                branches = engine.refresh(features, returns, history)
            rows.extend(engine.decide(branches, features, returns, history,
                                      date, is_refresh))
    final = branches if last == len(decisions) else None
    return block, rows, final, time.monotonic() - started


def run_parallel(args, prices, workers, command=None):
    """Run refresh blocks in parallel worker processes and merge their signals.

    Finished blocks are saved under ``blocks/`` and skipped when resuming.
    The merged ``signals.csv`` is written once every block is done, and the
    final block's models are saved to ``last_models.pkl`` (trusted pickle).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    engine = build_engine(args)
    _, start, _, _, _, decisions = _schedule(prices, args.start, args.end,
                                             args.years)
    fingerprint, settings = _fingerprint(prices, start.isoformat(), args.years,
                                         args.refresh_days, engine)
    output_dir = Path(args.output)
    blocks_dir = output_dir / "blocks"
    if (output_dir / "checkpoint.pkl").exists():
        raise ValueError("Folder holds a sequential checkpoint; use a new folder")
    stamp = output_dir / "fingerprint.txt"
    if stamp.exists() and stamp.read_text() != fingerprint:
        raise ValueError("Saved blocks use other data or settings; use a new folder")
    if not stamp.exists() and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty output folder or a compatible one")
    _atomic_bytes(stamp, fingerprint.encode())
    _atomic_bytes(output_dir / "settings.json", json.dumps(settings, indent=2).encode())
    if command is not None:
        _atomic_bytes(output_dir / "command.txt", command.encode("utf-8"))
    n_blocks = -(-len(decisions) // args.refresh_days)
    paths = [blocks_dir / f"block_{block:03d}.csv" for block in range(n_blocks)]
    # Later blocks train on more history and run longest, so submit them first.
    pending = [block for block in reversed(range(n_blocks))
               if not paths[block].exists()]
    threads = max(1, (os.cpu_count() or 1) // workers)
    print(f"{n_blocks} blocks, {len(pending)} to run on {workers} workers "
          f"x {threads} threads", flush=True)
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        futures = [pool.submit(_run_block, args, prices, block, threads)
                   for block in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            block, rows, final, seconds = future.result()
            _atomic_bytes(paths[block], pd.DataFrame(rows).to_csv(index=False).encode())
            if final is not None:
                _atomic_bytes(output_dir / "last_models.pkl", pickle.dumps(
                    dict(fingerprint=fingerprint, last_date=decisions[-1],
                         branches=final), protocol=pickle.HIGHEST_PROTOCOL))
            print(f"block {block + 1}/{n_blocks} done in {seconds:.0f}s "
                  f"({done}/{len(pending)}, elapsed "
                  f"{time.monotonic() - started:.0f}s)", flush=True)
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(table) != len(decisions) * engine.n_branches:
        raise ValueError("Merged blocks have missing or duplicated decisions")
    _atomic_bytes(output_dir / "signals.csv", table.to_csv(index=False).encode())
    print(f"Merged {len(table)} rows into {output_dir / 'signals.csv'}", flush=True)
    return table


def build_engine(args):
    """Build the engine selected by the command-line arguments."""
    if args.engine == "classic":
        return ClassicEngine(args.models, args.seed, args.max_iter, args.update_iter)
    if args.engine == "flow":
        return FlowEngine(args.seed, args.max_iter, args.update_iter,
                          n_states=args.states, device=args.device,
                          base_distribution=args.base, base_df=args.base_df)
    return MixedEngine(args.emissions, args.seed, args.max_iter, args.update_iter)


def main():
    """Run the command-line experiment from a repository checkout."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["classic", "flow", "mixed"],
                        default="classic")
    parser.add_argument("--start", default="2004-01-02")
    parser.add_argument("--end", default="2025-12-30")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=None,
                        help="EM iterations (classic, flow; default 100) or "
                             "Adam steps per refit (mixed; default 400)")
    parser.add_argument("--update-iter", type=int, default=3)
    parser.add_argument("--refresh-days", type=int, default=63)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        choices=list(MODEL_KEYS), metavar="MODEL",
                        help="classic only; options: " + " ".join(MODEL_KEYS))
    parser.add_argument("--states", type=int, default=2, help="flow only")
    parser.add_argument("--device", default=None, help="flow only")
    parser.add_argument("--base", choices=["student_t", "normal"],
                        default=FLOW_DEFAULTS["base_distribution"],
                        help="flow only; base distribution")
    parser.add_argument("--base-df", type=float, default=FLOW_DEFAULTS["base_df"],
                        help="flow only; fixed Student-t base df (> 2)")
    parser.add_argument("--emissions", nargs="+", default=["gaussian", "student_t"],
                        choices=["gaussian", "student_t"],
                        help="mixed only; one family per state")
    parser.add_argument("--workers", type=int, default=1,
                        help="run refresh blocks in this many processes; 1 = "
                             "sequential with a daily checkpoint")
    args = parser.parse_args()
    if args.max_iter is None:
        args.max_iter = 400 if args.engine == "mixed" else 100
    source = ROOT / "data/processed/sp500_index_price.csv"
    prices = pd.read_csv(source, parse_dates=["YYYYMMDD"])
    prices = prices.set_index("YYYYMMDD")["DlyPrcInd"]
    # Reconstruct the invocation; original shell whitespace is unavailable.
    prefix = (f"PYTHONPATH={shlex.quote(os.environ['PYTHONPATH'])} "
              if "PYTHONPATH" in os.environ else "")
    command = prefix + shlex.join([sys.executable, *sys.argv]) + "\n"
    if args.workers > 1:
        run_parallel(args, prices, args.workers, command=command)
        return
    run(prices, args.output, args.start, args.end, years=args.years,
        refresh_days=args.refresh_days, command=command,
        engine=build_engine(args))


if __name__ == "__main__":
    main()
