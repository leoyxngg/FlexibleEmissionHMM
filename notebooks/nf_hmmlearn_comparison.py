from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
import multiprocessing as mp
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, stats
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
from threadpoolctl import threadpool_limits

from flexible_emission_hmm import (
    GaussianMixtureEmission,
    HMM,
    StateParameters,
)
from hmmlearn_flow_hmm import FlowHMM
from model_evaluation.synthetic.state_alignment import StateAlignment
from model_evaluation.synthetic.state_alignment import state_log_loss_metrics
import synthetic_emission_comparison as reference
from volatility_markov_comparison import make_test_variants

MODELS = ("Gaussian HMM", "GMM HMM", "RQ-NSF Gaussian base", "RQ-NSF Student-t base")
FLOW_BASES = dict(zip(MODELS[2:], ("normal", "student_t")))
COLORS = dict(zip(MODELS, ("#4477AA", "#CC7722", "#228833", "#AA3377")))
STYLES = dict(zip(MODELS, ("--", "-.", "-", ":")))
KEYS = ["regime", "family", "n_train", "replicate"]
_WORKER_THREAD_LIMITS = None
SOURCES = {
    "mean": "data/synthetic/gmm_synthetic_experiments/20261024T212804Z-4r1by0vo",
    "volatility": "data/synthetic/volatility_markov_experiments/20261020T110835Z-8lz5ptfh",
}
METRICS = list(
    dict.fromkeys(
        reference.SUMMARY_METRICS
        + [
            "mean_true_state_posterior",
            "true_switches",
            "inferred_switches",
            "switch_count_error",
            "absolute_switch_count_error",
            "moment_quad_error_max",
        ]
    )
)

@dataclass(frozen=True)
class FlowSettings:
    """Configure the numerical emission M-step and state-specific splines."""

    count_bins: int = 8
    num_layers: int = 1
    m_step_steps: int = 20
    batch_size: int | None = None
    learning_rate: float = 1e-3
    bound: float = 5.0
    device: str = "cpu"
    dtype: str = "float64"
    base_distribution: str = "normal"
    base_df: float = 5.0
    convergence_patience: int = 5

@dataclass(frozen=True)
class TuningSettings:
    """Select flows using held-out suffixes of saved training sequences only."""

    count_bins: tuple = (8, 16, 32)
    num_layers: tuple = (1, 2)
    m_step_steps: tuple = (20,)
    learning_rates: tuple = (1e-3,)
    student_dfs: tuple = (5.0,)
    replicates: tuple = (0, 1, 2)
    train_fraction: float = 0.8
    n_starts: int = 3
    max_iter: int = 100

    def __post_init__(self):
        """Validate the temporal holdout and tuning budgets."""
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between zero and one")
        if self.n_starts < 1 or self.max_iter < 1 or not self.replicates:
            raise ValueError("Provide tuning replicates and positive fitting budgets")
        if len(set(self.replicates)) != len(self.replicates):
            raise ValueError("Tuning replicates must be unique")

def read_json(path):
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text())

def write_json(path, value):
    """Write JSON without nonstandard NaN values."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(reference.json_safe(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)

def digest(path):
    """Hash an input artifact for exact provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class SavedExperiment:
    """Load exact observations and selected baseline parameters from a saved run."""

    def __init__(self, root, regime, relative_path, shape_parameters):
        """Validate source metadata and retain its state-aligned baseline fits."""
        self.root = Path(root)
        self.regime = regime
        self.path = self.root / relative_path
        self.manifest = read_json(self.path / "manifest.json")
        self.settings = self.manifest["settings"]
        self.families = tuple(self.manifest["families"])
        self.transitions = np.asarray(self.manifest["transitions"])
        expected = np.full((3, 3), 0.0025)
        np.fill_diagonal(expected, 0.995)
        if not np.allclose(self.transitions, expected, atol=1e-14):
            raise ValueError(f"{self.path} is not the requested 0.995 experiment")
        self.index = {
            (r["replicate"], r["family"]): r["file"]
            for r in read_json(self.path / "dataset_index.json")
        }
        self.parameters = {
            (r["family"], r["n_train"], r["replicate"], r["model"]): r
            for r in read_json(self.path / "parameters.json")
        }
        self.saved_results = pd.DataFrame(read_json(self.path / "results.json"))
        self.saved_attempts = pd.DataFrame(read_json(self.path / "attempts.json"))
        state_rows = pd.DataFrame(read_json(self.path / "per_state.json"))
        if "condition" in state_rows:
            state_rows = state_rows[state_rows.condition == "clean"]
        self.state_kl = {
            (r["family"], r["n_train"], r["replicate"], r["model"], r["state"]): r[
                "emission_kl"
            ]
            for r in state_rows.to_dict("records")
        }
        self.shape_parameters = shape_parameters
        self.noise_levels = tuple(self.settings.get("noise_levels", (0.1, 0.25, 0.5)))
        # Mean-run perturbation results were saved separately from its clean results.
        perturbation = self.path / "mean_regime_perturbation/results.csv"
        if regime == "mean" and perturbation.exists():
            noisy = pd.read_csv(perturbation)
            self.saved_results = pd.concat(
                [
                    self.saved_results.assign(condition="clean"),
                    noisy[noisy.condition != "clean"],
                ],
                ignore_index=True,
            )
        elif "condition" not in self.saved_results:
            self.saved_results["condition"] = "clean"
        if self.saved_results.duplicated(
            ["family", "n_train", "replicate", "model", "condition"]
        ).any():
            raise ValueError("Duplicate baseline result identities")

    def data(self, replicate, family, splits=("train", "test")):
        """Recover saved train/test rows without regenerating any observations."""
        path = self.path / self.index[replicate, family]
        requested = splits
        splits = {}
        with np.load(path, allow_pickle=False) as archive:
            for split in requested:
                columns = {
                    key.split("__", 1)[1]: archive[key].copy()
                    for key in archive.files
                    if key.startswith(split + "__")
                }
                splits[split] = pd.DataFrame(columns)
                frame = splits[split]
                if not np.isfinite(frame.x).all() or not np.array_equal(
                    np.unique(frame.state), [0, 1, 2]
                ):
                    raise ValueError(
                        f"Invalid data or missing state in {path}: {split}"
                    )
        if "test" in splits and len(splits["test"]) != self.settings["n_test"]:
            raise ValueError("Saved test length does not match the manifest")
        return splits

    def variants(self, train, test, replicate, family, n_train):
        """Reproduce the saved measurement-noise schedule and random draws."""
        config = SimpleNamespace(
            master_seed=self.settings["master_seed"], noise_levels=self.noise_levels
        )
        variants, _ = make_test_variants(
            config, train, test, replicate, family, n_train
        )
        return variants

    def baseline(self, family, n_train, replicate, name):
        """Restore the saved selected model in its frozen true-state alignment."""
        record = self.parameters[family, n_train, replicate, name]
        means = np.asarray(record["means"], dtype=float)
        emission = GaussianMixtureEmission(3, n_components=means.shape[1])
        emission.n_features = 1
        emission.means = means[..., None]
        emission.covariances = np.asarray(record["variances"])[..., None, None]
        emission.mixture_weights = np.asarray(record["weights"])
        model = HMM(emission)
        model.states = StateParameters(
            np.asarray(record["aligned_initial_probs"]),
            np.asarray(record["aligned_transitions"]),
        )
        model.fitted = True
        return model

    def true_moments(self):
        """Return the state means and variances recorded by the generating run."""
        if self.regime == "volatility":
            return (
                np.asarray(self.manifest["means"]),
                np.asarray(self.manifest["stds"]) ** 2,
            )
        d = self.shape_parameters
        return np.asarray(d["means"]), np.full(3, d["variance"])

    def true_log_density(self, family, state, values):
        """Evaluate the manifest-defined density, including volatility scaling."""
        d = self.shape_parameters
        means, variances = self.true_moments()
        scale = np.sqrt(variances[state] / d["variance"])
        x = (np.asarray(values) - means[state]) / scale + d["means"][state]
        mean = d["means"][state]
        std = np.sqrt(d["variance"])
        if family == "Gaussian":
            result = stats.norm.logpdf(x, mean, std)
        elif family == "Student-t":
            df = d["student_t_df"]
            result = stats.t.logpdf(x, df, loc=mean, scale=std * np.sqrt((df - 2) / df))
        elif family == "Skewed Gamma":
            shape = d["gamma_shape"]
            result = stats.gamma.logpdf(
                x, shape, loc=mean - std * np.sqrt(shape), scale=std / np.sqrt(shape)
            )
        elif family == "Trimodal":
            result = logsumexp(
                np.log(d["mixture_weights"][state])
                + stats.norm.logpdf(
                    np.asarray(x)[..., None],
                    d["mixture_means"][state],
                    d["mixture_stds"][state],
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Unsupported saved family: {family}")
        return result - np.log(scale)

def load_sources(root, paths=None):
    """Load both 0.995 sources and check their common emission-shape provenance."""
    paths = SOURCES if paths is None else paths
    mean_manifest = read_json(Path(root) / paths["mean"] / "manifest.json")
    sources = {
        regime: SavedExperiment(root, regime, path, mean_manifest["distributions"])
        for regime, path in paths.items()
    }
    helper = "notebooks/synthetic_emission_comparison.py"
    if (
        sources["mean"].manifest["source_hashes"][helper]
        != sources["volatility"].manifest["source_hashes"][helper]
    ):
        raise ValueError("Saved runs used different base emission definitions")
    return sources

def emission_model(model):
    """Return the fitted emission object for either comparison backend."""
    return model.emission_ if isinstance(model, FlowHMM) else model.emission

def chain_parameters(model):
    """Return state parameters without changing either model's state ordering."""
    if isinstance(model, FlowHMM):
        return StateParameters(model.startprob_, model.transmat_)
    return model.states

def _validate_execution(n_jobs, torch_threads):
    for name, value in (("n_jobs", n_jobs), ("torch_threads", torch_threads)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

def _initialize_cpu_worker(torch_threads):
    global _WORKER_THREAD_LIMITS
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    _WORKER_THREAD_LIMITS = threadpool_limits(limits=torch_threads)

@contextmanager
def _cpu_executor(n_jobs, torch_threads):
    _validate_execution(n_jobs, torch_threads)
    previous = torch.get_num_threads()
    if previous != torch_threads:
        torch.set_num_threads(torch_threads)
    try:
        with threadpool_limits(limits=torch_threads):
            if n_jobs == 1:
                yield None
            else:
                executor = ProcessPoolExecutor(
                    max_workers=n_jobs,
                    mp_context=mp.get_context("spawn"),
                    initializer=_initialize_cpu_worker,
                    initargs=(torch_threads,),
                )
                try:
                    yield executor
                except BaseException:
                    # Python 3.10 has no public terminate_workers API. Stop
                    # active fits on interruption instead of waiting for every
                    # queued fold. Completed atomic checkpoints remain usable.
                    processes = list(executor._processes.values())
                    executor.shutdown(wait=False, cancel_futures=True)
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    for process in processes:
                        process.join(timeout=1)
                    raise
                else:
                    executor.shutdown(wait=True)
    finally:
        if previous != torch_threads:
            torch.set_num_threads(previous)

def _fit_single_start(job):
    train, settings, start, max_iter, tolerance, seed_parts, checkpoint_dir = job
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed = int.from_bytes(
        hashlib.sha256(json.dumps([*seed_parts, start]).encode()).digest()[:4],
        "little",
    )
    row = dict(start=start, seed=seed, status="failed", selected=False)
    saved = checkpoint_dir / f"start_{start}.json"
    if saved.exists():
        row = read_json(saved)
        row["selected"] = False
        return row
    began = time.perf_counter()
    try:
        model = FlowHMM(
            3, random_state=seed, n_iter=max_iter, tol=tolerance,
            init_params="e", **asdict(settings),
        )
        model.startprob_ = np.full(3, 1 / 3)
        model.transmat_ = np.full((3, 3), 1 / 3)
        model.fit(train)
        # Fit records the initial score and every actual post-update score.
        history = np.asarray(model.history_)
        if not np.isfinite(history).all():
            raise FloatingPointError("Nonfinite likelihood history")
        row.update(
            status="ok",
            train_ll=float(history[-1]),
            iterations=model.monitor_.iter,
            converged=model.converged_,
            iteration_limit_reached=model.monitor_.iter == max_iter,
            monitor_converged=bool(model.monitor_.converged),
            convergence_patience=model.convergence_patience,
            consecutive_small_changes=model.monitor_.consecutive_small_changes,
            stop_reason=model.stop_reason_,
            history=history.tolist(),
            likelihood_decreases=int(np.sum(np.diff(history) < -1e-6)),
            final_gain_per_observation=float(np.diff(history)[-1] / len(train)),
        )
        save_flow(saved.with_suffix(".pt"), model, settings)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["fit_seconds"] = time.perf_counter() - began
    write_json(saved, row)
    return row

def fit_starts(
    train, settings, n_starts, max_iter, tolerance, seed_parts, checkpoint_dir=None,
    *, n_jobs=1, torch_threads=1, _executor=None,
):
    """Select starts by training likelihood using optional independent CPU workers.

    Preserve start order, seeds, and first-start tie breaking across schedules.
    Use spawn workers only for CPU settings. Reuse an internal executor during
    whole experiment stages; otherwise create a bounded pool for this call.
    Each start owns a checkpoint, and only the selected model is restored in
    the parent process. Keep temporary checkpoints when no directory is given
    only until the selected model is loaded. Count fit_seconds per attempt.
    """
    _validate_execution(n_jobs, torch_threads)
    if isinstance(n_starts, bool) or not isinstance(n_starts, Integral) or n_starts < 1:
        raise ValueError("n_starts must be a positive integer")
    if n_jobs > 1 and torch.device(settings.device).type != "cpu":
        raise ValueError("Parallel starts require device='cpu'")
    directory = (TemporaryDirectory() if checkpoint_dir is None
                 else nullcontext(str(checkpoint_dir)))
    context = (_cpu_executor(n_jobs, torch_threads) if _executor is None
               else nullcontext(_executor))
    with directory as directory_name, context as executor:
        jobs = [
            (train, settings, start, max_iter, tolerance, seed_parts, directory_name)
            for start in range(n_starts)
        ]
        iterator = (map(_fit_single_start, jobs) if executor is None
                    else executor.map(_fit_single_start, jobs))
        attempts = list(iterator)
        valid = [row for row in attempts if row["status"] == "ok"]
        if not valid:
            return None, attempts
        best = max(valid, key=lambda row: row["train_ll"])
        best["selected"] = True
        model = restore_flow(
            Path(directory_name) / f"start_{best['start']}.pt", train
        )
        return model, attempts

def _fit_tuning_fold(job):
    (source, regime, family, candidate, replicate, settings,
     tuning, output, threads) = job
    output = Path(output)
    checkpoint = output / f"tune_{regime}_{family}_{candidate}_{replicate}.json"
    if checkpoint.exists():
        return read_json(checkpoint)
    frame = source.data(replicate, family, ("train",))["train"]
    n_train = max(source.settings["train_sizes"])
    x = frame[["x"]].to_numpy()[:n_train]
    cut = int(len(x) * tuning.train_fraction)
    row = dict(
        regime=regime, family=family, candidate=candidate, replicate=replicate,
        settings=asdict(settings), status="failed", prefix_length=cut,
        validation_length=len(x) - cut,
    )
    # A worker owns the entire fold; its restarts stay sequential. Do not nest pools.
    model, attempts = fit_starts(
        x[:cut], settings, tuning.n_starts, tuning.max_iter,
        cut * source.settings["tol_per_observation"],
        [source.settings["master_seed"], "tuning", regime, family, replicate],
        checkpoint_dir=output / (checkpoint.stem + "_starts"),
        torch_threads=threads,
    )
    row["attempts"] = attempts
    if model is not None:
        try:
            _, filtered, _ = model.filter(x[:cut])
            prior = filtered[-1] @ chain_parameters(model).transitions
            _, _, scores = model.filter(x[cut:], initial_prior=prior)
            row.update(status="ok", validation_nll=float(-scores.mean()))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    write_json(checkpoint, row)
    return row

def _tune_base(sources, output, base, tuning, n_jobs, torch_threads, executor):
    """Choose per-regime/family settings using training-only temporal holdouts.

    Select each restart by prefix training likelihood, then score its suffix
    conditional on the prefix. Require every configured validation fold to
    succeed. Never inspect final test likelihoods or labels for selection.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = dict(
        base=asdict(base),
        tuning=asdict(tuning),
        execution=dict(n_jobs=n_jobs, torch_threads=torch_threads),
        sources={
            k: {
                "manifest": digest(s.path / "manifest.json"),
                "datasets": {name: digest(s.path / name) for name in s.index.values()},
            }
            for k, s in sources.items()
        },
        backend="hmmlearn",
        hmmlearn_version=version("hmmlearn"),
        backend_implementation=digest(
            Path(__file__).parents[1] / "src/hmmlearn_flow_hmm/model.py"
        ),
        implementation=digest(__file__),
        emission_implementation=digest(
            Path(__file__).parents[1] / "src/flexible_emission_hmm/flow_emissions.py"
        ),
    )
    request_path = output / "tuning_request.json"
    if request_path.exists() and read_json(request_path) != reference.json_safe(
        request
    ):
        raise ValueError("Tuning settings changed; use a new output directory")
    write_json(request_path, request)
    rows, selections = [], []
    grid = list(
        itertools.product(
            tuning.count_bins,
            tuning.num_layers,
            tuning.m_step_steps,
            tuning.learning_rates,
            (
                tuning.student_dfs
                if base.base_distribution == "student_t"
                else (base.base_df,)
            ),
        )
    )
    jobs = []
    for regime, source in sources.items():
        for family in source.families:
            for candidate, (bins, layers, steps, rate, df) in enumerate(grid):
                settings = replace(
                    base, count_bins=bins, num_layers=layers, m_step_steps=steps,
                    learning_rate=rate, base_df=df,
                )
                for replicate in tuning.replicates:
                    jobs.append((
                        source, regime, family, candidate, replicate, settings,
                        tuning, output, torch_threads,
                    ))
    iterator = (map(_fit_tuning_fold, jobs) if executor is None
                else executor.map(_fit_tuning_fold, jobs))
    for row in iterator:
        rows.append(row)
        print(f"Validation {row['regime']} / {row['family']} / "
              f"candidate {row['candidate']} / replicate {row['replicate']}: "
              f"{row['status']}", flush=True)
    for regime, source in sources.items():
        for family in source.families:
            candidates = []
            for candidate in range(len(grid)):
                folds = [r for r in rows if (r["regime"], r["family"], r["candidate"])
                         == (regime, family, candidate)]
                if len(folds) == len(tuning.replicates) and all(
                    r["status"] == "ok" for r in folds
                ):
                    candidates.append((
                        np.mean([r["validation_nll"] for r in folds]),
                        candidate, folds[0]["settings"],
                    ))
            if not candidates:
                raise RuntimeError(f"No complete tuning candidate: {regime}, {family}")
            score, candidate, settings = min(candidates, key=lambda c: (c[0], c[1]))
            selections.append(dict(
                regime=regime, family=family, candidate=candidate,
                validation_nll=score, settings=settings,
            ))
            write_json(output / "selected_settings.json", selections)
            print(f"Tuned {regime} / {family}: {settings}", flush=True)
    write_json(output / "tuning_results.json", rows)
    return selections, rows

def tune_flows(
    sources, output, base=FlowSettings(), tuning=TuningSettings(),
    *, n_jobs=1, torch_threads=1,
):
    """Tune both bases using optional parallel CPU validation folds.

    Reuse one spawn pool across both bases, with one fold per worker and
    sequential restarts inside each fold. Preserve training-only selection,
    seeds, grid order, and fold coverage independently of completion order.
    """
    began = time.perf_counter()
    _validate_execution(n_jobs, torch_threads)
    if n_jobs > 1 and torch.device(base.device).type != "cpu":
        raise ValueError("Parallel tuning requires device='cpu'")
    output = Path(output)
    selections, rows = [], []
    with _cpu_executor(n_jobs, torch_threads) as executor:
        for distribution in FLOW_BASES.values():
            chosen, scores = _tune_base(
                sources, output / distribution,
                replace(base, base_distribution=distribution), tuning,
                n_jobs, torch_threads, executor,
            )
            selections.extend(chosen)
            rows.extend(scores)
    write_json(output / "selected_settings.json", selections)
    write_json(output / "tuning_results.json", rows)
    write_json(output / "execution_timing.json", dict(
        n_jobs=n_jobs, torch_threads=torch_threads,
        invocation_wall_seconds=time.perf_counter() - began,
        note="Elapsed time for this call, including startup and any resumed work",
    ))
    return selections, rows

def evaluate(model, frame, transitions, mapping=None):
    """Compute reference state, probability, transition, and switch metrics."""
    x, truth = frame[["x"]].to_numpy(), frame.state.to_numpy()
    if isinstance(model, FlowHMM):
        log_likelihood, probabilities = model.score_samples(x)
    else:
        inference = model.posterior(x)
        log_likelihood = inference.log_likelihood
        probabilities = inference.responsibilities
    predicted = model.predict(x)
    fitted_transitions = chain_parameters(model).transitions
    if mapping is not None:
        predicted = mapping[predicted]
        order = np.argsort(mapping)
        probabilities = probabilities[:, order]
        fitted_transitions = fitted_transitions[np.ix_(order, order)]
    metrics, per_state, matrix = reference.classification_metrics(truth, predicted)
    true_count = int(np.count_nonzero(np.diff(truth)))
    inferred_count = int(np.count_nonzero(np.diff(predicted)))
    metrics.update(
        ari=float(adjusted_rand_score(truth, predicted)),
        nmi=float(normalized_mutual_info_score(truth, predicted)),
        aligned_accuracy=float(np.mean(truth == predicted)),
        **state_log_loss_metrics(probabilities, truth),
        state_brier_score=float(
            np.mean(np.sum((probabilities - np.eye(3)[truth]) ** 2, axis=1))
        ),
        test_nll=float(-log_likelihood / len(x)),
        transition_rmse=float(
            np.sqrt(np.mean((fitted_transitions - transitions) ** 2))
        ),
        mean_true_state_posterior=float(probabilities[np.arange(len(x)), truth].mean()),
        true_switches=true_count,
        inferred_switches=inferred_count,
        switch_count_error=inferred_count - true_count,
        absolute_switch_count_error=abs(inferred_count - true_count),
    )
    return metrics, per_state, matrix, predicted, probabilities

def flow_recovery(source, family, model, mapping):
    """Integrate clean-target KL and fitted moments using adaptive 1D quadrature."""
    means, variances = source.true_moments()
    kl, errors, fitted_means, fitted_variances, moment_errors = [], [], [], [], []
    settings = source.settings
    options = dict(
        epsabs=settings["quad_epsabs"],
        epsrel=settings["quad_epsrel"],
        limit=settings["quad_limit"],
    )
    for state, learned in enumerate(np.argsort(mapping)):

        def log_q(x):
            """Evaluate the selected state's fitted scalar log density."""
            return float(emission_model(model).log_prob(np.array([[x]]))[0, learned])

        def ratio_integrand(x):
            """Weight the log density ratio by the true emission density."""
            log_p = float(source.true_log_density(family, state, x))
            if not np.isfinite(log_p) or log_p < -745:
                return 0.0
            return float(np.exp(log_p) * (log_p - log_q(x)))

        lower = -np.inf
        if family == "Skewed Gamma":
            lower = means[state] - np.sqrt(
                variances[state] * source.shape_parameters["gamma_shape"]
            )
        # Finite central segments resolve narrow mixture peaks; include both tails.
        std = np.sqrt(variances[state])
        breaks = np.linspace(means[state] - 12 * std, means[state] + 12 * std, 49)
        breaks = [lower, *[b for b in breaks if b > lower], np.inf]
        pieces = [
            integrate.quad(ratio_integrand, a, b, **options)
            for a, b in zip(breaks[:-1], breaks[1:])
        ]
        value, error = np.sum(pieces, axis=0)
        if value < -max(1e-6, 10 * error):
            raise FloatingPointError("Negative KL beyond quadrature tolerance")
        kl.append(float(value))
        errors.append(float(error))

        def moment_integrand(x):
            """Integrate fitted probability mass and the first two moments."""
            density = np.exp(log_q(x))
            return density * np.array([1.0, x, x * x])

        moment, moment_error = integrate.quad_vec(
            moment_integrand,
            -np.inf,
            np.inf,
            epsabs=settings["quad_epsabs"],
            epsrel=settings["quad_epsrel"],
            limit=settings["quad_limit"],
        )
        if not np.isclose(moment[0], 1, atol=1e-5):
            raise FloatingPointError("Fitted flow density failed normalization check")
        fitted_means.append(moment[1])
        fitted_variances.append(moment[2] - moment[1] ** 2)
        moment_errors.append(float(moment_error))
    metrics = dict(
        emission_kl=float(np.mean(kl)),
        kl_quad_error_max=float(max(errors)),
        mean_rmse=float(np.sqrt(np.mean((fitted_means - means) ** 2))),
        variance_rmse=float(np.sqrt(np.mean((fitted_variances - variances) ** 2))),
        moment_quad_error_max=float(max(moment_errors)),
    )
    if not np.isfinite(list(metrics.values())).all():
        raise FloatingPointError("Nonfinite flow recovery metrics")
    return metrics, kl

def save_flow(path, model, settings):
    """Save versioned hmmlearn flow parameters and final training diagnostics."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        dict(
            backend="hmmlearn",
            format_version=3,
            settings=asdict(settings),
            flows={
                k: v.detach().cpu()
                for k, v in model.emission_.flows.state_dict().items()
            },
            start=torch.tensor(model.startprob_),
            transitions=torch.tensor(model.transmat_),
            history=model.history_,
            converged=model.converged_,
            iterations=model.monitor_.iter,
            n_iter=model.n_iter,
            tol=model.tol,
            batch_rng_state=model.emission_._batch_rng.bit_generator.state,
        ),
        temporary,
    )
    temporary.replace(path)

def restore_flow(path, train, device=None, dtype=None):
    """Restore a hmmlearn checkpoint without refitting or using old-backend states."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_version = checkpoint.get("format_version")
    if checkpoint.get("backend") != "hmmlearn" or checkpoint_version not in (2, 3):
        raise ValueError("Expected a version-2 or version-3 hmmlearn checkpoint")
    settings = dict(checkpoint["settings"])
    if checkpoint_version == 2:
        settings.setdefault("convergence_patience", 1)
    if device is not None:
        settings["device"] = device
    if dtype is not None:
        settings["dtype"] = dtype
    model = FlowHMM(
        3,
        random_state=0,
        n_iter=checkpoint["n_iter"],
        tol=checkpoint["tol"],
        init_params="",
        **settings,
    )
    model.emission_ = model._make_emission(0)
    model.emission_.initialize(train)
    model.emission_.flows.load_state_dict(checkpoint["flows"])
    if "batch_rng_state" in checkpoint:
        model.emission_._batch_rng.bit_generator.state = checkpoint["batch_rng_state"]
    model.n_features = model.emission_.n_features
    model.startprob_ = checkpoint["start"].numpy()
    model.transmat_ = checkpoint["transitions"].numpy()
    model.history_ = checkpoint["history"]
    model.converged_ = checkpoint["converged"]
    model.monitor_.initial_log_prob = checkpoint["history"][0]
    for log_prob in checkpoint["history"][1:]:
        model.monitor_.report(log_prob)
    model.iteration_limit_reached_ = model.monitor_.iter >= model.n_iter
    model.stop_reason_ = "tolerance" if model.converged_ else "max_iter"
    if checkpoint_version == 2:
        model.stop_reason_ = "legacy"
    model._check()
    return model

def complete_groups(results, conditions=None):
    """Keep identical successful replicate groups for all models and conditions."""
    conditions = (
        list(results.condition.unique()) if conditions is None else list(conditions)
    )
    subset = results[results.condition.isin(conditions)]
    rows = subset[subset.evaluation_status == "ok"]
    identity = KEYS + ["model", "condition"]
    if rows.duplicated(identity).any():
        raise ValueError("Duplicate comparison rows")
    rows = rows[rows.model.isin(MODELS)]
    counts = rows.groupby(KEYS).size()
    good = counts[counts == len(MODELS) * len(conditions)].reset_index()[KEYS]
    return rows.merge(good, on=KEYS, validate="many_to_one")

def run_comparison(
    sources,
    output,
    selections,
    *,
    replicates=None,
    families=None,
    n_starts=None,
    max_iter=None,
    n_jobs=1,
    torch_threads=1,
):
    """Fit comparison models with optional persistent CPU restart workers.

    Keep metrics and reports in the parent process. Dispatch only independent
    flow starts to workers, with distinct seeds and checkpoint files. Preserve
    the source experiment's data, starts, and stopping budgets.
    """
    began = time.perf_counter()
    _validate_execution(n_jobs, torch_threads)
    if n_jobs > 1 and any(
        torch.device(row["settings"]["device"]).type != "cpu" for row in selections
    ):
        raise ValueError("Parallel final fitting requires device='cpu'")
    with _cpu_executor(n_jobs, torch_threads) as executor:
        result = _run_comparison(
            sources, output, selections, replicates=replicates, families=families,
            n_starts=n_starts, max_iter=max_iter, n_jobs=n_jobs,
            torch_threads=torch_threads, executor=executor,
        )
    write_json(Path(output) / "execution_timing.json", dict(
        n_jobs=n_jobs, torch_threads=torch_threads,
        invocation_wall_seconds=time.perf_counter() - began,
        note="Elapsed time for this call, including startup and any resumed work",
    ))
    return result

def _run_comparison(
    sources,
    output,
    selections,
    *,
    replicates=None,
    families=None,
    n_starts=None,
    max_iter=None,
    n_jobs=1,
    torch_threads=1,
    executor=None,
):
    """Fit flows on saved data and replay all baselines without refitting them.

    Inherit final training sizes, restarts, iteration caps, and tolerances from
    each source unless explicit overrides are recorded. Resume completed groups
    only when the run request and source artifact fingerprints match.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = {
        (r["regime"], r["family"], r["settings"]["base_distribution"]): FlowSettings(
            **r["settings"]
        )
        for r in selections
    }
    fingerprints = {}
    for regime, source in sources.items():
        names = [
            "manifest.json",
            "dataset_index.json",
            "parameters.json",
            "results.json",
            "attempts.json",
            "per_state.json",
        ]
        perturbation = "mean_regime_perturbation/results.csv"
        if (source.path / perturbation).exists():
            names.append(perturbation)
        names += list(source.index.values())
        fingerprints[regime] = {name: digest(source.path / name) for name in names}
    request = dict(
        sources={k: str(s.path.relative_to(s.root)) for k, s in sources.items()},
        fingerprints=fingerprints,
        selections=selections,
        execution=dict(n_jobs=n_jobs, torch_threads=torch_threads),
        overrides=dict(
            replicates=replicates,
            families=families,
            n_starts=n_starts,
            max_iter=max_iter,
        ),
        backend="hmmlearn",
        backend_implementation=digest(
            Path(__file__).parents[1] / "src/hmmlearn_flow_hmm/model.py"
        ),
        implementation=digest(__file__),
        emission_implementation=digest(
            Path(__file__).parents[1] / "src/flexible_emission_hmm/flow_emissions.py"
        ),
        versions=dict(
            torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
            hmmlearn=version("hmmlearn"),
        ),
        selection="Training LL for starts; training-only validation NLL for settings",
        comparison="All-condition complete Gaussian/GMM/RQ-NSF replicate groups",
        density_diagnostics="Clean-target adaptive 1D quadrature; CPU float64 evaluation",
        runtime_caveat="Baseline timings are historical; do not infer hardware speedups",
        created=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if {k: v for k, v in previous.items() if k != "created"} != reference.json_safe(
            {k: v for k, v in request.items() if k != "created"}
        ):
            raise ValueError(
                "Comparison inputs/settings changed; use a new output directory"
            )
    else:
        write_json(manifest_path, request)
    tables = {
        name: []
        for name in (
            "results",
            "attempts",
            "per_state",
            "confusion",
            "baseline_replay",
            "parameters",
            "coverage",
        )
    }
    for regime, source in sources.items():
        replicas = (
            range(source.settings["n_replicates"]) if replicates is None else replicates
        )
        for family in source.families:
            if families is not None and family not in families:
                continue
            for n_train in source.settings["train_sizes"]:
                for replicate in replicas:
                    identity = dict(
                        regime=regime,
                        family=family,
                        n_train=n_train,
                        replicate=replicate,
                    )
                    slug = f"{regime}_{family.replace(' ', '_')}_{n_train}_{replicate}"
                    group_dir = output / slug
                    group_dir.mkdir(exist_ok=True)
                    done = group_dir / "tables.json"
                    if done.exists():
                        group = read_json(done)
                    else:
                        group = {name: [] for name in tables}
                        data = source.data(replicate, family)
                        train = data["train"].iloc[:n_train]
                        if len(train) != n_train:
                            raise ValueError("Saved training prefix is too short")
                        x, y = train[["x"]].to_numpy(), train.state.to_numpy()
                        variants = source.variants(
                            train, data["test"], replicate, family, n_train
                        )
                        # Save exact recovered observations, labels and noise realizations.
                        np.savez_compressed(
                            group_dir / "data.npz",
                            **{
                                **{f"train__{c}": train[c].to_numpy() for c in train},
                                **{
                                    f"{condition}__{c}": frame[c].to_numpy()
                                    for condition, frame in variants.items()
                                    for c in frame
                                },
                            },
                        )
                        for split, frame in [("train", train), ("test", data["test"])]:
                            group["coverage"].append(
                                dict(
                                    identity,
                                    split=split,
                                    length=len(frame),
                                    state_visits=np.bincount(
                                        frame.state, minlength=3
                                    ).tolist(),
                                    switches=int(
                                        np.count_nonzero(np.diff(frame.state))
                                    ),
                                )
                            )
                        models = {
                            name: source.baseline(family, n_train, replicate, name)
                            for name in MODELS[:2]
                        }
                        group["parameters"].extend(
                            dict(
                                regime=regime,
                                origin="saved baseline",
                                **source.parameters[family, n_train, replicate, name],
                            )
                            for name in MODELS[:2]
                        )
                        flow_info = {}
                        for name, distribution in FLOW_BASES.items():
                            settings = selected[regime, family, distribution]
                            flow_path = group_dir / f"flow_{distribution}.pt"
                            attempts_path = group_dir / f"attempts_{distribution}.json"
                            if flow_path.exists() and attempts_path.exists():
                                flow = restore_flow(flow_path, x)
                                attempts = read_json(attempts_path)
                            else:
                                flow, attempts = fit_starts(
                                    x,
                                    settings,
                                    (
                                        source.settings["n_starts"]
                                        if n_starts is None
                                        else n_starts
                                    ),
                                    (
                                        source.settings["max_iter"]
                                        if max_iter is None
                                        else max_iter
                                    ),
                                    n_train * source.settings["tol_per_observation"],
                                    [
                                        source.settings["master_seed"],
                                        "final_flow",
                                        regime,
                                        family,
                                        replicate,
                                        n_train,
                                    ],
                                    checkpoint_dir=group_dir / f"starts_{distribution}",
                                    n_jobs=n_jobs,
                                    torch_threads=torch_threads,
                                    _executor=executor,
                                )
                                write_json(attempts_path, attempts)
                                if flow is not None:
                                    save_flow(flow_path, flow, settings)
                            group["attempts"].extend(
                                dict(identity, model=name, **r) for r in attempts
                            )
                            if flow is None:
                                continue
                            mapping = StateAlignment(3).fit(flow.predict(x), y).mapping
                            models[name] = flow
                            recovery, state_kl, recovery_warnings = {}, [], []
                            diagnostic_model = restore_flow(
                                flow_path, x, "cpu", "float64"
                            )
                            try:
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    recovery, state_kl = flow_recovery(
                                        source, family, diagnostic_model, mapping
                                    )
                                recovery_warnings = [str(w.message) for w in caught]
                            except Exception as exc:
                                recovery_warnings = [f"{type(exc).__name__}: {exc}"]
                            flow_info[name] = dict(
                                mapping=mapping,
                                recovery=recovery,
                                state_kl=state_kl,
                                warnings=recovery_warnings,
                                attempts=attempts,
                            )
                            order = np.argsort(mapping)
                            group["parameters"].append(
                                dict(
                                    identity,
                                    model=name,
                                    checkpoint=str(flow_path),
                                    settings=asdict(settings),
                                    mapping_learned_to_true=mapping.tolist(),
                                    aligned_transitions=chain_parameters(flow).transitions[
                                        np.ix_(order, order)
                                    ].tolist(),
                                    aligned_initial_probs=chain_parameters(flow).start[
                                        order
                                    ].tolist(),
                                )
                            )
                        historical = source.saved_attempts[
                            (source.saved_attempts.family == family)
                            & (source.saved_attempts.n_train == n_train)
                            & (source.saved_attempts.replicate == replicate)
                        ]
                        group["attempts"].extend(
                            dict(regime=regime, origin="saved baseline", **r)
                            for r in historical.to_dict("records")
                        )
                        for name in MODELS:
                            for condition, frame in variants.items():
                                row = dict(
                                    identity,
                                    model=name,
                                    condition=condition,
                                    evaluation_status="failed",
                                )
                                if name not in models:
                                    row["error"] = "All flow fits failed"
                                    group["results"].append(row)
                                    continue
                                model = models[name]
                                saved = source.saved_results[
                                    (source.saved_results.family == family)
                                    & (source.saved_results.n_train == n_train)
                                    & (source.saved_results.replicate == replicate)
                                    & (source.saved_results.model == name)
                                    & (source.saved_results.condition == condition)
                                ]
                                if name in MODELS[:2] and len(saved):
                                    row.update(saved.iloc[0].to_dict())
                                try:
                                    metrics, states, matrix, predicted, probs = (
                                        evaluate(
                                            model,
                                            frame,
                                            source.transitions,
                                            (
                                                flow_info[name]["mapping"]
                                                if name in FLOW_BASES
                                                else None
                                            ),
                                        )
                                    )
                                    if name in MODELS[:2]:
                                        for metric, value in metrics.items():
                                            if (
                                                len(saved)
                                                and metric in saved.columns
                                                and pd.notna(saved.iloc[0][metric])
                                            ):
                                                delta = value - float(
                                                    saved.iloc[0][metric]
                                                )
                                                passed = bool(
                                                    np.isclose(
                                                        delta, 0, atol=1e-8, rtol=0
                                                    )
                                                )
                                                group["baseline_replay"].append(
                                                    dict(
                                                        identity,
                                                        model=name,
                                                        condition=condition,
                                                        metric=metric,
                                                        delta=delta,
                                                        passed=passed,
                                                    )
                                                )
                                                if not passed:
                                                    raise ValueError(
                                                        f"Baseline replay mismatch: {metric}"
                                                    )
                                    else:
                                        info = flow_info[name]
                                        recovery = info["recovery"]
                                        state_kl = info["state_kl"]
                                        metrics.update(recovery)
                                        attempts = info["attempts"]
                                        chosen = next(
                                            r for r in attempts if r["selected"]
                                        )
                                        row.update(
                                            {
                                                k: chosen[k]
                                                for k in (
                                                    "train_ll",
                                                    "iterations",
                                                    "converged",
                                                    "fit_seconds",
                                                )
                                            }
                                        )
                                        row.update(
                                            selected_start=chosen["start"],
                                            total_fit_seconds=sum(
                                                r["fit_seconds"] for r in attempts
                                            ),
                                            recovery_status=(
                                                "ok" if recovery else "failed"
                                            ),
                                            evaluation_warnings=info["warnings"],
                                        )
                                    row.update(metrics, evaluation_status="ok")
                                    np.savez_compressed(
                                        group_dir
                                        / f"{name.replace(' ', '_')}_{condition}.npz",
                                        predicted=predicted,
                                        probabilities=probs,
                                    )
                                    for values in states:
                                        if (
                                            name in FLOW_BASES
                                            and flow_info[name]["state_kl"]
                                        ):
                                            values["emission_kl"] = state_kl[
                                                values["state"]
                                            ]
                                        elif name in MODELS[:2]:
                                            values["emission_kl"] = source.state_kl[
                                                family,
                                                n_train,
                                                replicate,
                                                name,
                                                values["state"],
                                            ]
                                        group["per_state"].append(
                                            dict(
                                                identity,
                                                model=name,
                                                condition=condition,
                                                **values,
                                            )
                                        )
                                    group["confusion"].extend(
                                        dict(
                                            identity,
                                            model=name,
                                            condition=condition,
                                            true_state=k,
                                            predicted_state=j,
                                            count=int(matrix[k, j]),
                                        )
                                        for k in range(3)
                                        for j in range(3)
                                    )
                                except Exception as exc:
                                    row.update(
                                        evaluation_status="failed",
                                        error=f"{type(exc).__name__}: {exc}",
                                    )
                                group["results"].append(row)
                        write_json(done, group)
                    for name in tables:
                        tables[name].extend(group[name])
                    print(f"Completed {slug}", flush=True)
                    for name, rows in tables.items():
                        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}

def paired_differences(results, metrics, bootstrap_samples=2000, seed=42):
    """Bootstrap replicate-level differences for every pair of emission models."""
    rows = []
    group_keys = ["regime", "family", "n_train", "condition"]
    for identity, frame in results.groupby(group_keys):
        for first, second in itertools.combinations(MODELS, 2):
            for metric in metrics:
                if metric not in frame:
                    continue
                wide = frame.pivot(index="replicate", columns="model", values=metric)
                wide = wide.dropna(subset=list(MODELS))
                delta = (wide[second] - wide[first]).to_numpy()
                if len(delta) == 0:
                    continue
                low = high = np.nan
                if len(delta) >= 2:
                    rng = np.random.default_rng(seed)
                    draws = rng.choice(delta, (bootstrap_samples, len(delta))).mean(1)
                    low, high = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    dict(
                        zip(group_keys, identity),
                        first=first,
                        second=second,
                        metric=metric,
                        n_pairs=len(delta),
                        mean_delta=delta.mean(),
                        ci_low=low,
                        ci_high=high,
                    )
                )
    return pd.DataFrame(rows)

def plot_metrics(results, regime, metrics, *, noise=False):
    """Plot four-model means and sample SD against training size or noise."""
    part = results[results.regime == regime]
    families = list(part.family.unique())
    fig, axes = plt.subplots(
        len(families),
        len(metrics),
        squeeze=False,
        figsize=(4.2 * len(metrics), 2.8 * len(families)),
        layout="constrained",
    )
    conditions = list(part.condition.unique())
    for row, family in enumerate(families):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for name in MODELS:
                frame = part[(part.family == family) & (part.model == name)]
                if noise:
                    frame = frame[frame.n_train == part.n_train.max()]
                    summary = (
                        frame.groupby("condition")[metric]
                        .agg(["mean", "std"])
                        .reindex(conditions)
                    )
                    x = np.arange(len(conditions))
                    ax.set_xticks(x, [c.replace("noise_", "") for c in conditions])
                else:
                    frame = frame[frame.condition == "clean"]
                    summary = frame.groupby("n_train")[metric].agg(["mean", "std"])
                    x = summary.index.to_numpy()
                    ax.set_xticks(x, [f"{n:,}" for n in x])
                    if len(x) > 1:
                        ax.set_xscale("log")
                ax.plot(
                    x,
                    summary["mean"],
                    marker="o",
                    linestyle=STYLES[name],
                    color=COLORS[name],
                    label=name,
                )
                finite = summary["std"].notna().to_numpy()
                if finite.any():
                    ax.errorbar(
                        np.asarray(x)[finite],
                        summary["mean"].to_numpy()[finite],
                        yerr=summary["std"].to_numpy()[finite],
                        fmt="none",
                        color=COLORS[name],
                        capsize=3,
                    )
            ax.set(
                title=f"{family}: {metric}",
                ylabel=metric,
                xlabel=(
                    "Final noise SD / training SD" if noise else "Training observations"
                ),
            )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{regime.capitalize()} regimes · persistence 0.995 · mean ± 1 sample SD"
    )
    return fig

def plot_confusions(confusion, paired, regime):
    """Display pooled counts and row percentages for the same replicate groups."""
    keys = KEYS + ["model", "condition"]
    part = confusion.merge(paired[keys], on=keys, validate="many_to_one")
    part = part[(part.regime == regime) & (part.condition == "clean")]
    part = part[part.n_train == part.n_train.max()]
    families = list(part.family.unique())
    fig, axes = plt.subplots(
        len(families),
        len(MODELS),
        squeeze=False,
        figsize=(16, 3 * len(families)),
        layout="constrained",
    )
    for row, family in enumerate(families):
        for col, name in enumerate(MODELS):
            frame = part[(part.family == family) & (part.model == name)]
            matrix = (
                frame.groupby(["true_state", "predicted_state"])["count"]
                .sum()
                .unstack()
                .to_numpy()
            )
            normalized = matrix / matrix.sum(1, keepdims=True)
            ax = axes[row, col]
            ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for k, j in itertools.product(range(3), repeat=2):
                ax.text(
                    j,
                    k,
                    f"{matrix[k, j]:,}\n{normalized[k, j]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if normalized[k, j] > 0.55 else "black",
                )
            ax.set(
                title=f"{family} · {name}",
                xlabel="Predicted aligned state",
                ylabel="True state",
                xticks=range(3),
                yticks=range(3),
            )
    fig.suptitle(f"{regime.capitalize()} regimes: clean test confusion matrices")
    return fig

def example_models(source, output, family, replicate=0):
    """Recover all four selected models for a fixed illustrative replicate."""
    n_train = max(source.settings["train_sizes"])
    data = source.data(replicate, family)
    group = (
        Path(output)
        / f"{source.regime}_{family.replace(' ', '_')}_{n_train}_{replicate}"
    )
    records = read_json(group / "tables.json")["parameters"]
    models = {
        name: (source.baseline(family, n_train, replicate, name), np.arange(3))
        for name in MODELS[:2]
    }
    for name, distribution in FLOW_BASES.items():
        record = next(r for r in records if r["model"] == name)
        mapping = np.asarray(record["mapping_learned_to_true"])
        models[name] = (
            restore_flow(
                group / f"flow_{distribution}.pt",
                data["train"][["x"]].to_numpy(),
                "cpu",
                "float64",
            ),
            np.argsort(mapping),
        )
    return data, models

def plot_emissions(source, output, family, *, flow_only=False, replicate=0):
    """Overlay true and fitted densities for the requested saved replicate."""
    _, models = example_models(source, output, family, replicate=replicate)
    means, variances = source.true_moments()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    for state, ax in enumerate(axes):
        width = 5 * np.sqrt(variances[state])
        grid = np.linspace(means[state] - width, means[state] + width, 2001)
        ax.plot(
            grid,
            np.exp(source.true_log_density(family, state, grid)),
            color="black",
            label="True density",
        )
        for name in FLOW_BASES if flow_only else MODELS:
            model, order = models[name]
            log_density = emission_model(model).log_prob(grid[:, None])[:, order[state]]
            density = np.exp(log_density)
            ax.plot(
                grid, density, color=COLORS[name], linestyle=STYLES[name], label=name
            )
        ax.set(title=f"State {state}", xlabel="Observation", ylabel="Density")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{source.regime.capitalize()} · {family} · replicate {replicate} · "
        "fitted emissions"
    )
    return fig

def plot_decoding(
    source, output, family="Trimodal", condition="clean", points=5000, *, replicate=0,
):
    """Show observations, truth, and each model's decoded path on identical rows."""
    data = source.data(replicate, family)
    n_train = max(source.settings["train_sizes"])
    frame = source.variants(
        data["train"].iloc[:n_train], data["test"], replicate, family, n_train
    )[condition].iloc[:points]
    fig, axes = plt.subplots(
        len(MODELS) + 2, 1, figsize=(14, 10), sharex=True, layout="constrained"
    )
    positions = np.arange(len(frame))
    for state, color in enumerate(("#4477AA", "#CC6677", "#228833")):
        mask = frame.state.to_numpy() == state
        axes[0].scatter(
            positions[mask],
            frame.x.to_numpy()[mask],
            s=4,
            color=color,
            label=f"State {state}",
        )
    axes[0].legend(ncol=3)
    axes[0].set_ylabel("Observation")
    axes[1].step(positions, frame.state, where="mid", color="black")
    axes[1].set_ylabel("True state")
    for ax, name in zip(axes[2:], MODELS):
        group = (
            Path(output)
            / f"{source.regime}_{family.replace(' ', '_')}_{n_train}_{replicate}"
        )
        with np.load(group / f"{name.replace(' ', '_')}_{condition}.npz") as archive:
            predicted = archive["predicted"][: len(frame)]
        ax.step(positions, predicted, where="mid", color=COLORS[name])
        wrong = predicted != frame.state.to_numpy()
        ax.scatter(positions[wrong], predicted[wrong], s=10, marker="x", color="black")
        ax.set_ylabel(name)
    for ax in axes[1:]:
        ax.set(yticks=range(3), ylim=(-0.3, 2.3))
    axes[-1].set_xlabel("Test observation index")
    fig.suptitle(
        f"{source.regime.capitalize()} · {family} · {condition} · replicate {replicate}"
    )
    return fig

def metric_groups(results, metrics):
    """Require every model to have finite values for the plotted metric set."""
    if any(metric not in results for metric in metrics):
        return results.iloc[:0]
    finite = np.isfinite(results[list(metrics)].to_numpy(dtype=float)).all(1)
    return complete_groups(results[finite], conditions=list(results.condition.unique()))

def plot_differences(differences, regime, metrics):
    """Show paired mean differences and replicate-bootstrap confidence intervals."""
    part = differences[
        (differences.regime == regime) & (differences.condition == "clean")
    ]
    part = part[part.n_train == part.n_train.max()]
    families = list(part.family.unique())
    pairs = list(itertools.combinations(MODELS, 2))
    short = dict(zip(MODELS, ("Gaussian", "GMM", "NSF-N", "NSF-t")))
    fig, axes = plt.subplots(
        len(families),
        len(metrics),
        squeeze=False,
        figsize=(4.5 * len(metrics), 3 * len(families)),
        layout="constrained",
    )
    for row, family in enumerate(families):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for position, (first, second) in enumerate(pairs):
                frame = part[
                    (part.family == family)
                    & (part.metric == metric)
                    & (part["first"] == first)
                    & (part["second"] == second)
                ]
                if frame.empty:
                    continue
                record = frame.iloc[0]
                if np.isfinite(record.ci_low):
                    ax.hlines(
                        position,
                        record.ci_low,
                        record.ci_high,
                        color=COLORS[second],
                        linewidth=2,
                    )
                ax.plot(record.mean_delta, position, "o", color=COLORS[second])
            ax.axvline(0, color=".5", linestyle="--")
            ax.set(
                yticks=range(len(pairs)),
                yticklabels=[f"{short[b]} − {short[a]}" for a, b in pairs],
                title=f"{family}: {metric}",
                xlabel="Paired difference (native units)",
            )
    fig.suptitle(f"{regime.capitalize()} · mean and 95% replicate-bootstrap interval")
    return fig

def plot_per_state(per_state, paired, regime):
    """Compare aligned state precision, recall, and F1 on common replicate groups."""
    keys = KEYS + ["model", "condition"]
    part = per_state.merge(paired[keys], on=keys, validate="many_to_one")
    part = part[(part.regime == regime) & (part.condition == "clean")]
    part = part[part.n_train == part.n_train.max()]
    families = list(part.family.unique())
    fig, axes = plt.subplots(
        len(families),
        3,
        squeeze=False,
        figsize=(13, 3 * len(families)),
        layout="constrained",
    )
    for row, family in enumerate(families):
        for col, metric in enumerate(("precision", "recall", "f1")):
            ax = axes[row, col]
            for offset, name in enumerate(MODELS):
                summary = (
                    part[(part.family == family) & (part.model == name)]
                    .groupby("state")[metric]
                    .agg(["mean", "std"])
                )
                x = summary.index.to_numpy() + (offset - (len(MODELS) - 1) / 2) * 0.08
                ax.plot(x, summary["mean"], "o", color=COLORS[name], label=name)
                finite = summary["std"].notna()
                if finite.any():
                    ax.errorbar(
                        x[finite],
                        summary.loc[finite, "mean"],
                        yerr=summary.loc[finite, "std"],
                        fmt="none",
                        color=COLORS[name],
                        capsize=3,
                    )
            ax.set(
                title=f"{family}: {metric}",
                xticks=range(3),
                xlabel="True aligned state",
                ylabel=metric,
                ylim=(0, 1.02),
            )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{regime.capitalize()} · per-state mean ± 1 sample SD")
    return fig

def gaussian_floors(sources):
    """Compute the clean-target KL to the optimal moment-matched Gaussian."""
    rows = []
    for regime, source in sources.items():
        means, variances = source.true_moments()
        for family in source.families:
            for state in range(3):
                lower = -np.inf
                if family == "Skewed Gamma":
                    lower = means[state] - np.sqrt(
                        variances[state] * source.shape_parameters["gamma_shape"]
                    )

                def integrand(x):
                    """Weight the optimal Gaussian log ratio by true density."""
                    log_p = float(source.true_log_density(family, state, x))
                    if not np.isfinite(log_p) or log_p < -745:
                        return 0.0
                    log_q = stats.norm.logpdf(
                        x, means[state], np.sqrt(variances[state])
                    )
                    return np.exp(log_p) * (log_p - log_q)

                kl, error = integrate.quad(
                    integrand, lower, np.inf, epsabs=1e-9, epsrel=1e-9, limit=500
                )
                rows.append(
                    dict(
                        regime=regime,
                        family=family,
                        state=state,
                        gaussian_kl_floor=kl,
                        quad_error=error,
                    )
                )
    return pd.DataFrame(rows)
