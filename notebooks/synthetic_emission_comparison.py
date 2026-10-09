"""Support code for gmm_synthetic_experiments.ipynb.

Importing this module does not generate data or fit models. The notebook controls
execution, settings, plots, and exports. Baseline parameters and seed recipes are
copied from gaussian_synthetic_experiments.ipynb; that notebook is never executed.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import scipy
from scipy import integrate, stats
from scipy.special import logsumexp
import sklearn
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
)

from flexible_emission_hmm import (
    HMM, GaussianEmission, GaussianMixtureEmission, BaumWelchTrainer,
)
from model_evaluation.synthetic import SyntheticEvaluator
from model_evaluation.synthetic.state_alignment import STATE_LOG_LOSS_FLOOR
from synthetic_data_generation.markov_process_generation import (
    GaussianEmissionGenerator, StudentTEmissionGenerator,
    MultimodalEmissionGenerator, GammaEmissionGenerator,
)


STATES = np.arange(3)
INITIAL = np.full(3, 1 / 3)
TRANSITIONS = np.full((3, 3), .05)
np.fill_diagonal(TRANSITIONS, .90)
MEANS = np.array([-2., 0., 2.])
T_DF = 2.5
GAMMA_SHAPE = 4
BASE_FAMILIES = ("Gaussian", "Student-t", "Trimodal", "Skewed Gamma")
WIDE_FAMILY = "Wide asymmetric trimodal"
MODEL_NAMES = ("Gaussian HMM", "GMM HMM")
MODEL_COLORS = {"Gaussian HMM": "#4477AA", "GMM HMM": "#CC7722"}
MODEL_STYLES = {"Gaussian HMM": ("o", "--"), "GMM HMM": ("s", "-")}

MIXTURE_WEIGHTS = np.array([[.25, .50, .25], [.20, .45, .35], [.35, .40, .25]])
RAW_MIXTURE_MEANS = np.array([[-3.4, -2.1, -.6], [-1.4, .2, 1.7], [.5, 2.1, 3.5]])
RAW_MIXTURE_STDS = np.array([[.22, .35, .28], [.30, .24, .42], [.38, .25, .32]])
_raw_mean = np.sum(MIXTURE_WEIGHTS * RAW_MIXTURE_MEANS, axis=1)
_raw_std = np.sqrt(np.sum(MIXTURE_WEIGHTS * (
    RAW_MIXTURE_STDS**2 + (RAW_MIXTURE_MEANS - _raw_mean[:, None])**2
), axis=1))
MIXTURE_MEANS = (RAW_MIXTURE_MEANS - _raw_mean[:, None]) / _raw_std[:, None] + MEANS[:, None]
MIXTURE_STDS = RAW_MIXTURE_STDS / _raw_std[:, None]
WIDE_WEIGHTS = np.array([[.20, .30, .50], [.45, .15, .40], [.25, .55, .20]])
WIDE_MEANS = np.array([[-10., -3., 10.], [-7., 3., 16.], [-13., 0., 8.]])
WIDE_STDS = np.array([[.40, .50, .60], [.65, .40, .80], [.50, .70, .45]])


@dataclass(frozen=True)
class ExperimentConfig:
    master_seed: int = 42
    train_sizes: tuple = (1_000, 5_000, 20_000)
    n_test: int = 5_000
    n_density: int = 10_000
    n_replicates: int = 20
    n_starts: int = 10
    n_components: int = 3
    max_iter: int = 300
    tol_per_observation: float = 1e-6
    min_variance: float = 1e-6
    include_wide: bool = True
    quad_epsabs: float = 1e-9
    quad_epsrel: float = 1e-9
    quad_limit: int = 500
    bootstrap_samples: int = 2_000

    def __post_init__(self):
        positive = (*self.train_sizes, self.n_test, self.n_density,
                    self.n_replicates, self.n_starts, self.n_components,
                    self.max_iter, self.quad_limit, self.bootstrap_samples)
        if not self.train_sizes or any(not isinstance(v, int) or v < 1 for v in positive):
            raise ValueError("Lengths, replicate/start/component counts, and limits must be positive integers")
        if tuple(sorted(set(self.train_sizes))) != self.train_sizes:
            raise ValueError("train_sizes must be an increasing tuple without duplicates")
        if min(self.train_sizes) < 3 or self.n_test < 3:
            raise ValueError("Train/test sequences must allow all three states")
        if any(not np.isfinite(v) or v <= 0 for v in (
            self.min_variance, self.quad_epsabs, self.quad_epsrel
        )) or not np.isfinite(self.tol_per_observation) or self.tol_per_observation < 0:
            raise ValueError("Invalid variance floor or fitting/quadrature tolerance")

    @property
    def families(self):
        return BASE_FAMILIES + ((WIDE_FAMILY,) if self.include_wide else ())


def seed_for(config, *parts):
    payload = json.dumps([config.master_seed, *parts]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def mixture_parameters(family):
    if family == "Trimodal":
        return MIXTURE_WEIGHTS, MIXTURE_MEANS, MIXTURE_STDS
    if family == WIDE_FAMILY:
        return WIDE_WEIGHTS, WIDE_MEANS, WIDE_STDS
    raise ValueError(f"Not a mixture family: {family}")


def true_moments(family):
    if family != WIDE_FAMILY:
        return MEANS.copy(), np.ones(3)
    means = np.sum(WIDE_WEIGHTS * WIDE_MEANS, axis=1)
    variances = np.sum(WIDE_WEIGHTS * (WIDE_STDS**2 + (WIDE_MEANS - means[:, None])**2), axis=1)
    return means, variances


def make_generator(family, seed, length):
    parameters = []
    means, variances = true_moments(family)
    for k in STATES:
        mean, std = means[k], np.sqrt(variances[k])
        if family == "Gaussian":
            cls, values = GaussianEmissionGenerator, [mean, std]
        elif family == "Student-t":
            cls, values = StudentTEmissionGenerator, [mean, std, T_DF]
        elif family == "Skewed Gamma":
            cls, values = GammaEmissionGenerator, [mean, std, GAMMA_SHAPE]
        elif family == "Trimodal":
            cls, values = MultimodalEmissionGenerator, [mean, std, MIXTURE_WEIGHTS[k], MIXTURE_MEANS[k], MIXTURE_STDS[k]]
        else:
            raise ValueError(f"Unknown generator family: {family}")
        parameters.append(values)
    return cls(parameters, STATES, INITIAL, TRANSITIONS, length, random_state=seed)


def sample_emissions(family, path, seed):
    if family != WIDE_FAMILY:
        return make_generator(family, seed, len(path)).sample_emissions(path, return_components=True)
    # Same vectorized recipe as the extra experiment in the Gaussian notebook.
    rng = np.random.default_rng(seed)
    thresholds = np.cumsum(WIDE_WEIGHTS[path], axis=1)[:, :-1]
    components = np.sum(rng.random(len(path))[:, None] >= thresholds, axis=1)
    values = rng.normal(WIDE_MEANS[path, components], WIDE_STDS[path, components])
    return values[:, None], components


def true_log_density(family, state, values):
    x = np.asarray(values).reshape(-1)
    if family == "Gaussian":
        return stats.norm.logpdf(x, MEANS[state], 1.)
    if family == "Student-t":
        return stats.t.logpdf(x, T_DF, loc=MEANS[state], scale=np.sqrt((T_DF - 2) / T_DF))
    if family == "Skewed Gamma":
        return stats.gamma.logpdf(x, GAMMA_SHAPE, loc=MEANS[state] - np.sqrt(GAMMA_SHAPE), scale=1 / np.sqrt(GAMMA_SHAPE))
    weights, means, stds = mixture_parameters(family)
    return logsumexp(np.log(weights[state]) + stats.norm.logpdf(x[:, None], means[state], stds[state]), axis=1)


def generate_datasets(config):
    datasets, coverage = {}, []
    for replicate in range(config.n_replicates):
        paths = {
            split: make_generator("Gaussian", seed_for(config, "path", replicate, split), length).sample_state_path()
            for split, length in (("train", max(config.train_sizes)), ("test", config.n_test))
        }
        for family in config.families:
            splits = {}
            for split, path in paths.items():
                X, components = sample_emissions(family, path, seed_for(config, "observations", replicate, family, split))
                if not np.isfinite(X).all():
                    raise ValueError("Generated observations must be finite")
                splits[split] = pd.DataFrame({"time": np.arange(len(path)), "state": path.copy(), "component": components, "x": X[:, 0]})
                for length in (config.train_sizes if split == "train" else (config.n_test,)):
                    labels = path[:length]
                    if np.unique(labels).size != 3:
                        raise ValueError(f"Missing true state: replicate={replicate}, split={split}, n={length}; increase sequence length")
                    counts = np.zeros((3, 3), dtype=int)
                    np.add.at(counts, (labels[:-1], labels[1:]), 1)
                    coverage.append({"family": family, "replicate": replicate, "split": split, "n": length,
                                     "state_visits": np.bincount(labels, minlength=3).tolist(),
                                     "transition_counts": counts.tolist(), "switches": int(np.sum(labels[1:] != labels[:-1]))})
            labels = np.repeat(STATES, config.n_density)
            X, components = sample_emissions(family, labels, seed_for(config, "density", replicate, family))
            splits["density"] = pd.DataFrame({"draw": np.tile(np.arange(config.n_density), 3), "state": labels, "component": components, "x": X[:, 0]})
            datasets[replicate, family] = splits
    return datasets, pd.DataFrame(coverage)


def fitted_components(model, order):
    """Return weights/means/variances, shaped (true state, component), in 1D."""
    emission = model.emission
    if isinstance(emission, GaussianMixtureEmission):
        return (emission.mixture_weights[order], emission.means[order, :, 0],
                emission.covariances[order, :, 0, 0])
    return (np.ones((3, 1)), emission.means[order, 0, None],
            emission.covariances[order, 0, 0, None])


def component_log_density(values, weights, means, variances):
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    return logsumexp(log_weights + stats.norm.logpdf(
        np.asarray(values).reshape(-1, 1), means, np.sqrt(variances)), axis=1)


def emission_kl(config, family, state, weights, means, variances):
    """Deterministic KL(true || fitted), resolving mixture peaks separately."""
    options = dict(epsabs=config.quad_epsabs, epsrel=config.quad_epsrel, limit=config.quad_limit)

    def log_ratio(x):
        return float(true_log_density(family, state, [x])[0] -
                     component_log_density([x], weights, means, variances)[0])

    if family in ("Trimodal", WIDE_FAMILY):
        true_weights, centers, widths = mixture_parameters(family)
        value, error = 0., 0.
        for weight, center, width in zip(true_weights[state], centers[state], widths[state]):
            def integrand(z):
                density = stats.norm.pdf(z)
                return 0. if density == 0 else density * log_ratio(center + width * z)
            part, bound = integrate.quad(integrand, -np.inf, np.inf, **options)
            value += weight * part
            error += weight * bound
        return float(value), float(error)

    def integrand(x):
        p = np.exp(true_log_density(family, state, [x])[0])
        return 0. if p == 0 else p * log_ratio(x)
    lower = MEANS[state] - np.sqrt(GAMMA_SHAPE) if family == "Skewed Gamma" else -np.inf
    value, error = integrate.quad(integrand, lower, np.inf, **options)
    return float(value), float(error)


def classification_metrics(truth, predicted):
    matrix = confusion_matrix(truth, predicted, labels=STATES)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=STATES, zero_division=0)
    scores = {"balanced_accuracy": float(recall.mean()),
              "mcc": float(matthews_corrcoef(truth, predicted))}
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        scores[f"{name}_macro"] = float(values.mean())
        scores[f"{name}_weighted"] = float(np.average(values, weights=support))
    per_state = [{"state": int(k), "precision": precision[k], "recall": recall[k],
                  "f1": f1[k], "support": int(support[k])} for k in STATES]
    true_switch = truth[1:] != truth[:-1]
    predicted_switch = predicted[1:] != predicted[:-1]
    p, r, f, _ = precision_recall_fscore_support(true_switch, predicted_switch, average="binary", zero_division=0)
    scores.update(switch_precision=float(p), switch_recall=float(r), switch_f1=float(f),
                  true_switch_rate=float(true_switch.mean()), predicted_switch_rate=float(predicted_switch.mean()))
    return scores, per_state, matrix


def run_experiment(config, datasets):
    """Fit both model families on identical data; choose starts by training LL only."""
    attempt_rows, result_rows, state_rows, parameter_rows, confusion_rows = [], [], [], [], []
    examples = {}
    for family in config.families:
        for n_train in config.train_sizes:
            for replicate in range(config.n_replicates):
                train = datasets[replicate, family]["train"].iloc[:n_train]
                test = datasets[replicate, family]["test"]
                X_train, y_train = train[["x"]].to_numpy(), train.state.to_numpy()
                X_test, y_test = test[["x"]].to_numpy(), test.state.to_numpy()
                for model_name in MODEL_NAMES:
                    identity = dict(family=family, n_train=n_train, replicate=replicate, model=model_name)
                    best, local_attempts = None, []
                    for start in range(config.n_starts):
                        # Gaussian seeds exactly match the reference notebook. Using
                        # the same seed for GMM does not imply identical initial means.
                        seed = seed_for(config, "fit", replicate, start)
                        record = dict(identity, start=start, seed=seed, status="failed", selected=False,
                                      converged=False, iterations=np.nan, train_ll=np.nan,
                                      likelihood_decreases=0, error=None)
                        began, caught = time.perf_counter(), []
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                kwargs = dict(n_states=3, min_variance=config.min_variance, random_state=seed)
                                emission = (GaussianEmission(**kwargs) if model_name == MODEL_NAMES[0] else
                                            GaussianMixtureEmission(n_components=config.n_components, **kwargs))
                                model = HMM(emission, trainer=BaumWelchTrainer(
                                    max_iter=config.max_iter, tol=n_train * config.tol_per_observation))
                                model.fit(X_train)
                            history = np.asarray(model.history)
                            if not np.isfinite(history).all():
                                raise FloatingPointError("Nonfinite training likelihood history")
                            train_ll = float(history[-1])
                            record.update(status="ok", train_ll=train_ll, converged=bool(model.trainer.converged),
                                          iterations=model.trainer.n_iter, history=history.tolist(),
                                          final_gain_per_observation=float((history[-1] - history[-2]) / n_train),
                                          likelihood_decreases=int(np.sum(np.diff(history) < -1e-6)))
                            if best is None or train_ll > best[0]:
                                best = (train_ll, model, record)  # Earlier start wins exact ties.
                        except Exception as exc:
                            record["error"] = f"{type(exc).__name__}: {exc}"
                        record.update(fit_seconds=time.perf_counter() - began, warnings=[str(w.message) for w in caught])
                        local_attempts.append(record)
                        attempt_rows.append(record)
                    row = dict(identity, evaluation_status="all_fits_failed", converged=False,
                               total_fit_seconds=sum(r["fit_seconds"] for r in local_attempts))
                    if best is not None:
                        train_ll, model, chosen = best
                        chosen["selected"] = True
                        row.update(selected_start=chosen["start"], train_ll=train_ll,
                                   converged=chosen["converged"], iterations=chosen["iterations"],
                                   fit_seconds=chosen["fit_seconds"], evaluation_status="failed")
                        caught_eval = []
                        try:
                            evaluator = SyntheticEvaluator(model).fit_alignment(X_train, y_train)
                            evaluation = evaluator.evaluate(X_test, y_test, true_transitions=TRANSITIONS)
                            row.update(evaluation.metrics)
                            row["test_nll"] = -row.pop("log_likelihood_per_observation")
                            extra, per_state, matrix = classification_metrics(y_test, evaluation.aligned_states[0])
                            row.update(extra)
                            order = np.argsort(evaluation.state_mapping)
                            weights, means, variances = fitted_components(model, order)
                            fitted_means = np.sum(weights * means, axis=1)
                            fitted_variances = np.sum(weights * (variances + (means - fitted_means[:, None])**2), axis=1)
                            true_means, true_variances = true_moments(family)
                            row.update(mean_rmse=float(np.sqrt(np.mean((fitted_means - true_means)**2))),
                                       variance_rmse=float(np.sqrt(np.mean((fitted_variances - true_variances)**2))))
                            with warnings.catch_warnings(record=True) as caught_eval:
                                warnings.simplefilter("always")
                                kl = [emission_kl(config, family, k, weights[k], means[k], variances[k]) for k in STATES]
                            row.update(emission_kl=float(np.mean([v for v, _ in kl])),
                                       kl_quad_error_max=float(max(e for _, e in kl)))
                            if not np.isfinite(list(extra.values()) + [row[m] for m in (
                                "test_nll", "emission_kl", "kl_quad_error_max", "mean_rmse", "variance_rmse")]).all():
                                raise FloatingPointError("Nonfinite evaluation metric")
                            if any(value < -max(1e-6, 10 * error) for value, error in kl):
                                raise FloatingPointError("Negative KL beyond quadrature tolerance")
                            row["evaluation_status"] = "ok"
                            state_rows.extend(dict(identity, **values, emission_kl=kl[values["state"]][0],
                                                   kl_quad_error=kl[values["state"]][1]) for values in per_state)
                            confusion_rows.extend(dict(identity, true_state=int(k), predicted_state=int(j),
                                                       count=int(matrix[k, j])) for k in STATES for j in STATES)
                            parameter_rows.append(dict(identity, mapping_learned_to_true=evaluation.state_mapping.tolist(),
                                training_alignment_counts=evaluator.alignment.training_counts.tolist(),
                                weights=weights.tolist(), means=means.tolist(), variances=variances.tolist(),
                                aligned_transitions=evaluation.aligned_transitions.tolist(),
                                aligned_initial_probs=model.states.start[order].tolist()))
                            if replicate == 0 and n_train == max(config.train_sizes):
                                examples[family, model_name] = dict(model=model, evaluation=evaluation)
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        row["evaluation_warnings"] = [str(w.message) for w in caught_eval]
                    result_rows.append(row)
                    print(f"{family} | N={n_train:,} | rep={replicate:02d} | {model_name}: "
                          f"{row['evaluation_status']}, converged={row['converged']}", flush=True)
    return dict(results=pd.DataFrame(result_rows), attempts=pd.DataFrame(attempt_rows),
                per_state=pd.DataFrame(state_rows), parameters=pd.DataFrame(parameter_rows),
                confusion=pd.DataFrame(confusion_rows), examples=examples)


PAIR_KEYS = ["family", "n_train", "replicate"]
SUMMARY_METRICS = [
    "aligned_accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted", "balanced_accuracy",
    "mcc", "ari", "nmi", "switch_precision", "switch_recall", "switch_f1",
    "true_switch_rate", "predicted_switch_rate", "state_log_loss", "state_brier_score",
    "state_log_loss_capped_fraction", "state_zero_probability_fraction",
    "test_nll", "emission_kl", "transition_rmse", "mean_rmse", "variance_rmse",
    "fit_seconds", "total_fit_seconds", "kl_quad_error_max",
]


def paired_results(results):
    successful = results[results.evaluation_status == "ok"].copy()
    counts = successful.groupby(PAIR_KEYS).model.nunique()
    keys = counts[counts == len(MODEL_NAMES)].reset_index()[PAIR_KEYS]
    return successful.merge(keys, on=PAIR_KEYS, validate="many_to_one")


def paired_differences(config, paired, metrics):
    """Bootstrap complete replicate pairs, never individual time points."""
    rows = []
    for (family, n_train), group in paired.groupby(["family", "n_train"]):
        for metric in metrics:
            wide = group.pivot(index="replicate", columns="model", values=metric)
            delta = (wide[MODEL_NAMES[1]] - wide[MODEL_NAMES[0]]).to_numpy()
            low, high = np.nan, np.nan
            if len(delta) >= 2:
                rng = np.random.default_rng(seed_for(config, "bootstrap", family, int(n_train), metric))
                samples = rng.choice(delta, size=(config.bootstrap_samples, len(delta)), replace=True).mean(axis=1)
                low, high = np.quantile(samples, [.025, .975])
            rows.append(dict(family=family, n_train=n_train, metric=metric, n_pairs=len(delta),
                             mean_delta=delta.mean(), ci_low=low, ci_high=high))
    return pd.DataFrame(rows)


def plot_learning_curves(config, paired, metrics):
    fig, axes = plt.subplots(len(config.families), len(metrics),
                             figsize=(4.4 * len(metrics), 3.0 * len(config.families)),
                             squeeze=False, layout="constrained")
    for row, family in enumerate(config.families):
        for column, (metric, label, rate) in enumerate(metrics):
            ax = axes[row, column]
            for model in MODEL_NAMES:
                group = paired[(paired.family == family) & (paired.model == model)]
                summary = group.groupby("n_train")[metric].agg(["mean", "std", "count"]).reindex(config.train_sizes)
                marker, linestyle = MODEL_STYLES[model]
                ax.plot(config.train_sizes, summary["mean"], marker=marker, linestyle=linestyle,
                        color=MODEL_COLORS[model], label=model)
                ax.errorbar(config.train_sizes, summary["mean"], yerr=summary["std"],
                            fmt="none", capsize=3, color=MODEL_COLORS[model])
            ax.set(xscale="log", xlabel="Training observations", ylabel=label, title=f"{family}\n{label}")
            ax.set_xticks(config.train_sizes, [f"{n:,}" for n in config.train_sizes])
            ax.minorticks_off()
            if rate:
                ax.set_ylim(0, 1.02)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend()
    fig.suptitle("Paired successful replicates: mean ± 1 sample SD (not a confidence interval)")
    return fig


def plot_confusions(config, confusion, paired, n_train):
    keys = paired[PAIR_KEYS + ["model"]]
    available = confusion.merge(keys, on=PAIR_KEYS + ["model"], validate="many_to_one")
    fig, axes = plt.subplots(len(config.families), 2, figsize=(9, 3.5 * len(config.families)),
                             squeeze=False, layout="constrained")
    for row, family in enumerate(config.families):
        for col, model in enumerate(MODEL_NAMES):
            ax = axes[row, col]
            part = available[(available.family == family) & (available.model == model) & (available.n_train == n_train)]
            ax.grid(False)
            ax.set(title=f"{family} · {model}", xlabel="Predicted aligned state", ylabel="True state")
            if part.empty:
                ax.text(.5, .5, "No complete pairs", ha="center", transform=ax.transAxes)
                continue
            matrix = part.groupby(["true_state", "predicted_state"])["count"].sum().unstack().reindex(index=STATES, columns=STATES, fill_value=0).to_numpy()
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for k in STATES:
                for j in STATES:
                    ax.text(j, k, f"{matrix[k, j]:,}\n{normalized[k, j]:.1%}", ha="center", va="center",
                            color="white" if normalized[k, j] > .55 else "black")
            ax.set_xticks(STATES)
            ax.set_yticks(STATES)
            ax.set_title(f"{family} · {model}\n{part.replicate.nunique()} paired replicates; {matrix.sum():,} observations")
            fig.colorbar(im, ax=ax, fraction=.046, label="Within-true-state proportion", format=PercentFormatter(1))
    fig.suptitle(f"Held-out confusion matrices · N train = {n_train:,}\nPooled counts and row percentages; fixed training-label alignment")
    return fig


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def save_artifacts(project_root, config, datasets, coverage, run):
    root = Path(project_root) / "data/synthetic/gmm_synthetic_experiments"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=root))

    def save_json(name, value):
        (run_dir / name).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")

    sources = [Path(__file__), Path(project_root) / "notebooks/gmm_synthetic_experiments.ipynb",
               Path(project_root) / "notebooks/gaussian_synthetic_experiments.ipynb"]
    sources += sorted((Path(project_root) / "src").rglob("*.py"))
    manifest = dict(settings=asdict(config), families=config.families, models=MODEL_NAMES,
                    initial=INITIAL, transitions=TRANSITIONS,
                    baseline_source="notebooks/gaussian_synthetic_experiments.ipynb",
                    seed_recipe="First 4 SHA256 bytes, little endian, of json.dumps([master_seed, *parts])",
                    selection="Highest finite training LL; earlier start breaks ties; no test selection",
                    alignment="Hungarian assignment on training Viterbi labels, frozen for test",
                    comparison="Only complete successful Gaussian/GMM replicate pairs",
                    state_log_loss_floor=STATE_LOG_LOSS_FLOOR,
                    zero_division="Precision, recall, and F1 use zero_division=0; test labels cover all states",
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                  scipy=scipy.__version__, sklearn=sklearn.__version__),
                    source_hashes={str(p.relative_to(project_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    distributions=dict(means=MEANS, variance=1., student_t_df=T_DF, gamma_shape=GAMMA_SHAPE,
                                       mixture_weights=MIXTURE_WEIGHTS, mixture_means=MIXTURE_MEANS,
                                       mixture_stds=MIXTURE_STDS, wide_weights=WIDE_WEIGHTS,
                                       wide_means=WIDE_MEANS, wide_stds=WIDE_STDS))
    save_json("manifest.json", manifest)
    for name in ("results", "attempts", "per_state", "parameters", "confusion"):
        run[name].to_csv(run_dir / f"{name}.csv", index=False)
        save_json(f"{name}.json", run[name].to_dict("records"))
    save_json("coverage.json", coverage.to_dict("records"))
    index = []
    for number, ((replicate, family), splits) in enumerate(datasets.items()):
        filename = f"dataset_{number:04d}.npz"
        np.savez_compressed(run_dir / filename, **{
            f"{split}__{column}": frame[column].to_numpy()
            for split, frame in splits.items() for column in frame.columns})
        index.append(dict(replicate=replicate, family=family, file=filename))
    save_json("dataset_index.json", index)
    # Save example aligned predictions/posteriors without pickling live models.
    for number, ((family, model), example) in enumerate(run["examples"].items()):
        evaluation = example["evaluation"]
        filename = f"example_{number:03d}.npz"
        np.savez_compressed(run_dir / filename, family=family, model=model,
                            replicate=0, n_train=max(config.train_sizes),
                            truth=datasets[0, family]["test"].state.to_numpy(),
                            predicted=evaluation.aligned_states[0], probabilities=evaluation.aligned_probabilities[0])
    return run_dir
