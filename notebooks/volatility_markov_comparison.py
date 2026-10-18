"""Compare volatility-only Markov regimes on clean and noisy test sequences.

Reuse the reference emission families and metrics without changing their globals.
Importing this module does not generate observations, fit models, or write files.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from flexible_emission_hmm import (
    BaumWelchTrainer, GaussianEmission, GaussianMixtureEmission, HMM,
)
from model_evaluation.synthetic import SyntheticEvaluator
from synthetic_data_generation.markov_process_generation import (
    GaussianEmissionGenerator,
)
import synthetic_emission_comparison as reference


@dataclass(frozen=True)
class VolatilityConfig(reference.ExperimentConfig):
    """Configure volatility regimes, fitting budgets, and test noise strengths."""

    train_sizes: tuple = (20_000,)
    n_replicates: int = 3
    n_starts: int = 3
    max_iter: int = 1000
    state_stds: tuple = (1.0, 2.0, 4.0)
    self_transition: float = 0.98
    noise_levels: tuple = (0.1, 0.25, 0.5)

    def __post_init__(self):
        super().__post_init__()
        if len(self.state_stds) != 3 or not all(
            np.isfinite(value) and value > 0 for value in self.state_stds
        ):
            raise ValueError("Provide three positive finite state standard deviations")
        if not np.isfinite(self.self_transition) or not 0 < self.self_transition < 1:
            raise ValueError("self_transition must lie strictly between zero and one")
        if any(not np.isfinite(value) or value <= 0 for value in self.noise_levels):
            raise ValueError("Noise levels must be positive and finite")
        if tuple(sorted(set(self.noise_levels))) != self.noise_levels:
            raise ValueError("Noise levels must be increasing without duplicates")

    @property
    def transitions(self):
        """Return the symmetric three-state transition matrix."""
        matrix = np.full((3, 3), (1 - self.self_transition) / 2)
        np.fill_diagonal(matrix, self.self_transition)
        return matrix


def generate_datasets(config):
    """Generate shared Markov paths and affine-scaled reference emissions."""
    datasets, coverage = {}, []
    for replicate in range(config.n_replicates):
        paths = {}
        for split, length in (("train", max(config.train_sizes)),
                              ("test", config.n_test)):
            generator = GaussianEmissionGenerator(
                [(0, 1)] * 3, reference.STATES, reference.INITIAL,
                config.transitions, length,
                random_state=reference.seed_for(
                    config, "volatility_path", replicate, split),
            )
            paths[split] = generator.sample_state_path()
        for family in config.families:
            means, variances = reference.true_moments(family)
            scale = np.asarray(config.state_stds) / np.sqrt(variances)
            splits = {}
            for split, path in paths.items():
                values, components = reference.sample_emissions(
                    family, path,
                    reference.seed_for(
                        config, "volatility_emissions", replicate, family, split),
                )
                values = (values[:, 0] - means[path]) * scale[path]
                if not np.isfinite(values).all():
                    raise ValueError("Generated observations must be finite")
                splits[split] = pd.DataFrame(dict(
                    time=np.arange(len(path)), state=path.copy(),
                    component=components, x=values,
                ))
                sizes = config.train_sizes if split == "train" else (config.n_test,)
                for size in sizes:
                    labels = path[:size]
                    if np.unique(labels).size != 3:
                        raise ValueError(
                            f"Missing regime: replicate={replicate}, "
                            f"{split}, n={size}; "
                            "increase sequence length or inspect transition settings"
                        )
                    counts = np.zeros((3, 3), dtype=int)
                    np.add.at(counts, (labels[:-1], labels[1:]), 1)
                    boundaries = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, size]
                    lengths = np.diff(boundaries)
                    run_states = labels[boundaries[:-1]]
                    for state in reference.STATES:
                        durations = lengths[run_states == state]
                        coverage.append(dict(
                            family=family, replicate=replicate, split=split, n=size,
                            state=int(state), visits=int(np.sum(labels == state)),
                            episodes=len(durations),
                            mean_episode_length=durations.mean(),
                            min_episode_length=durations.min(),
                            max_episode_length=durations.max(),
                            switches=int(np.count_nonzero(np.diff(labels))),
                        ))
            datasets[replicate, family] = splits
    return datasets, pd.DataFrame(coverage)


def make_test_variants(config, train, test, replicate, family, n_train):
    """Ramp Gaussian measurement-noise standard deviation from zero to a target.

    Use a linear time ramp and one normal noise vector across severity levels. Test
    labels, ordering, and clean returns remain unchanged. Do not normalize
    noisy test data afterward or use test labels to choose noise strength.
    """
    train_scale = float(train.x.std(ddof=0))
    rng = np.random.default_rng(reference.seed_for(
        config, "measurement_noise", replicate, family, n_train
    ))
    noise = rng.normal(size=len(test))
    ramp = np.linspace(0.0, 1.0, len(test))
    variants = {"clean": test.copy()}
    variants["clean"]["noise"] = 0.0
    variants["clean"]["noise_std"] = 0.0
    for level in config.noise_levels:
        frame = test.copy()
        frame["noise_std"] = level * train_scale * ramp
        frame["noise"] = frame.noise_std * noise
        frame["x"] = frame.x + frame.noise
        variants[f"noise_{level:g}"] = frame
    return variants, train_scale


def perturbed_emission_density(family, state, values, state_std, noise_stds):
    """Approximate the true noisy density averaged over a state's test times.

    Convolve the known clean density with the Gaussian noise-scale mixture.
    Approximate positive noise scales by 128 bins and integrate kernel mass
    over grid cells, treating zero noise as a point mass. Use a padded FFT
    grid without renormalizing away omitted tails. This is a time-averaged
    reference density, not a density estimate from realized noisy returns.
    """
    from scipy.signal import fftconvolve
    from scipy.special import ndtr

    values = np.asarray(values, dtype=float)
    noise_stds = np.asarray(noise_stds, dtype=float)
    if (noise_stds.ndim != 1 or noise_stds.size == 0
            or not np.isfinite(noise_stds).all() or (noise_stds < 0).any()):
        raise ValueError("Provide finite nonnegative noise scales for this state")
    if not np.isfinite(state_std) or state_std <= 0:
        raise ValueError("state_std must be positive and finite")
    original_means, original_variances = reference.true_moments(family)
    scale = state_std / np.sqrt(original_variances[state])
    extent = max(16 * state_std, 8 * noise_stds.max(),
                 np.max(np.abs(values)) + 8 * noise_stds.max())
    grid = np.linspace(-extent, extent, 32_769)
    step = grid[1] - grid[0]
    clean = np.exp(reference.true_log_density(
        family, state, grid / scale + original_means[state]
    )) / scale
    kernel = np.zeros_like(grid)
    kernel[len(grid) // 2] = np.mean(noise_stds == 0)
    positive = noise_stds[noise_stds > 0]
    if positive.size:
        counts, edges = np.histogram(positive, bins=128)
        sums, _ = np.histogram(positive, bins=edges, weights=positive)
        for count, total in zip(counts, sums):
            if count:
                sigma = total / count
                kernel += (count / len(noise_stds)) * (
                    ndtr((grid + step / 2) / sigma)
                    - ndtr((grid - step / 2) / sigma)
                )
    density = fftconvolve(clean, kernel, mode="same")
    return np.maximum(np.interp(values, grid, density), 0.0)


def emission_recovery(config, family, model, order):
    """Compute exact clean-target moments and reference quadrature emission KL.

    Transform fitted densities back into each reference family's coordinates.
    KL is invariant under this invertible affine transformation. These
    parameter metrics concern clean emissions, not noisy test distributions.
    """
    weights, means, variances = reference.fitted_components(model, order)
    fitted_means = np.sum(weights * means, axis=1)
    fitted_variances = np.sum(
        weights * (variances + (means - fitted_means[:, None]) ** 2), axis=1
    )
    original_means, original_variances = reference.true_moments(family)
    scale = np.asarray(config.state_stds) / np.sqrt(original_variances)
    kl = [reference.emission_kl(
        config, family, int(state), weights[state],
        means[state] / scale[state] + original_means[state],
        variances[state] / scale[state] ** 2,
    ) for state in reference.STATES]
    if any(value < -max(1e-6, 10 * error) for value, error in kl):
        raise FloatingPointError("Negative KL beyond quadrature tolerance")
    metrics = dict(
        mean_rmse=float(np.sqrt(np.mean(fitted_means ** 2))),
        variance_rmse=float(np.sqrt(np.mean(
            (fitted_variances - np.asarray(config.state_stds) ** 2) ** 2))),
        emission_kl=float(np.mean([value for value, _ in kl])),
        kl_quad_error_max=float(max(error for _, error in kl)),
    )
    return metrics, kl, weights, means, variances


def run_experiment(config, datasets):
    """Fit on clean training data once and evaluate every paired test variant."""
    results, attempts, per_state = [], [], []
    confusion, parameters, predictions = [], [], []
    examples, variants_saved = {}, {}
    for family in config.families:
        for n_train in config.train_sizes:
            for replicate in range(config.n_replicates):
                splits = datasets[replicate, family]
                train, test = splits["train"].iloc[:n_train], splits["test"]
                x_train, y_train = train[["x"]].to_numpy(), train.state.to_numpy()
                variants, train_scale = make_test_variants(
                    config, train, test, replicate, family, n_train
                )
                variants_saved[replicate, family, n_train] = variants
                for model_name in reference.MODEL_NAMES:
                    identity = dict(family=family, n_train=n_train,
                                    replicate=replicate, model=model_name)
                    best, local = None, []
                    for start in range(config.n_starts):
                        seed = reference.seed_for(
                            config, "volatility_fit", replicate, start)
                        record = dict(identity, start=start, seed=seed, selected=False,
                                      status="failed", converged=False)
                        began, caught = time.perf_counter(), []
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                kwargs = dict(n_states=3, random_state=seed,
                                              min_variance=config.min_variance)
                                emission = (
                                    GaussianEmission(**kwargs)
                                    if model_name == reference.MODEL_NAMES[0]
                                    else GaussianMixtureEmission(
                                        n_components=config.n_components, **kwargs
                                    )
                                )
                                model = HMM(emission, trainer=BaumWelchTrainer(
                                    max_iter=config.max_iter,
                                    tol=n_train * config.tol_per_observation,
                                ))
                                model.fit(x_train)
                            history = np.asarray(model.history)
                            if not np.isfinite(history).all() or history.size == 0:
                                raise FloatingPointError(
                                    "Invalid training likelihood history"
                                )
                            train_ll = float(history[-1])
                            record.update(
                                status="ok", train_ll=train_ll,
                                history=history.tolist(),
                                converged=bool(model.trainer.converged),
                                iterations=model.trainer.n_iter,
                                likelihood_decreases=int(
                                    np.sum(np.diff(history) < -1e-6)),
                            )
                            if best is None or train_ll > best[0]:
                                best = train_ll, model, record
                        except Exception as exc:
                            record["error"] = f"{type(exc).__name__}: {exc}"
                        record.update(fit_seconds=time.perf_counter() - began,
                                      warnings=[str(item.message) for item in caught])
                        local.append(record)
                        attempts.append(record)
                    shared = dict(identity, total_fit_seconds=sum(
                        record["fit_seconds"] for record in local))
                    evaluator, recovery = None, {}
                    setup_error, recovery_warnings = None, []
                    if best is not None:
                        train_ll, model, chosen = best
                        chosen["selected"] = True
                        shared.update(
                            selected_start=chosen["start"], train_ll=train_ll,
                            fit_seconds=chosen["fit_seconds"],
                            iterations=chosen["iterations"],
                            converged=chosen["converged"],
                        )
                        try:
                            evaluator = SyntheticEvaluator(model).fit_alignment(
                                x_train, y_train
                            )
                            order = np.argsort(evaluator.alignment.mapping)
                            with warnings.catch_warnings(
                                record=True
                            ) as recovery_warnings:
                                warnings.simplefilter("always")
                                recovered = emission_recovery(
                                    config, family, model, order
                                )
                            recovery, kl, weights, means, variances = recovered
                            if not np.isfinite(list(recovery.values())).all():
                                raise FloatingPointError(
                                    "Nonfinite clean emission recovery metrics"
                                )
                            parameters.append(dict(
                                identity, weights=weights.tolist(),
                                means=means.tolist(),
                                variances=variances.tolist(),
                                mapping_learned_to_true=(
                                    evaluator.alignment.mapping.tolist()),
                                training_alignment_counts=(
                                    evaluator.alignment.training_counts.tolist()),
                                aligned_transitions=(
                                    evaluator.alignment.align_transitions(
                                        model.states.transitions).tolist()),
                                aligned_initial_probs=(
                                    model.states.start[order].tolist()),
                            ))
                        except Exception as exc:
                            evaluator = None
                            setup_error = f"{type(exc).__name__}: {exc}"
                    for condition, frame in variants.items():
                        row = dict(shared, condition=condition,
                                   noise_std_end=float(frame.noise_std.iloc[-1]),
                                   noise_std_rms=float(np.sqrt(
                                       np.mean(frame.noise_std.to_numpy() ** 2))),
                                   training_scale=train_scale,
                                   evaluation_status=(
                                       "all_fits_failed" if best is None else "failed"),
                                   evaluation_warnings=[
                                       str(item.message) for item in recovery_warnings])
                        if setup_error is not None:
                            row["error"] = setup_error
                        if evaluator is not None:
                            try:
                                truth = frame.state.to_numpy()
                                evaluation = evaluator.evaluate(
                                    frame[["x"]].to_numpy(), truth,
                                    true_transitions=config.transitions,
                                )
                                row.update(evaluation.metrics)
                                row["test_nll"] = -row.pop(
                                    "log_likelihood_per_observation")
                                predicted = evaluation.aligned_states[0]
                                classification = reference.classification_metrics(
                                    truth, predicted
                                )
                                scores, state_scores, matrix = classification
                                row.update(scores, **recovery)
                                probabilities = evaluation.aligned_probabilities[0]
                                true_count = int(np.count_nonzero(np.diff(truth)))
                                inferred_count = int(
                                    np.count_nonzero(np.diff(predicted)))
                                row.update(
                                    mean_true_state_posterior=float(
                                        probabilities[
                                            np.arange(len(truth)), truth].mean()),
                                    true_switches=true_count,
                                    inferred_switches=inferred_count,
                                    switch_count_error=inferred_count - true_count,
                                    absolute_switch_count_error=abs(
                                        inferred_count - true_count),
                                )
                                finite_scores = [row[key] for key in SCORE_METRICS]
                                if not np.isfinite(finite_scores).all():
                                    raise FloatingPointError("Nonfinite test metric")
                                row["evaluation_status"] = "ok"
                                key = dict(identity, condition=condition)
                                per_state.extend(dict(key, **values,
                                    emission_kl=kl[values["state"]][0])
                                                 for values in state_scores)
                                confusion.extend(dict(
                                    key, true_state=int(k), predicted_state=int(j),
                                    count=int(matrix[k, j]),
                                ) for k in reference.STATES for j in reference.STATES)
                                predictions.append((key, predicted, probabilities))
                                if (replicate == 0
                                        and n_train == max(config.train_sizes)):
                                    examples[family, model_name, condition] = dict(
                                        frame=frame, truth=truth, predicted=predicted,
                                        probabilities=probabilities,
                                    )
                            except Exception as exc:
                                row["error"] = f"{type(exc).__name__}: {exc}"
                        results.append(row)
                    print(f"{family} | N={n_train:,} | rep={replicate} | {model_name}",
                          flush=True)
    return dict(
        results=pd.DataFrame(results), attempts=pd.DataFrame(attempts),
        per_state=pd.DataFrame(per_state), confusion=pd.DataFrame(confusion),
        parameters=pd.DataFrame(parameters), predictions=predictions,
        examples=examples, variants=variants_saved,
    )


SCORE_METRICS = [
    metric for metric in reference.SUMMARY_METRICS
    if metric not in ("fit_seconds", "total_fit_seconds")
] + ["mean_true_state_posterior", "true_switches", "inferred_switches",
     "switch_count_error", "absolute_switch_count_error"]


def paired_comparisons(config, results):
    """Match both models and all conditions before computing paired changes."""
    keys = reference.PAIR_KEYS
    successful = results[results.evaluation_status == "ok"].copy()
    expected = len(reference.MODEL_NAMES) * (1 + len(config.noise_levels))
    counts = successful.groupby(keys).size()
    complete = counts[counts == expected].reset_index()[keys]
    paired = successful.merge(complete, on=keys, validate="many_to_one")
    if paired.empty:
        return paired, pd.DataFrame(), pd.DataFrame()
    index = keys + ["model"]
    metrics = [metric for metric in SCORE_METRICS
               if metric not in ("emission_kl", "kl_quad_error_max", "transition_rmse",
                                 "mean_rmse", "variance_rmse")]
    clean = paired[paired.condition == "clean"][index + metrics]
    noisy = paired[paired.condition != "clean"]
    joined = noisy.merge(
        clean, on=index, suffixes=("", "_clean"), validate="many_to_one")
    changes = joined[index + ["condition", "noise_std_end", "noise_std_rms"]].copy()
    for metric in metrics:
        changes[metric] = joined[metric] - joined[f"{metric}_clean"]
    intervals = []
    for (family, size, model, condition), group in changes.groupby(
        ["family", "n_train", "model", "condition"]
    ):
        for metric in metrics:
            delta = group[metric].to_numpy()
            low, high = np.nan, np.nan
            if len(delta) >= 2:
                rng = np.random.default_rng(reference.seed_for(
                    config, "volatility_bootstrap", family, int(size),
                    model, condition, metric
                ))
                samples = rng.choice(
                    delta, size=(config.bootstrap_samples, len(delta)), replace=True
                ).mean(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
            intervals.append(dict(
                family=family, n_train=size, model=model, condition=condition,
                metric=metric, n_pairs=len(delta), mean_delta=float(delta.mean()),
                ci_low=low, ci_high=high,
            ))
    return paired, changes, pd.DataFrame(intervals)


def save_artifacts(project_root, config, datasets, coverage, run):
    """Save isolated volatility results, test variants, predictions, and provenance."""
    root = Path(project_root) / "data/synthetic/volatility_markov_experiments"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(
        prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=root
    ))

    def save_json(name, value):
        (run_dir / name).write_text(
            json.dumps(reference.json_safe(value), indent=2, allow_nan=False) + "\n"
        )

    sources = [Path(__file__), Path(reference.__file__),
               Path(project_root) / "notebooks/gmm_synthetic_experiments.ipynb"]
    sources += sorted((Path(project_root) / "src").rglob("*.py"))
    save_json("manifest.json", dict(
        settings=asdict(config), families=config.families,
        transitions=config.transitions, means=[0, 0, 0],
        stds=config.state_stds, initial=reference.INITIAL,
        selection="Highest finite clean-training LL; earlier start wins ties",
        alignment="Clean-training Hungarian alignment frozen across all test variants",
        perturbation=(
            'Additive Gaussian noise; SD ramps linearly from zero to '
            'level * training_std across the full test path; shared base noise'),
        comparison=(
            'Same clean test path, labels, models, and all-condition '
            'complete pairs'),
        wide_family=(
            'Standardized to zero means and prescribed variances; '
            'original shape retained'),
        kl=(
            'Clean-target emission KL invariant to affine '
            'transformation; unchanged by test noise'),
        noise_nll=(
            'Likelihood under clean fitted model; a robustness score, '
            'not parameter recovery'),
        source_hashes={str(path.relative_to(project_root)):
                       hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sources},
        versions=dict(numpy=np.__version__, pandas=pd.__version__),
    ))
    for name in ("results", "attempts", "per_state", "confusion", "parameters"):
        run[name].to_csv(run_dir / f"{name}.csv", index=False)
        save_json(f"{name}.json", run[name].to_dict("records"))
    coverage.to_csv(run_dir / "coverage.csv", index=False)
    dataset_index, variant_index, prediction_index = [], [], []
    for number, ((replicate, family), splits) in enumerate(datasets.items()):
        filename = f"dataset_{number:04d}.npz"
        np.savez_compressed(run_dir / filename, **{
            f"{split}__{column}": frame[column].to_numpy()
            for split, frame in splits.items() for column in frame.columns
        })
        dataset_index.append(dict(replicate=replicate, family=family, file=filename))
    for number, (key, variants) in enumerate(run["variants"].items()):
        replicate, family, size = key
        filename = f"test_variants_{number:04d}.npz"
        np.savez_compressed(run_dir / filename, **{
            f"{condition}__{column}": frame[column].to_numpy()
            for condition, frame in variants.items() for column in frame.columns
        })
        variant_index.append(dict(replicate=replicate, family=family,
                                  n_train=size, file=filename))
    for number, (key, predicted, probabilities) in enumerate(run["predictions"]):
        filename = f"prediction_{number:05d}.npz"
        np.savez_compressed(run_dir / filename, predicted=predicted,
                            probabilities=probabilities)
        prediction_index.append(dict(key, file=filename))
    save_json("dataset_index.json", dataset_index)
    save_json("variant_index.json", variant_index)
    save_json("prediction_index.json", prediction_index)
    return run_dir
