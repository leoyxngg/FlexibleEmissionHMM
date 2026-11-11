"""Extend saved synthetic data and train all four HMM model families."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import torch

from flexible_emission_hmm import (
    BaumWelchTrainer, GaussianEmission, GaussianMixtureEmission, HMM,
    StateParameters,
)
from synthetic_data_generation.markov_process_generation import (
    GammaEmissionGenerator, GaussianEmissionGenerator,
    MultimodalEmissionGenerator, StudentTEmissionGenerator,
)
import nf_hmmlearn_comparison as experiment


MODEL_NAMES = {
    "gaussian": "Gaussian HMM", "gmm": "GMM HMM",
    "normal": "RQ-NSF Gaussian base", "student_t": "RQ-NSF Student-t base",
}
FLOW_BASES = {"normal", "student_t"}
PERSISTENCE = 0.995

def seed_for(master_seed, *parts):
    """Derive a deterministic seed using the saved experiments' recipe."""
    payload = json.dumps([master_seed, *parts]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

def write_json(path, value):
    """Publish a JSON artifact with its parent directory present."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    experiment.write_json(path, value)

def validate_persistence(manifest):
    """Require the original symmetric three-state 0.995 transition matrix."""
    expected = np.full((3, 3), (1 - PERSISTENCE) / 2)
    np.fill_diagonal(expected, PERSISTENCE)
    if not np.allclose(manifest["transitions"], expected, atol=1e-14, rtol=0):
        raise ValueError("The expanded experiment requires persistence 0.995")

def emission_generator(shape, manifest, family, seed, length):
    """Construct emissions from frozen population parameters, not defaults."""
    means = np.asarray(shape["means"])
    std = np.sqrt(shape["variance"])
    parameters = []
    for state, mean in enumerate(means):
        if family == "Gaussian":
            cls, values = GaussianEmissionGenerator, [mean, std]
        elif family == "Student-t":
            cls = StudentTEmissionGenerator
            values = [mean, std, shape["student_t_df"]]
        elif family == "Skewed Gamma":
            cls = GammaEmissionGenerator
            values = [mean, std, shape["gamma_shape"]]
        elif family == "Trimodal":
            cls = MultimodalEmissionGenerator
            values = [
                mean, std, shape["mixture_weights"][state],
                shape["mixture_means"][state], shape["mixture_stds"][state],
            ]
        else:
            raise ValueError(f"Unsupported emission family: {family}")
        parameters.append(values)
    return cls(
        parameters, np.arange(3), np.asarray(manifest["initial"]),
        np.asarray(manifest["transitions"]), length, random_state=seed,
    )

def generate_replicate(manifest, shape, regime, replicate):
    """Generate paired families on independent, reproducible train/test paths."""
    validate_persistence(manifest)
    settings = manifest["settings"]
    result = {family: {} for family in manifest["families"]}
    for split, length in (
        ("train", max(settings["train_sizes"])), ("test", settings["n_test"]),
    ):
        path_seed = seed_for(
            settings["master_seed"],
            "path" if regime == "mean" else "volatility_path", replicate, split,
        )
        generator = emission_generator(
            shape, manifest, "Gaussian", path_seed, length,
        )
        path = generator.sample_state_path()
        if not np.array_equal(np.unique(path), np.arange(3)):
            raise ValueError(
                f"Missing state in {regime} replicate {replicate} {split}; "
                "do not silently resample or change persistence"
            )
        for family in manifest["families"]:
            emission_seed = seed_for(
                settings["master_seed"],
                "observations" if regime == "mean" else "volatility_emissions",
                replicate, family, split,
            )
            generator = emission_generator(
                shape, manifest, family, emission_seed, length,
            )
            values, components = generator.sample_emissions(
                path, return_components=True,
            )
            values = values[:, 0]
            if regime == "volatility":
                means = np.asarray(shape["means"])
                scale = np.asarray(manifest["stds"]) / np.sqrt(shape["variance"])
                values = (
                    (values - means[path]) * scale[path]
                    + np.asarray(manifest["means"])[path]
                )
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite generated observations")
            for column, data in (
                ("time", np.arange(length)), ("state", path),
                ("component", components), ("x", values),
            ):
                result[family][f"{split}__{column}"] = np.asarray(data).copy()
    return result

def prepare_datasets(root, sources, output, *, n_replicates=100, n_starts=100):
    """Preserve original archives and append frozen-parameter replicates."""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.relative_to(root)
    if n_replicates < 1 or n_starts < 1:
        raise ValueError("Replicate and initialization counts must be positive")
    manifests = {
        regime: experiment.read_json(root / relative / "manifest.json")
        for regime, relative in sources.items()
    }
    if set(manifests) != {"mean", "volatility"}:
        raise ValueError("Provide both source regimes")
    shape = manifests["mean"]["distributions"]
    source_hashes, original_indices = {}, {}
    for regime, relative in sources.items():
        manifest = manifests[regime]
        validate_persistence(manifest)
        if n_replicates < manifest["settings"]["n_replicates"]:
            raise ValueError("Keep all original dataset replicates")
        index = experiment.read_json(root / relative / "dataset_index.json")
        original_indices[regime] = {
            (row["replicate"], row["family"]): row["file"] for row in index
        }
        names = {"manifest.json", "dataset_index.json", *(r["file"] for r in index)}
        for name in names:
            path = root / relative / name
            source_hashes[path.relative_to(root).as_posix()] = experiment.digest(path)
    request = {
        "format_version": 1, "n_replicates": n_replicates,
        "n_starts": n_starts, "persistence": PERSISTENCE,
        "sources": sources, "source_hashes": source_hashes,
        "generator_implementation": experiment.digest(__file__),
        "sampling_implementation": experiment.digest(
            root / "src/synthetic_data_generation/markov_process_generation.py"
        ),
        "numpy_version": np.__version__,
        "new_splits": ["train", "test"],
    }
    request_path = output / "data_request.json"
    if request_path.exists() and experiment.read_json(request_path) != request:
        raise ValueError("Dataset request changed; choose a new output directory")
    write_json(request_path, request)
    coverage = []
    for regime, manifest in manifests.items():
        directory = output / regime
        directory.mkdir(exist_ok=True)
        index_path = directory / "dataset_index.json"
        entries = experiment.read_json(index_path) if index_path.exists() else []
        known = {(r["replicate"], r["family"]): r for r in entries}
        original_count = manifest["settings"]["n_replicates"]
        expanded = {
            **manifest,
            "settings": {
                **manifest["settings"], "n_replicates": n_replicates,
                "n_starts": n_starts,
            },
            "dataset_extension": {
                "original_count": original_count,
                "preserved_replicates": list(range(original_count)),
                "generated_replicates": list(range(original_count, n_replicates)),
                "request": "../data_request.json",
            },
        }
        write_json(directory / "manifest.json", expanded)
        for replicate in range(n_replicates):
            generated = None
            for number, family in enumerate(manifest["families"]):
                key = replicate, family
                destination = directory / f"rep_{replicate:03d}_family_{number}.npz"
                if key in known:
                    if experiment.digest(destination) != known[key]["sha256"]:
                        raise ValueError(f"Dataset changed: {destination}")
                else:
                    with tempfile.NamedTemporaryFile(
                        dir=directory, suffix=".npz", delete=False,
                    ) as handle:
                        temporary = Path(handle.name)
                    try:
                        if replicate < original_count:
                            original = original_indices[regime][key]
                            source = root / sources[regime] / original
                            shutil.copy2(source, temporary)
                            origin = "preserved"
                        else:
                            if generated is None:
                                generated = generate_replicate(
                                    manifest, shape, regime, replicate,
                                )
                            np.savez_compressed(temporary, **generated[family])
                            origin = "generated"
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                    known[key] = {
                        "replicate": replicate, "family": family,
                        "file": destination.name, "origin": origin,
                        "sha256": experiment.digest(destination),
                    }
                    write_json(index_path, list(known.values()))
                with np.load(destination, allow_pickle=False) as archive:
                    for split in ("train", "test"):
                        labels = archive[f"{split}__state"]
                        values = archive[f"{split}__x"]
                        if (not np.isfinite(values).all()
                                or not np.array_equal(np.unique(labels), np.arange(3))):
                            raise ValueError(f"Invalid dataset: {destination} {split}")
                        coverage.append({
                            "regime": regime, "family": family,
                            "replicate": replicate, "split": split,
                            "origin": known[key]["origin"], "length": len(labels),
                            "state_visits": np.bincount(labels, minlength=3).tolist(),
                            "switches": int(np.count_nonzero(np.diff(labels))),
                        })
            if replicate % 10 == 9 or replicate == n_replicates - 1:
                print(f"Prepared {regime}: {replicate + 1}/{n_replicates} replicates",
                      flush=True)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    report = {
        "n_replicates": n_replicates,
        "datasets": sum(len(m["families"]) * n_replicates for m in manifests.values()),
        "persistence": PERSISTENCE,
        "sources": {regime: (output / regime).relative_to(root).as_posix()
                    for regime in manifests},
    }
    write_json(output / "data_summary.json", report)
    return report

class DatasetSource(experiment.SavedExperiment):
    """Read extended data without requiring historical baseline fits."""

    def __init__(self, root, regime, relative, shape_parameters):
        """Load frozen data metadata and inherit the existing density methods."""
        self.root, self.regime = Path(root), regime
        self.path = self.root / relative
        self.manifest = experiment.read_json(self.path / "manifest.json")
        validate_persistence(self.manifest)
        self.settings = self.manifest["settings"]
        self.families = tuple(self.manifest["families"])
        self.transitions = np.asarray(self.manifest["transitions"])
        self.shape_parameters = shape_parameters
        self.noise_levels = tuple(self.settings.get("noise_levels", (0.1, 0.25, 0.5)))
        self.index = {
            (row["replicate"], row["family"]): row["file"]
            for row in experiment.read_json(self.path / "dataset_index.json")
        }

def load_sources(root, sources):
    """Load paired extended datasets for fitting and evaluation."""
    shape = experiment.read_json(
        Path(root) / sources["mean"] / "manifest.json"
    )["distributions"]
    return {regime: DatasetSource(root, regime, relative, shape)
            for regime, relative in sources.items()}

def save_baseline(path, model, group):
    """Save baseline parameters without pickling a live model."""
    emission = model.emission
    values = {
        "backend": "flexible_emission_hmm", "format_version": 1,
        "base": group["base"], "settings": group["settings"],
        "start": torch.tensor(model.states.start),
        "transitions": torch.tensor(model.states.transitions),
        "means": torch.tensor(emission.means),
        "covariances": torch.tensor(emission.covariances),
        "n_features": emission.n_features, "history": model.history,
        "converged": model.trainer.converged,
        "iterations": model.trainer.n_iter,
    }
    if group["base"] == "gmm":
        values["weights"] = torch.tensor(emission.mixture_weights)
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(values, temporary)
    temporary.replace(path)

def baseline_model(group, seed):
    """Construct the original Gaussian or GMM training implementation."""
    settings = group["settings"]
    options = {
        "n_states": 3, "min_variance": settings["min_variance"],
        "random_state": seed,
    }
    emission = (
        GaussianEmission(**options) if group["base"] == "gaussian" else
        GaussianMixtureEmission(n_components=settings["n_components"], **options)
    )
    return HMM(emission, trainer=BaumWelchTrainer(
        max_iter=group["max_iter"], tol=group["tolerance"],
    ))

def restore_model(path, train, group):
    """Restore a fitted model without running training or initialization."""
    if group["base"] in FLOW_BASES:
        return experiment.restore_flow(path, train, "cpu", "float64")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("backend") != "flexible_emission_hmm"
            or saved.get("format_version") != 1
            or saved.get("base") != group["base"]
            or saved.get("settings") != group["settings"]):
        raise ValueError("Baseline checkpoint does not match its group")
    model = baseline_model(group, 0)
    model.emission.n_features = saved["n_features"]
    model.emission.means = saved["means"].numpy()
    model.emission.covariances = saved["covariances"].numpy()
    if group["base"] == "gmm":
        model.emission.mixture_weights = saved["weights"].numpy()
    model.states = StateParameters(saved["start"].numpy(), saved["transitions"].numpy())
    model.history = saved["history"]
    model.trainer.converged = saved["converged"]
    model.trainer.n_iter = saved["iterations"]
    model.fitted = True
    return model

def fit_start(job):
    """Train one global initialization and save its outcome and checkpoint."""
    train, group, start, directory = job
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if group["base"] in FLOW_BASES:
        return experiment._fit_single_start((
            train, experiment.FlowSettings(**group["settings"]), start,
            group["max_iter"], group["tolerance"], group["seed_parts"], directory,
        ))
    path = directory / f"start_{start}.json"
    if path.exists():
        return {**experiment.read_json(path), "selected": False}
    seed = experiment.initialization_seed(group["seed_parts"], start)
    row = {"start": start, "seed": seed, "status": "failed", "selected": False}
    began = time.perf_counter()
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = baseline_model(group, seed).fit(train)
        history = np.asarray(model.history)
        if not np.isfinite(history).all():
            raise FloatingPointError("Nonfinite training likelihood")
        row.update(
            status="ok", train_ll=float(history[-1]),
            history=history.tolist(), iterations=model.trainer.n_iter,
            converged=bool(model.trainer.converged),
            iteration_limit_reached=model.trainer.n_iter == group["max_iter"],
            likelihood_decreases=int(np.sum(np.diff(history) < -1e-6)),
            stop_reason="tolerance" if model.trainer.converged else "max_iter",
            final_gain_per_observation=float(np.diff(history)[-1] / len(train)),
        )
        save_baseline(path.with_suffix(".pt"), model, group)
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    row.update(
        fit_seconds=time.perf_counter() - began,
        warnings=[str(item.message) for item in caught],
    )
    write_json(path, row)
    return row
