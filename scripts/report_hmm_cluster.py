import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "cluster_runs/hmm_100x100/comparison"
GROUP_KEYS = ["regime", "family", "n_train", "model"]
PLOT_SETS = {
    "classification": [
        "aligned_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ],
    "recovery": ["test_nll", "emission_kl", "state_brier_score", "transition_rmse"],
    "parameters": ["mean_rmse", "variance_rmse", "state_log_loss", "switch_f1"],
}
NOISE_METRICS = [
    "f1_macro",
    "state_brier_score",
    "test_nll",
    "absolute_switch_count_error",
]
DIFFERENCE_METRICS = ["f1_macro", "test_nll", "emission_kl", "total_fit_seconds"]

def runtime():
    """Load the original plotting implementation without starting experiments."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    for directory in [ROOT / "src", ROOT / "notebooks"]:
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import hmm_cluster_experiment as implementation

    return implementation

def read_json(path):
    """Read a saved JSON document."""
    return json.loads(Path(path).read_text())

def digest(path):
    """Hash an input table for the report provenance record."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

class SavedRun:
    """Restore all four retrained model families in the saved state alignment."""

    def __init__(self, directory, plan, implementation):
        self.directory = Path(directory)
        self.plan = plan
        self.implementation = implementation
        self.groups = {
            (g["regime"], g["family"], g["n_train"], g["replicate"], g["model"]): g
            for g in plan["groups"]
        }

    def condition_directory(self, source, family, replicate):
        """Locate the evaluated checkpoint directory for one dataset."""
        size = max(source.settings["train_sizes"])
        return (
            self.directory
            / f"{source.regime}_{family.replace(' ', '_')}_{size}_{replicate}"
        )

    def example_models(self, source, output, family, replicate=0):
        """Load current checkpoints and use saved alignment for every model."""
        import numpy as np

        directory = self.condition_directory(source, family, replicate)
        rows = read_json(directory / "tables.json")["parameters"]
        records = {row["model"]: row for row in rows}
        data = source.data(replicate, family)
        size = max(source.settings["train_sizes"])
        train = data["train"][["x"]].to_numpy()[:size]
        models = {}
        for name in self.implementation.MODEL_NAMES.values():
            group = self.groups[source.regime, family, size, replicate, name]
            record = records[name]
            mapping = np.asarray(record["mapping_learned_to_true"], dtype=int)
            if not np.array_equal(np.sort(mapping), np.arange(3)):
                raise ValueError(f"Invalid saved state alignment: {directory}/{name}")
            checkpoint = directory / f"model_{group['base']}.pt"
            model = self.implementation.restore_model(checkpoint, train, group)
            models[name] = model, np.argsort(mapping)
        return data, models

    @contextmanager
    def emission_plot_adapter(self):
        """Adapt the original density plotter without altering frozen source files."""
        experiment = self.implementation.experiment
        with patch.object(experiment, "example_models", self.example_models):
            yield

    def plot_decoding(self, source, family, condition, replicate, points):
        """Plot saved decoded paths using the current prediction filenames."""
        import matplotlib.pyplot as plt
        import numpy as np

        experiment = self.implementation.experiment
        data = source.data(replicate, family)
        size = max(source.settings["train_sizes"])
        frame = source.variants(
            data["train"].iloc[:size], data["test"], replicate, family, size
        )[condition].iloc[:points]
        directory = self.condition_directory(source, family, replicate)
        fig, axes = plt.subplots(
            6, 1, figsize=(14, 10), sharex=True, layout="constrained"
        )
        positions = np.arange(len(frame))
        truth = frame.state.to_numpy()
        for state, color in enumerate(("#4477AA", "#CC6677", "#228833")):
            mask = truth == state
            axes[0].scatter(
                positions[mask],
                frame.x.to_numpy()[mask],
                s=4,
                color=color,
                label=f"State {state}",
            )
        axes[0].legend(ncol=3)
        axes[0].set_ylabel("Observation")
        axes[1].step(positions, truth, where="mid", color="black")
        axes[1].set_ylabel("True state")
        for axis, (base, name) in zip(
            axes[2:], self.implementation.MODEL_NAMES.items()
        ):
            with np.load(
                directory / f"{base}_{condition}.npz", allow_pickle=False
            ) as saved:
                predicted = saved["predicted"][: len(frame)]
            if len(predicted) != len(frame):
                raise ValueError("Saved predictions do not match test observations")
            axis.step(positions, predicted, where="mid", color=experiment.COLORS[name])
            wrong = predicted != truth
            axis.scatter(
                positions[wrong], predicted[wrong], s=10, marker="x", color="black"
            )
            axis.set_ylabel(name)
        for axis in axes[1:]:
            axis.set(yticks=range(3), ylim=(-0.3, 2.3))
        axes[-1].set_xlabel("Test observation index")
        fig.suptitle(
            f"{source.regime.capitalize()} · {family} · {condition} · replicate {replicate}"
        )
        return fig

def representative_volatility(results, experiment):
    """Select the original notebook's median-distance representative replicate."""
    import numpy as np
    import pandas as pd

    metrics = [
        "f1_macro",
        "emission_kl",
        "transition_rmse",
        "absolute_switch_count_error",
    ]
    part = results[
        (results.regime == "volatility")
        & (results.condition == "clean")
        & (results.evaluation_status == "ok")
    ]
    if part.empty or any(metric not in part for metric in metrics):
        return None, pd.DataFrame()
    part = part[part.n_train == part.n_train.max()]
    families = list(results.loc[results.regime == "volatility", "family"].unique())
    columns = pd.MultiIndex.from_product([metrics, families, experiment.MODELS])
    features = (
        part.pivot(index="replicate", columns=["family", "model"], values=metrics)
        .reindex(columns=columns)
        .sort_index()
    )
    features = features[np.isfinite(features.to_numpy()).all(axis=1)]
    if len(features) < 2:
        return None, pd.DataFrame()
    centers = features.median()
    scales = features.quantile(0.75) - features.quantile(0.25)
    fallback = (features.max() - features.min()).replace(0, 1.0)
    scales = scales.mask(scales.eq(0), fallback)
    scores = features.sub(centers).abs().div(scales).mean(axis=1)
    return int(scores.idxmin()), scores.rename("representativeness_score").reset_index()

def export_report(
    directory,
    output,
    plan,
    *,
    examples=True,
    replicate=0,
    points=5000,
    bootstrap_samples=2000,
):
    """Export summaries and original-style plots from completed evaluations."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    implementation = runtime()
    experiment = implementation.experiment
    directory, output = Path(directory), Path(output)
    names = ["results", "attempts", "per_state", "confusion", "parameters", "coverage"]
    missing = [name for name in names if not (directory / f"{name}.csv").exists()]
    if missing or not (directory / "manifest.json").exists():
        raise FileNotFoundError(
            "Finish hmm_cluster.py evaluate first; missing tables: "
            + ", ".join(missing)
        )
    manifest = read_json(directory / "manifest.json")
    if manifest["plan_id"] != plan["id"]:
        raise ValueError("Evaluation tables belong to a different plan")
    tables = {name: pd.read_csv(directory / f"{name}.csv") for name in names}
    results, attempts = tables["results"], tables["attempts"]
    identities = experiment.KEYS + ["model", "condition"]
    if results.duplicated(identities).any():
        raise ValueError("Duplicate evaluation identities")
    sources = implementation.load_sources(ROOT, plan["sources"])
    expected = set()
    for group in plan["groups"]:
        conditions = ["clean"] + [
            f"noise_{level}" for level in sources[group["regime"]].noise_levels
        ]
        for condition in conditions:
            expected.add(
                tuple(group[key] for key in experiment.KEYS)
                + (group["model"], condition)
            )
    if set(results[identities].itertuples(index=False, name=None)) != expected:
        raise ValueError("Evaluation tables do not cover the complete frozen plan")
    attempt_keys = experiment.KEYS + ["model", "start"]
    if attempts.duplicated(attempt_keys).any():
        raise ValueError("Duplicate initialization records in attempts.csv")
    start_values = attempts.start.to_numpy()
    starts = plan["n_starts"]
    if not (
        np.isfinite(start_values).all()
        and (start_values == np.floor(start_values)).all()
        and (start_values >= 0).all()
        and (start_values < starts).all()
    ):
        raise ValueError("Invalid initialization indices in attempts.csv")
    expected_groups = {
        tuple(group[key] for key in experiment.KEYS) + (group["model"],)
        for group in plan["groups"]
    }
    attempt_counts = attempts.groupby(experiment.KEYS + ["model"]).size()
    if (
        set(attempt_counts.index) != expected_groups
        or not attempt_counts.eq(starts).all()
    ):
        raise ValueError("Initialization records do not cover the full frozen budget")
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    coverage = []
    notes = []

    def save(fig, name):
        handles, labels = [], []
        for axis in fig.axes:
            legend = axis.get_legend()
            if legend is None:
                continue
            axis_handles, axis_labels = axis.get_legend_handles_labels()
            for handle, label in zip(axis_handles, axis_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            legend.remove()
        if labels:
            fig.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=min(5, len(labels)),
                fontsize=8,
                frameon=False,
            )
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(name + ".png")

    def paired_plot(data, regime, metrics, name, noise=False):
        metrics = [metric for metric in metrics if metric in data]
        selected = experiment.metric_groups(data, metrics) if metrics else data.iloc[:0]
        selected = selected[selected.regime == regime]
        coverage.append(
            {
                "figure": name,
                "metrics": ",".join(metrics),
                "paired_datasets": len(selected[experiment.KEYS].drop_duplicates()),
            }
        )
        if selected.empty:
            notes.append(f"Skipped {name}: no complete finite model comparisons.")
            return
        save(experiment.plot_metrics(selected, regime, metrics, noise=noise), name)

    numeric = [metric for metric in experiment.METRICS if metric in results]
    results[numeric] = results[numeric].replace([np.inf, -np.inf], np.nan)
    clean = experiment.complete_groups(results, ["clean"])
    robust = experiment.complete_groups(results)
    clean.groupby(GROUP_KEYS)[numeric].agg(["mean", "std", "count"]).to_csv(
        output / "clean_metric_summary.csv"
    )
    robust.groupby(GROUP_KEYS + ["condition"])[numeric].agg(
        ["mean", "std", "count"]
    ).to_csv(output / "noise_robustness_summary.csv")
    results.assign(ok=results.evaluation_status.eq("ok")).groupby(
        GROUP_KEYS + ["condition"]
    ).agg(evaluations=("ok", "size"), successful=("ok", "sum")).to_csv(
        output / "evaluation_coverage.csv"
    )
    results[results.evaluation_status != "ok"].to_csv(
        output / "evaluation_errors.csv", index=False
    )
    diagnostics = [
        column
        for column in [
            *identities,
            "recovery_status",
            "kl_quad_error_max",
            "moment_quad_error_max",
            "evaluation_warnings",
        ]
        if column in results
    ]
    results[diagnostics].to_csv(output / "recovery_diagnostics.csv", index=False)
    diagnostic_rows = attempts.assign(
        failed=attempts.status.ne("ok"),
        converged_ok=attempts.converged.isin([True, "True", 1]),
    )
    diagnostic_rows.groupby(GROUP_KEYS).agg(
        starts=("status", "size"),
        failures=("failed", "sum"),
        converged_starts=("converged_ok", "sum"),
        mean_iterations=("iterations", "mean"),
        total_fit_seconds=("fit_seconds", "sum"),
        likelihood_decreases=("likelihood_decreases", "sum"),
    ).to_csv(output / "fit_diagnostics.csv")
    keys = experiment.KEYS + ["model", "condition"]
    tables["per_state"].merge(clean[keys], on=keys, validate="many_to_one").groupby(
        GROUP_KEYS + ["state"]
    )[["precision", "recall", "f1", "support"]].agg(["mean", "std"]).to_csv(
        output / "per_state_summary.csv"
    )
    for label, data, metrics in [
        ("paired_model_differences", clean, DIFFERENCE_METRICS),
        ("noise_paired_model_differences", robust, NOISE_METRICS),
    ]:
        frames = [
            experiment.paired_differences(
                experiment.metric_groups(data, [metric]),
                [metric],
                bootstrap_samples=bootstrap_samples,
            )
            for metric in metrics
            if metric in data
        ]
        differences = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        differences.to_csv(output / f"{label}.csv", index=False)
        if label == "paired_model_differences" and not differences.empty:
            for regime in sources:
                if not differences[differences.regime == regime].empty:
                    available = [m for m in metrics if m in differences.metric.unique()]
                    save(
                        experiment.plot_differences(differences, regime, available),
                        f"{regime}_{label}",
                    )
    noise_metrics = [metric for metric in NOISE_METRICS if metric in robust]
    clean_scores = robust[robust.condition == "clean"]
    deltas = robust.merge(
        clean_scores[experiment.KEYS + ["model"] + noise_metrics],
        on=experiment.KEYS + ["model"],
        suffixes=("", "_clean"),
        validate="many_to_one",
    )
    for metric in noise_metrics:
        deltas[f"{metric}_noisy_minus_clean"] = (
            deltas[metric] - deltas[f"{metric}_clean"]
        )
    deltas.to_csv(output / "noisy_minus_clean.csv", index=False)
    for regime in sources:
        for label, metrics in PLOT_SETS.items():
            paired_plot(clean, regime, metrics, f"{regime}_{label}")
        paired_plot(
            robust, regime, NOISE_METRICS, f"{regime}_noise_robustness", noise=True
        )
        if not clean[clean.regime == regime].empty:
            save(
                experiment.plot_confusions(tables["confusion"], clean, regime),
                f"{regime}_confusion_matrices",
            )
            save(
                experiment.plot_per_state(tables["per_state"], clean, regime),
                f"{regime}_per_state_classification",
            )
    experiment.gaussian_floors(sources).to_csv(
        output / "gaussian_kl_floors.csv", index=False
    )

    saved = SavedRun(directory, plan, implementation)
    fig, axes = plt.subplots(
        1, len(sources), squeeze=False, figsize=(13, 4), layout="constrained"
    )
    for axis, regime in zip(axes[0], sources):
        for base, name in implementation.MODEL_NAMES.items():
            chosen = attempts[
                (attempts.regime == regime)
                & (attempts.family == "Trimodal")
                & (attempts.replicate == replicate)
                & (attempts.model == name)
                & attempts.selected.isin([True, "True", 1])
            ]
            if chosen.empty:
                continue
            row = chosen.iloc[0]
            folder = directory / f"{regime}_Trimodal_{int(row.n_train)}_{replicate}"
            record = read_json(
                folder / f"starts_{base}" / f"start_{int(row.start)}.json"
            )
            history = np.asarray(record["history"]) / row.n_train
            axis.plot(
                np.arange(len(history)),
                history,
                label=name,
                color=experiment.COLORS[name],
                linestyle=experiment.STYLES[name],
            )
        axis.set(
            title=f"{regime}: Trimodal, replicate {replicate}",
            xlabel="Outer EM update",
            ylabel="Training log likelihood / observation",
        )
    axes[0, 0].legend(fontsize=8)
    save(fig, "selected_training_histories")

    representative, scores = representative_volatility(results, experiment)
    scores.to_csv(output / "representative_volatility_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "replicate": representative,
                "method": "median-distance across families and models",
            }
        ]
    ).to_csv(output / "representative_volatility_replicates.csv", index=False)
    if examples:
        with saved.emission_plot_adapter():
            for regime, source in sources.items():
                for family in source.families:
                    for flow_only in [False, True]:
                        suffix = "flow_emissions" if flow_only else "fitted_emissions"
                        save(
                            experiment.plot_emissions(
                                source,
                                directory,
                                family,
                                flow_only=flow_only,
                                replicate=replicate,
                            ),
                            f"{regime}_{family}_replicate_{replicate}_{suffix}",
                        )
                for condition in ["clean", "noise_0.5"]:
                    if condition in results.condition.unique():
                        save(
                            saved.plot_decoding(
                                source, "Trimodal", condition, replicate, points
                            ),
                            f"{regime}_Trimodal_{condition}_replicate_{replicate}_decoded_paths",
                        )
            if representative is not None:
                source = sources["volatility"]
                for family in source.families:
                    save(
                        experiment.plot_emissions(
                            source,
                            directory,
                            family,
                            flow_only=True,
                            replicate=representative,
                        ),
                        f"volatility_{family}_replicate_{representative}_representative_flow_emissions",
                    )
                    save(
                        saved.plot_decoding(
                            source, family, "clean", representative, points
                        ),
                        f"volatility_{family}_replicate_{representative}_representative_decoded_paths",
                    )
    pd.DataFrame(coverage).to_csv(output / "plot_coverage.csv", index=False)
    notes.extend(
        [
            "Metric error bars are mean +/- one sample SD across dataset replicates, matching the original notebook.",
            "Paired-difference intervals use replicate-level bootstrap samples, not individual observations.",
            "Representative replicate selection is descriptive and uses the original notebook's outcome-based rule.",
            "The separate old 30-initialization diagnostic is not rerun or silently mixed into this experiment.",
        ]
    )
    if len({group["n_train"] for group in plan["groups"]}) == 1:
        notes.append(
            "This run has one training size; training-size plots contain one point per model, not a multi-size learning curve."
        )
    (output / "README.txt").write_text("\n".join(notes) + "\n")
    provenance = {
        "plan_id": plan["id"],
        "input": str(directory.resolve()),
        "input_hashes": {
            f"{name}.csv": digest(directory / f"{name}.csv") for name in names
        },
        "evaluation_rows": len(results),
        "attempt_rows": len(attempts),
        "figures": figures,
        "notes": notes,
        "bootstrap_samples": bootstrap_samples,
        "illustrative_replicate": replicate,
    }
    (output / "report_manifest.json").write_text(json.dumps(provenance, indent=2))
    return {
        "figures": len(figures),
        "output": str(output),
        "evaluation_rows": len(results),
        "attempt_rows": len(attempts),
    }

def main(argv=None):
    """Export reports only after the cluster evaluation tables are complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "cluster_runs/hmm_100x100/plan.json"
    )
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--points", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--no-examples", action="store_true")
    args = parser.parse_args(argv)
    if args.points < 1 or args.bootstrap_samples < 1:
        parser.error("Points and bootstrap samples must be positive")
    implementation = runtime()
    with implementation.experiment._cpu_executor(1, 1):
        result = export_report(
            args.input.resolve(),
            (args.output or args.input / "report").resolve(),
            read_json(args.plan),
            examples=not args.no_examples,
            replicate=args.replicate,
            points=args.points,
            bootstrap_samples=args.bootstrap_samples,
        )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
