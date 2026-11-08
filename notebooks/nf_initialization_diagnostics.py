"""Compare and inspect independent flow-HMM starts on one saved dataset."""

from concurrent.futures import as_completed
from dataclasses import asdict
import logging
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nf_hmmlearn_comparison as experiment


def _restore_checkpoint(path, train):
    """Restore a model without replaying historical convergence messages."""
    logger = logging.getLogger("hmmlearn.base")
    disabled = logger.disabled
    try:
        logger.disabled = True
        return experiment.restore_flow(path, train, "cpu", "float64")
    finally:
        logger.disabled = disabled

def run_initialization_diagnostic(
    source, experiment_dir, output, *, n_starts=30, n_jobs=8, torch_threads=1,
):
    """Fit new Gaussian-base starts and cache clean-test diagnostic plots.

    Use the selected settings and saved training budget for volatility Trimodal
    replicate zero. Select the best start by training likelihood only. Keep
    checkpoint seeds separate from the main comparison and resume matching runs.
    """
    began = time.perf_counter()
    if source.regime != "volatility":
        raise ValueError("This diagnostic requires the volatility source.")
    experiment._validate_execution(n_starts, torch_threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    starts_dir = output / "starts"
    starts_dir.mkdir(exist_ok=True)
    manifest = experiment.read_json(Path(experiment_dir) / "manifest.json")
    selection = next(
        row for row in manifest["selections"]
        if (row["regime"], row["family"], row["settings"]["base_distribution"])
        == ("volatility", "Trimodal", "normal")
    )
    settings = experiment.FlowSettings(**selection["settings"])
    if settings.device != "cpu":
        raise ValueError("This CPU diagnostic requires CPU-selected settings.")
    n_train = max(source.settings["train_sizes"])
    data = source.data(0, "Trimodal")
    train = data["train"].iloc[:n_train]
    test = data["test"]
    x = train[["x"]].to_numpy()
    seed_parts = [
        source.settings["master_seed"], "initialization_diagnostic_v1",
        "volatility", "Trimodal", 0, n_train,
    ]
    request = dict(
        regime="volatility", family="Trimodal", replicate=0,
        settings=asdict(settings), n_starts=n_starts, n_train=n_train,
        max_iter=source.settings["max_iter"],
        tolerance=n_train * source.settings["tol_per_observation"],
        seed_parts=seed_parts, initialization="random locations; uniform chain",
        source_manifest=experiment.digest(source.path / "manifest.json"),
        saved_data=experiment.digest(
            Path(experiment_dir) / f"volatility_Trimodal_{n_train}_0/data.npz"
        ),
        implementation=experiment.digest(__file__),
        comparison_implementation=experiment.digest(experiment.__file__),
        model_implementation=experiment.digest(
            Path(experiment.__file__).parents[1] / "src/hmmlearn_flow_hmm/model.py"
        ),
        emission_implementation=experiment.digest(
            Path(experiment.__file__).parents[1]
            / "src/flexible_emission_hmm/flow_emissions.py"
        ),
    )
    request_path = output / "request.json"
    if request_path.exists() and experiment.read_json(request_path) != request:
        raise ValueError("Diagnostic settings changed; use a new diagnostic directory.")
    experiment.write_json(request_path, request)
    np.savez_compressed(
        output / "data.npz", train=x, train_states=train.state.to_numpy(),
        test=test.x.to_numpy(), test_states=test.state.to_numpy(),
    )
    jobs = [
        (x, settings, start, request["max_iter"], request["tolerance"],
         seed_parts, str(starts_dir))
        for start in range(n_starts)
    ]
    rows = []
    with experiment._cpu_executor(n_jobs, torch_threads) as executor:
        iterator = (
            map(experiment._fit_single_start, jobs) if executor is None else
            (future.result() for future in as_completed([
                executor.submit(experiment._fit_single_start, job) for job in jobs
            ]))
        )
        for row in iterator:
            rows.append(row)
            print(
                f"Fit {len(rows)}/{n_starts}: start {row['start']}, "
                f"{row['status']}, {row.get('iterations', 0)} EM updates, "
                f"{row['fit_seconds']:.1f}s", flush=True,
            )
    valid = [row for row in rows if row["status"] == "ok"]
    if not valid:
        experiment.write_json(output / "attempts.json", rows)
        raise RuntimeError("All diagnostic starts failed; inspect attempts.json.")
    best = max(valid, key=lambda row: (row["train_ll"], -row["start"]))["start"]
    means, variances = source.true_moments()
    grids = np.array([
        np.linspace(mean - 5 * np.sqrt(var), mean + 5 * np.sqrt(var), 2001)
        for mean, var in zip(means, variances)
    ])
    true_densities = np.array([
        np.exp(source.true_log_density("Trimodal", state, grid))
        for state, grid in enumerate(grids)
    ])
    for row in sorted(rows, key=lambda item: item["start"]):
        row["selected"] = row["start"] == best
        if row["status"] != "ok":
            continue
        cache = output / f"evaluation_{row['start']}.json"
        archive = output / f"evaluation_{row['start']}.npz"
        if cache.exists() and archive.exists():
            row.update(experiment.read_json(cache))
            continue
        model = _restore_checkpoint(starts_dir / f"start_{row['start']}.pt", x)
        mapping = experiment.StateAlignment(3).fit(
            model.predict(x), train.state.to_numpy(),
        ).mapping
        metrics, _, _, predicted, probabilities = experiment.evaluate(
            model, test, source.transitions, mapping,
        )
        order = np.argsort(mapping)
        fitted = np.array([
            np.exp(experiment.emission_model(model).log_prob(grid[:, None])[
                :, order[state]
            ])
            for state, grid in enumerate(grids)
        ])
        np.savez_compressed(
            archive, grids=grids, true_densities=true_densities,
            fitted_densities=fitted, predicted=predicted,
            probabilities=probabilities, mapping=mapping,
            aligned_transitions=model.transmat_[np.ix_(order, order)],
        )
        experiment.write_json(cache, metrics)
        row.update(metrics)
        print(
            f"Evaluated start {row['start']}: {row['inferred_switches']} switches",
            flush=True,
        )
    rows.sort(key=lambda row: row["start"])
    experiment.write_json(output / "attempts.json", rows)
    table = pd.DataFrame(rows).drop(columns="history", errors="ignore")
    table.to_csv(output / "summary.csv", index=False)
    experiment.write_json(output / "execution_timing.json", dict(
        invocation_wall_seconds=time.perf_counter() - began,
        n_jobs=n_jobs, torch_threads=torch_threads,
    ))
    return table

def plot_initialization(output, start, *, window=None):
    """Plot cached densities and decoded states for one diagnostic start."""
    output = Path(output)
    rows = experiment.read_json(output / "attempts.json")
    row = next(item for item in rows if item["start"] == int(start))
    with np.load(output / "data.npz") as data:
        observations = data["test"]
        truth = data["test_states"]
    with np.load(output / f"evaluation_{start}.npz") as data:
        grids = data["grids"]
        true_densities = data["true_densities"]
        fitted = data["fitted_densities"]
        predicted = data["predicted"]
    density_fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    for state, ax in enumerate(axes):
        ax.plot(grids[state], true_densities[state], color="black", label="True density")
        ax.plot(grids[state], fitted[state], color="#228833", label="Gaussian-base flow")
        ax.set(title=f"State {state}", xlabel="Observation", ylabel="Density")
    axes[0].legend(fontsize=8)
    density_fig.suptitle(f"Volatility · Trimodal · replicate 0 · start {start}")
    first, last = (0, len(truth)) if window is None else window
    first, last = max(0, int(first)), min(len(truth), int(last))
    if first >= last:
        plt.close(density_fig)
        raise ValueError("Choose a nonempty test interval.")
    positions = np.arange(first, last)
    decoding_fig, axes = plt.subplots(
        3, 1, figsize=(14, 7), sharex=True, layout="constrained",
    )
    for state, color in enumerate(("#4477AA", "#CC6677", "#228833")):
        mask = truth[first:last] == state
        axes[0].scatter(
            positions[mask], observations[first:last][mask], s=4,
            color=color, label=f"State {state}",
        )
    axes[0].legend(ncol=3)
    axes[0].set_ylabel("Observation")
    axes[1].step(positions, truth[first:last], where="mid", color="black")
    axes[1].set_ylabel("True state")
    axes[2].step(positions, predicted[first:last], where="mid", color="#228833")
    wrong = predicted[first:last] != truth[first:last]
    axes[2].scatter(
        positions[wrong], predicted[first:last][wrong], s=10, marker="x", color="black",
    )
    axes[2].set_ylabel("Gaussian-base flow")
    for ax in axes[1:]:
        ax.set(yticks=range(3), ylim=(-0.3, 2.3))
    axes[-1].set_xlabel("Test observation index")
    decoding_fig.suptitle(
        f"Start {start} · training LL {row['train_ll']:.1f} · "
        f"Macro F1 {100 * row['f1_macro']:.2f}% · "
        f"switches {row['inferred_switches']} (true {row['true_switches']})"
    )
    return density_fig, decoding_fig

def initialization_explorer(output):
    """Build a start selector and time-window slider for cached diagnostics."""
    import ipywidgets as widgets
    from IPython.display import display

    output = Path(output)
    rows = experiment.read_json(output / "attempts.json")
    valid = [row for row in rows if row["status"] == "ok"]
    selected = next(row for row in valid if row["selected"])
    with np.load(output / "data.npz") as data:
        n_test = len(data["test"])
    chooser = widgets.Dropdown(
        options=[
            (f"Start {row['start']:02d} · LL {row['train_ll']:.1f}", row["start"])
            for row in valid
        ],
        value=selected["start"], description="Initialization:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="450px"),
    )
    window = widgets.IntRangeSlider(
        value=(0, n_test), min=0, max=n_test, step=1,
        description="Test interval:", continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="650px"),
    )
    view = widgets.Output()

    def render(change=None):
        """Refresh both cached plots for the selected start and interval."""
        with view:
            view.clear_output(wait=True)
            if window.value[0] == window.value[1]:
                print("Choose a nonempty test interval.")
                return
            figures = plot_initialization(output, chooser.value, window=window.value)
            for figure in figures:
                display(figure)
                plt.close(figure)

    chooser.observe(render, names="value")
    window.observe(render, names="value")
    render()
    return widgets.VBox([chooser, window, view])
