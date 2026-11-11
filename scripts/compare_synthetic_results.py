import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / (
    "data/synthetic/nf_spline_experiments/"
    "four_emissions_p995_hmmlearn_m10_l2_b8-16_s2_i150"
    "_batch2048_float64_cpu8x1_v2_patience5_v2"
)
RUN = ROOT / "cluster_runs/hmm_100x100"
NEW = RUN / "comparison"
OUTPUT = NEW / "previous_experiment_comparison"
KEYS = ["regime", "family", "n_train", "replicate", "model", "condition"]
GROUPS = ["regime", "family", "model"]
METRICS = [
    "f1_macro",
    "test_nll",
    "emission_kl",
    "transition_rmse",
    "true_switches",
    "inferred_switches",
    "absolute_switch_count_error",
    "switch_count_error",
    "switch_f1",
]
MODELS = ["Gaussian HMM", "GMM HMM", "RQ-NSF Gaussian base", "RQ-NSF Student-t base"]

def digest(path):
    """Hash an analysis input."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bootstrap_mean(values):
    """Estimate a mean and its 95% replicate-bootstrap interval."""
    values = np.asarray(values)
    rng = np.random.default_rng(42)
    bounds = np.quantile(
        rng.choice(values, (2000, len(values))).mean(axis=1), [0.025, 0.975]
    )
    return float(np.mean(values)), float(bounds[0]), float(bounds[1])

def main():
    """Save inspectable comparisons and a scientific comparison figure."""
    old = pd.read_csv(OLD / "results.csv", float_precision="round_trip")
    new = pd.read_csv(NEW / "results.csv", float_precision="round_trip")
    for frame in (old, new):
        assert not frame.duplicated(KEYS).any()
        assert frame.evaluation_status.eq("ok").all()
        assert np.isfinite(
            frame[["f1_macro", "test_nll", "transition_rmse"]].to_numpy()
        ).all()
        assert np.isfinite(
            frame.loc[frame.condition.eq("clean"), METRICS].to_numpy()
        ).all()
    original_replicates = set(old.replicate)
    assert original_replicates == set(range(10))
    matched = new[new.replicate.isin(original_replicates)]
    joined = old.merge(
        matched, on=KEYS, suffixes=("_old", "_new"), validate="one_to_one"
    )
    assert len(joined) == len(old) == 1280
    manifest = json.loads((OLD / "manifest.json").read_text())
    plan = json.loads((RUN / "plan.json").read_text())
    assert manifest["selections"] == plan["selections"]
    preserved = []
    for regime, relative in manifest["sources"].items():
        source = ROOT / relative
        destination = ROOT / plan["sources"][regime]
        index = {
            (row["replicate"], row["family"]): row
            for row in json.loads((destination / "dataset_index.json").read_text())
        }
        for row in json.loads((source / "dataset_index.json").read_text()):
            target = index[row["replicate"], row["family"]]
            expected = manifest["fingerprints"][regime][row["file"]]
            assert digest(source / row["file"]) == expected
            assert digest(destination / target["file"]) == expected
            preserved.append((regime, row["family"], row["replicate"]))
    OUTPUT.mkdir(exist_ok=True)
    summaries, plot_rows = [], []
    for label, data in [
        ("previous_10", old),
        ("current_same_10", matched),
        ("current_all_100", new),
    ]:
        for identity, frame in data[data.condition.eq("clean")].groupby(GROUPS):
            for metric in METRICS:
                mean, low, high = bootstrap_mean(frame[metric])
                row = dict(
                    zip(GROUPS, identity),
                    run=label,
                    metric=metric,
                    n_replicates=len(frame),
                    mean=mean,
                    std=frame[metric].std(),
                    ci_low=low,
                    ci_high=high,
                )
                summaries.append(row)
                if metric == "f1_macro" and identity[0] == "volatility":
                    plot_rows.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "clean_comparison.csv", index=False)
    for metric in METRICS:
        joined[f"{metric}_new_minus_old"] = (
            joined[f"{metric}_new"] - joined[f"{metric}_old"]
        )
    joined.to_csv(OUTPUT / "matched_replicate_comparison.csv", index=False)
    deltas = []
    for identity, frame in joined[joined.condition.eq("clean")].groupby(GROUPS):
        for metric in METRICS:
            mean, low, high = bootstrap_mean(frame[f"{metric}_new_minus_old"])
            deltas.append(
                dict(
                    zip(GROUPS, identity),
                    metric=metric,
                    n_pairs=len(frame),
                    mean_delta=mean,
                    ci_low=low,
                    ci_high=high,
                )
            )
    pd.DataFrame(deltas).to_csv(OUTPUT / "matched_mean_differences.csv", index=False)
    columns = [
        "regime",
        "family",
        "n_train",
        "replicate",
        "model",
        "start",
        "seed",
        "status",
        "train_ll",
    ]
    attempts = pd.read_csv(NEW / "attempts.csv", usecols=columns)
    valid = attempts[attempts.status.eq("ok")]
    groups = KEYS[:5]
    gains = pd.concat(
        {
            "best_first_10": valid[valid.start.lt(10)].groupby(groups).train_ll.max(),
            "best_all_100": valid.groupby(groups).train_ll.max(),
        },
        axis=1,
    )
    gains["gain_per_training_observation"] = (
        gains.best_all_100 - gains.best_first_10
    ) / gains.index.get_level_values("n_train")
    assert gains.gain_per_training_observation.ge(0).all()
    gains.to_csv(OUTPUT / "within_current_run_restart_gain.csv")
    old_attempts = pd.read_csv(OLD / "attempts.csv", usecols=columns)
    seed_join = old_attempts.merge(
        attempts,
        on=groups + ["start"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    seed_join["same_seed"] = seed_join.seed_old.eq(seed_join.seed_new)
    seed_agreement = seed_join.groupby(["regime", "model"]).same_seed.mean()
    seed_agreement.to_csv(OUTPUT / "initialization_seed_agreement.csv")
    noise = pd.concat(
        {
            "previous_10": old.groupby(["regime", "model", "condition"])[
                METRICS
            ].mean(),
            "current_same_10": matched.groupby(["regime", "model", "condition"])[
                METRICS
            ].mean(),
            "current_all_100": new.groupby(["regime", "model", "condition"])[
                METRICS
            ].mean(),
        },
        axis=1,
    )
    noise.to_csv(OUTPUT / "noise_comparison.csv")
    provenance = {
        "old": str(OLD),
        "new": str(NEW),
        "preserved_datasets_verified": len(preserved),
        "flow_settings_identical": True,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 42,
        "input_hashes": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                OLD / "results.csv",
                OLD / "attempts.csv",
                OLD / "manifest.json",
                NEW / "results.csv",
                NEW / "attempts.csv",
                RUN / "plan.json",
            ]
        },
        "caveats": [
            "Previous run has 10 starts and 10 replicates; current run has 100 of each.",
            "Matched data reduce sampling differences but do not isolate restart count from fresh refitting.",
            "Volatility Gaussian/GMM baseline seeds differ between the historical and current fits.",
            "Historical and current fitting durations should not be treated as hardware speedup measurements.",
            "The current density repair affects 11 GMM winners outside replicates 0 through 9.",
            "Historical mean-regime Gaussian/GMM noisy rows omit emission KL; compare KL on clean rows only.",
        ],
    }
    (OUTPUT / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    plot = pd.DataFrame(plot_rows)
    versions = [
        ("previous_10", "Previous: 10 starts, 10 datasets", "#CC7722", "o", -0.2),
        ("current_same_10", "Current: 100 starts, same 10 datasets", "#4477AA", "s", 0),
        (
            "current_all_100",
            "Current: 100 starts, all 100 datasets",
            "#228833",
            "^",
            0.2,
        ),
    ]
    for axis, family in zip(
        axes.flat, ["Gaussian", "Student-t", "Trimodal", "Skewed Gamma"]
    ):
        for version, label, color, marker, offset in versions:
            values = (
                plot[(plot.family == family) & (plot.run == version)]
                .set_index("model")
                .loc[MODELS]
            )
            y = np.arange(4) + offset
            axis.hlines(y, values.ci_low, values.ci_high, color=color, linewidth=1.5)
            axis.plot(values["mean"], y, marker, color=color, label=label, markersize=6)
        axis.set(
            title=family,
            xlabel="Mean clean-test macro F1",
            xlim=(0.35, 1.01),
            yticks=np.arange(4),
            yticklabels=[
                "Gaussian HMM",
                "GMM HMM",
                "Gaussian-base flow",
                "Student-t-base flow",
            ],
        )
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1, frameon=False)
    fig.suptitle(
        "Volatility-only experiment: previous and current results\n"
        "Points: replicate means; intervals: 95% replicate-bootstrap intervals"
    )
    fig.savefig(OUTPUT / "volatility_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    cases = [
        ("Gaussian", "RQ-NSF Gaussian base", "Gaussian data / Gaussian-base flow"),
        ("Trimodal", "RQ-NSF Gaussian base", "Trimodal data / Gaussian-base flow"),
        ("Trimodal", "RQ-NSF Student-t base", "Trimodal data / Student-t-base flow"),
        ("Trimodal", "GMM HMM", "Trimodal data / GMM HMM"),
        ("Student-t", "Gaussian HMM", "Student-t data / Gaussian HMM"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    for axis, metric, title in zip(
        axes,
        ["transition_rmse", "absolute_switch_count_error"],
        [
            "Transition probability RMSE",
            "Mean absolute switch-count error / 5,000 observations",
        ],
    ):
        part = summary[(summary.regime == "volatility") & (summary.metric == metric)]
        for version, label, color, marker, offset in versions:
            values = part[part.run == version].set_index(["family", "model"])
            values = values.loc[[(family, model) for family, model, _ in cases]]
            y = np.arange(len(cases)) + offset
            axis.hlines(y, values.ci_low, values.ci_high, color=color, linewidth=1.5)
            axis.plot(values["mean"], y, marker, color=color, label=label, markersize=6)
        axis.set(
            yticks=np.arange(len(cases)),
            yticklabels=[label for _, _, label in cases],
            title=title,
            xlabel="Mean error (log scale; lower is better)",
        )
        axis.set_xscale("log")
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1, frameon=False)
    fig.suptitle(
        "Transition and switch recovery: selected clean volatility cases\n"
        "Points: replicate means; intervals: 95% replicate-bootstrap intervals"
    )
    fig.savefig(
        OUTPUT / "transition_switch_comparison.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        json.dumps(
            {
                "matched_evaluation_rows": len(joined),
                "preserved_datasets_verified": len(preserved),
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )
    print(
        summary.pivot(index=GROUPS + ["metric"], columns="run", values="mean")
        .query("metric == 'f1_macro'")
        .round(5)
        .to_string()
    )

if __name__ == "__main__":
    main()
