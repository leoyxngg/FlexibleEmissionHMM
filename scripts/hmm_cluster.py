import argparse
from concurrent.futures import as_completed
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import warnings

import flow_cluster as common


PROJECT_ROOT = common.PROJECT_ROOT
MODELS = {
    "gaussian": "Gaussian HMM", "gmm": "GMM HMM",
    "normal": "RQ-NSF Gaussian base", "student_t": "RQ-NSF Student-t base",
}
WORKFLOW = "all_hmms_100x100_v1"
DEFAULT_RUN = "cluster_runs/hmm_100x100"

def runtime(root):
    """Load numerical dependencies only for data generation or computation."""
    for relative in ("src", "notebooks"):
        path = str(Path(root) / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
    import hmm_cluster_experiment
    return hmm_cluster_experiment

def build_plan(root, data, manifest, *, n_starts=100, chunk_size=10,
               regimes=None, families=None, replicates=None):
    """Freeze an all-model plan over the prepared paired datasets."""
    root = Path(root).resolve()
    data = common.project_path(root, data)
    if n_starts < 1 or chunk_size < 1:
        raise ValueError("Start and chunk counts must be positive")
    catalog = common.read_json(data / "data_summary.json")
    manifest_path = common.project_path(root, manifest)
    selections = common.read_json(manifest_path)["selections"]
    source_hashes = {
        path.relative_to(root).as_posix(): common.digest(path)
        for path in (data / "data_request.json", data / "data_summary.json")
    }
    groups = []
    found = {"regimes": set(), "families": set(), "replicates": set()}
    for regime, relative in catalog["sources"].items():
        directory = common.project_path(root, relative)
        saved = common.read_json(directory / "manifest.json")
        transitions = saved["transitions"]
        if len(transitions) != 3:
            raise ValueError("Require a three-state transition matrix")
        for state, row in enumerate(transitions):
            if (len(row) != 3 or not all(math.isfinite(value) for value in row)
                    or abs(row[state] - 0.995) > 1e-14
                    or any(abs(value - 0.0025) > 1e-14
                           for target, value in enumerate(row) if state != target)):
                raise ValueError("All dataset sources must use persistence 0.995")
        settings = saved["settings"]
        index = common.read_json(directory / "dataset_index.json")
        indexed = {(row["replicate"], row["family"]): row for row in index}
        expected = {(replicate, family)
                    for replicate in range(settings["n_replicates"])
                    for family in saved["families"]}
        if len(indexed) != len(index) or set(indexed) != expected:
            raise ValueError(f"Dataset index is incomplete or duplicated: {regime}")
        for path in (directory / "manifest.json", directory / "dataset_index.json"):
            source_hashes[path.relative_to(root).as_posix()] = common.digest(path)
        for entry in index:
            path = common.project_path(root, Path(relative) / entry["file"])
            checksum = common.digest(path)
            if checksum != entry["sha256"]:
                raise ValueError(f"Prepared dataset changed: {path}")
            source_hashes[path.relative_to(root).as_posix()] = checksum
        if regimes and regime not in regimes:
            continue
        found["regimes"].add(regime)
        for family in saved["families"]:
            if families and family not in families:
                continue
            found["families"].add(family)
            for replicate in range(settings["n_replicates"]):
                if replicates is not None and replicate not in replicates:
                    continue
                found["replicates"].add(replicate)
                for n_train in settings["train_sizes"]:
                    slug = (
                        f"{regime}_{family.replace(' ', '_')}_"
                        f"{n_train}_{replicate}"
                    )
                    for base, model in MODELS.items():
                        if base in ("gaussian", "gmm"):
                            selected = {
                                "n_components": (
                                    1 if base == "gaussian" else
                                    settings["n_components"]
                                ),
                                "min_variance": settings["min_variance"],
                            }
                            seed_parts = [settings["master_seed"], "fit", replicate]
                        else:
                            matches = [
                                row["settings"] for row in selections
                                if (row["regime"], row["family"],
                                    row["settings"]["base_distribution"])
                                == (regime, family, base)
                            ]
                            if len(matches) != 1:
                                raise ValueError(
                                    f"Missing/duplicate settings: {slug}/{base}"
                                )
                            selected = matches[0]
                            if (selected["device"] != "cpu"
                                    or selected["dtype"] != "float64"):
                                raise ValueError("Require CPU float64 flow settings")
                            seed_parts = [
                                settings["master_seed"], "final_flow", regime,
                                family, replicate, n_train,
                            ]
                        groups.append({
                            "id": f"{slug}/{base}", "slug": slug,
                            "regime": regime, "family": family,
                            "n_train": n_train, "replicate": replicate,
                            "base": base, "model": model, "settings": selected,
                            "max_iter": settings["max_iter"],
                            "tolerance": n_train * settings["tol_per_observation"],
                            "seed_parts": seed_parts,
                            "dataset": (
                                f"{relative}/{indexed[replicate, family]['file']}"
                            ),
                        })
    for key, requested in (
        ("regimes", regimes), ("families", families), ("replicates", replicates),
    ):
        if requested is not None and set(requested) - found[key]:
            raise ValueError(f"Unknown {key}: {set(requested) - found[key]}")
    if not groups or len({group["id"] for group in groups}) != len(groups):
        raise ValueError("Require distinct, nonempty model groups")
    implementations = [
        "scripts/hmm_cluster.py", "scripts/flow_cluster.py",
        "notebooks/hmm_cluster_experiment.py",
        "notebooks/nf_hmmlearn_comparison.py",
        "notebooks/synthetic_emission_comparison.py",
        "notebooks/volatility_markov_comparison.py",
        "pyproject.toml", "uv.lock", ".python-version",
    ]
    implementations += [path.relative_to(root).as_posix()
                        for path in sorted((root / "src").rglob("*.py"))]
    plan = {
        "format_version": 1, "workflow": WORKFLOW,
        "n_starts": n_starts, "chunk_size": chunk_size,
        "n_replicates": catalog["n_replicates"], "persistence": 0.995,
        "sources": catalog["sources"], "source_hashes": source_hashes,
        "implementation_hashes": {
            name: common.digest(common.project_path(root, name))
            for name in implementations
        },
        "settings_manifest": manifest_path.relative_to(root).as_posix(),
        "settings_manifest_hash": common.digest(manifest_path),
        "selections": selections, "groups": groups,
        "selection": "Highest finite training LL; earliest start breaks ties",
        "baselines": "Retrain Gaussian and GMM with the same initialization budget",
    }
    plan["id"] = common.object_digest(plan)
    return plan

def summary(plan):
    """Describe model counts, data coverage, and the complete fitting budget."""
    return {
        "plan_id": plan["id"], "workflow": plan["workflow"],
        "dataset_conditions": len({group["slug"] for group in plan["groups"]}),
        "models": list(MODELS.values()), "model_groups": len(plan["groups"]),
        "dataset_replicates_per_regime_family": plan["n_replicates"],
        "selected_replicates": sorted({g["replicate"] for g in plan["groups"]}),
        "starts_per_model_dataset": plan["n_starts"],
        "training_attempts": len(plan["groups"]) * plan["n_starts"],
        "jobs": len(common.tasks(plan)), "starts_per_job": plan["chunk_size"],
        "persistence": plan["persistence"],
    }

def validate_worker(root, plan, group):
    """Validate only assigned data plus shared code to avoid repeated full scans."""
    content = {key: value for key, value in plan.items() if key != "id"}
    if (plan.get("workflow") != WORKFLOW
            or common.object_digest(content) != plan.get("id")):
        raise ValueError("Invalid all-model plan")
    relevant = {
        **plan["implementation_hashes"],
        plan["settings_manifest"]: plan["settings_manifest_hash"],
        group["dataset"]: plan["source_hashes"][group["dataset"]],
    }
    for relative in plan["sources"].values():
        for name in ("manifest.json", "dataset_index.json"):
            path = f"{relative}/{name}"
            relevant[path] = plan["source_hashes"][path]
    for relative, expected in relevant.items():
        if common.digest(common.project_path(root, relative)) != expected:
            raise ValueError(f"Assigned input or implementation changed: {relative}")

def run_worker(root, plan, task_id, output, *, workers=1, start_ids=None):
    """Train a chunk of one model on its assigned paired dataset."""
    task = next((task for task in common.tasks(plan) if task["id"] == task_id), None)
    if task is None or workers < 1:
        raise ValueError("Require a valid task and positive process count")
    group = task["group"]
    validate_worker(root, plan, group)
    assigned = list(range(task["begin"], task["end"]))
    requested = assigned if start_ids is None else sorted(set(start_ids))
    if not requested or not set(requested).issubset(assigned):
        raise ValueError("Requested starts are outside the assigned range")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "plan_id": plan["id"], "task_id": task_id,
        "requested_starts": requested,
    }
    metadata = output / "chunk.json"
    if metadata.exists():
        previous = common.read_json(metadata)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Output belongs to a different assignment")
    elif any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    implementation = runtime(root)
    source = implementation.load_sources(root, plan["sources"])[group["regime"]]
    frame = source.data(group["replicate"], group["family"], ("train",))["train"]
    train = frame[["x"]].to_numpy()[:group["n_train"]]
    if len(train) != group["n_train"]:
        raise ValueError("Training prefix is too short")
    starts = output / "starts"
    starts.mkdir(exist_ok=True)
    report = {
        **identity, "group_id": group["id"], "begin": task["begin"],
        "end": task["end"], "complete": False, "artifacts": [],
        "versions": {name: version(name) for name in ("torch", "numpy", "hmmlearn")},
        "job_id": os.environ.get("UTSC_JOB_ID"),
    }
    common.write_json(metadata, report)
    jobs = []
    for start in requested:
        if (starts / f"start_{start}.json").exists():
            common._attempt_artifact(output, group, start)
        jobs.append((train, group, start, str(starts)))
    with implementation.experiment._cpu_executor(workers, 1) as executor:
        if executor is None:
            iterator = map(implementation.fit_start, jobs)
        else:
            futures = [executor.submit(implementation.fit_start, job) for job in jobs]
            iterator = (future.result() for future in as_completed(futures))
        for row in iterator:
            artifact, row = common._attempt_artifact(output, group, row["start"])
            report["artifacts"].append(artifact)
            common.write_json(metadata, report)
            print(f"{group['model']} {task_id}: start {row['start']} "
                  f"{row['status']} ({len(report['artifacts'])}/{len(jobs)})",
                  flush=True)
    report["complete"] = True
    common.write_json(metadata, report)
    return report

def submit_plan(root, plan_path, plan, *, execute=False, datacluster=None,
                lab_user=None, receipt=None, preview_output=None, **resources):
    """Preview or explicitly submit all-model jobs through DataCluster."""
    common.validate_plan(root, plan)
    if plan.get("workflow") != WORKFLOW:
        raise ValueError("Use an all-model plan with this entry point")
    spec = common.submission_spec(root, plan_path, plan, **resources)
    spec["command"][2] = "scripts/hmm_cluster.py"
    preview = {**summary(plan), "submission": spec, "execute": execute}
    if preview_output:
        common.write_json(preview_output, preview)
    if not execute:
        return {
            **summary(plan), "execute": False,
            "selected_jobs": len(spec["params"]), "command": spec["command"],
            "machines": spec["machines"], "cpus_per_job": spec["cpus"],
            "memory_gb_per_job": spec["memory_gb"], "timeout": spec["timeout"],
            "preview_output": str(preview_output) if preview_output else None,
        }
    if not lab_user:
        raise ValueError("Provide --lab-user before explicit submission")
    receipt = Path(receipt or Path(plan_path).with_name("submission.json"))
    if receipt.exists():
        raise FileExistsError("Use a new receipt for a deliberate resubmission")
    if datacluster:
        sys.path.insert(0, str(Path(datacluster).expanduser().resolve() / "src"))
    from utsc_cluster import Client, Env
    client = Client()
    if client.cfg.utorid != lab_user:
        raise ValueError("The expected lab username does not match DataCluster")
    params = spec["params"]
    project, command = spec.pop("project"), spec.pop("command")
    batch = client.map(project, command, env=Env.python(extras=["flows", "cpu"]),
                       **spec)
    result = {
        "plan_id": plan["id"], "batch_id": batch.id,
        "jobs": [{"task_id": item["task_id"], "job_id": job.id,
                  "results_dir": str(job.results_dir)}
                 for item, job in zip(params, batch.jobs)],
    }
    common.write_json(receipt, result)
    return {"batch_id": batch.id, "jobs": len(batch.jobs), "receipt": str(receipt)}

def _evaluate_condition(implementation, source, groups, output, n_starts,
                        *, density=True):
    import numpy as np
    experiment = implementation.experiment
    identity = {key: groups[0][key]
                for key in ("regime", "family", "n_train", "replicate")}
    directory = Path(output) / groups[0]["slug"]
    cached = directory / "tables.json"
    if cached.exists():
        return common.read_json(cached)
    tables = {name: [] for name in common.TABLES}
    data = source.data(identity["replicate"], identity["family"])
    train = data["train"].iloc[:identity["n_train"]]
    x, y = train[["x"]].to_numpy(), train.state.to_numpy()
    variants = source.variants(train, data["test"], identity["replicate"],
                               identity["family"], identity["n_train"])
    np.savez_compressed(directory / "data.npz", **{
        **{f"train__{column}": train[column].to_numpy() for column in train},
        **{f"{condition}__{column}": frame[column].to_numpy()
           for condition, frame in variants.items() for column in frame},
    })
    for split, frame in (("train", train), ("test", data["test"])):
        tables["coverage"].append({
            **identity, "split": split, "length": len(frame),
            "state_visits": np.bincount(frame.state, minlength=3).tolist(),
            "switches": int(np.count_nonzero(np.diff(frame.state))),
        })
    for group in groups:
        cache = common._cache_directory(output, group)
        attempts = [common.read_json(cache / f"start_{start}.json")
                    for start in range(n_starts)]
        valid = [row for row in attempts if row["status"] == "ok"]
        chosen = max(valid, key=lambda row: (row["train_ll"], -row["start"]))
        for row in attempts:
            row["selected"] = row["start"] == chosen["start"]
            tables["attempts"].append({
                **identity, "model": group["model"],
                **{key: value for key, value in row.items() if key != "history"},
            })
        checkpoint = directory / f"model_{group['base']}.pt"
        shutil.copy2(cache / f"start_{chosen['start']}.pt", checkpoint)
        model = implementation.restore_model(checkpoint, x, group)
        mapping = experiment.StateAlignment(3).fit(model.predict(x), y).mapping
        order = np.argsort(mapping)
        parameters = experiment.chain_parameters(model)
        parameter_row = {
            **identity, "model": group["model"], "origin": "retrained",
            "checkpoint": str(checkpoint), "settings": group["settings"],
            "selected_start": chosen["start"], "train_ll": chosen["train_ll"],
            "mapping_learned_to_true": mapping.tolist(),
            "aligned_transitions": (
                parameters.transitions[np.ix_(order, order)].tolist()
            ),
            "aligned_initial_probs": parameters.start[order].tolist(),
        }
        if group["base"] in ("gaussian", "gmm"):
            weights, means, variances = experiment.reference.fitted_components(
                model, order,
            )
            parameter_row.update(
                weights=weights.tolist(), means=means.tolist(),
                variances=variances.tolist(),
            )
        tables["parameters"].append(parameter_row)
        recovery, state_kl, diagnostic_warnings = {}, [], []
        if density:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    recovery, state_kl = experiment.flow_recovery(
                        source, group["family"], model, mapping,
                    )
                diagnostic_warnings = [str(item.message) for item in caught]
            except Exception as exc:
                diagnostic_warnings = [f"{type(exc).__name__}: {exc}"]
        for condition, frame in variants.items():
            row = {
                **identity, "model": group["model"], "condition": condition,
                "evaluation_status": "failed", "selected_start": chosen["start"],
                "train_ll": chosen["train_ll"], "iterations": chosen["iterations"],
                "converged": chosen["converged"], "fit_seconds": chosen["fit_seconds"],
                "total_fit_seconds": sum(item["fit_seconds"] for item in attempts),
                "recovery_status": (
                    "skipped" if not density else "ok" if recovery else "failed"
                ),
                "evaluation_warnings": diagnostic_warnings,
            }
            try:
                metrics, states, matrix, predicted, probabilities = experiment.evaluate(
                    model, frame, source.transitions, mapping,
                )
                row.update(metrics, **recovery, evaluation_status="ok")
                for values in states:
                    if state_kl:
                        values["emission_kl"] = state_kl[values["state"]]
                    tables["per_state"].append({
                        **identity, "model": group["model"], "condition": condition,
                        **values,
                    })
                tables["confusion"].extend({
                    **identity, "model": group["model"], "condition": condition,
                    "true_state": k, "predicted_state": j, "count": int(matrix[k, j]),
                } for k in range(3) for j in range(3))
                np.savez_compressed(
                    directory / f"{group['base']}_{condition}.npz",
                    predicted=predicted, probabilities=probabilities,
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            tables["results"].append(row)
    common.write_json(cached, experiment.reference.json_safe(tables))
    return tables

def evaluate_results(root, plan, output, *, density=True):
    """Select all four winners and evaluate held-out data without any fitting."""
    common.validate_plan(root, plan)
    output = Path(output)
    index = common.read_json(output / "cluster_collection.json")
    if index["plan_id"] != plan["id"]:
        raise ValueError("Collection belongs to another plan")
    common._verify_merged(output, plan, index["records"])
    expected = {f"{group['id']}/{start}" for group in plan["groups"]
                for start in range(plan["n_starts"])}
    if set(index["records"]) != expected:
        raise ValueError("Collect every initialization before evaluating")
    for group in plan["groups"]:
        if not any(index["records"][f"{group['id']}/{start}"]["status"] == "ok"
                   for start in range(plan["n_starts"])):
            raise ValueError(f"All starts failed: {group['id']}")
    request = {"plan_id": plan["id"], "density_diagnostics": density,
               "selection": plan["selection"], "baselines": plan["baselines"]}
    request_path = output / "evaluation_request.json"
    if request_path.exists() and common.read_json(request_path) != request:
        raise ValueError(
            "Evaluation settings changed; use a fresh collection directory"
        )
    common.write_json(request_path, request)
    implementation = runtime(root)
    sources = implementation.load_sources(root, plan["sources"])
    conditions = {}
    for group in plan["groups"]:
        conditions.setdefault(group["slug"], []).append(group)
    tables = {name: [] for name in common.TABLES}
    with implementation.experiment._cpu_executor(1, 1):
        for slug, groups in conditions.items():
            if {group["base"] for group in groups} != set(MODELS):
                raise ValueError(f"Evaluation requires all four models: {slug}")
            result = _evaluate_condition(
                implementation, sources[groups[0]["regime"]], groups, output,
                plan["n_starts"], density=density,
            )
            for name in tables:
                tables[name].extend(result[name])
            print(f"Evaluated {slug}", flush=True)
    import pandas as pd
    for name, rows in tables.items():
        if name == "baseline_replay":
            pd.DataFrame(columns=[
                "regime", "family", "n_train", "replicate", "model", "condition",
                "metric", "delta", "passed",
            ]).to_csv(output / f"{name}.csv", index=False)
        else:
            pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    common.write_json(output / "manifest.json", {
        **request, "workflow": WORKFLOW, "sources": plan["sources"],
        "selections": plan["selections"], "n_starts": plan["n_starts"],
        "n_replicates": plan["n_replicates"], "persistence": plan["persistence"],
        "baseline_replay": "Not applicable: both baselines are retrained",
    })
    return {name: len(rows) for name, rows in tables.items()}

def main(argv=None):
    """Run preparation, planning, submission, collection, or evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser(
        "prepare-data", help="Generate data locally; no model fitting",
    )
    prepare.add_argument("--output", type=Path, default=Path(DEFAULT_RUN) / "data")
    prepare.add_argument("--replicates", type=int, default=100)
    prepare.add_argument("--starts", type=int, default=100)
    planning = sub.add_parser("plan", help="Freeze a plan locally")
    planning.add_argument("--data", default=f"{DEFAULT_RUN}/data")
    planning.add_argument("--manifest", default=f"{common.SAVED_RUN}/manifest.json")
    planning.add_argument(
        "--output", type=Path, default=Path(DEFAULT_RUN) / "plan.json",
    )
    planning.add_argument("--starts", type=int, default=100)
    planning.add_argument("--chunk-size", type=int, default=10)
    planning.add_argument("--regime", action="append", choices=list(common.SOURCES))
    planning.add_argument("--family", action="append")
    planning.add_argument("--replicate", action="append", type=int)
    for stage in ("submit", "worker", "collect", "evaluate"):
        command = sub.add_parser(stage)
        command.add_argument(
            "--plan", type=Path, default=Path(DEFAULT_RUN) / "plan.json",
        )
        if stage == "submit":
            command.add_argument("--execute", action="store_true")
            command.add_argument("--datacluster", type=Path)
            command.add_argument("--lab-user")
            command.add_argument("--receipt", type=Path)
            command.add_argument("--preview-output", type=Path)
            command.add_argument("--machines", type=int, default=100)
            command.add_argument("--cpus", type=int, default=4)
            command.add_argument("--memory", type=float, default=6)
            command.add_argument("--timeout", default="4h")
            command.add_argument("--tasks-from", type=Path)
        elif stage == "worker":
            command.add_argument("--task-id")
            command.add_argument("--workers", type=int, default=1)
            command.add_argument("--output", type=Path)
        elif stage == "collect":
            command.add_argument("--results", action="append", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--skip-density", action="store_true")
    args = parser.parse_args(argv)
    root = PROJECT_ROOT
    if args.stage == "prepare-data":
        result = runtime(root).prepare_datasets(
            root, common.SOURCES, common.project_path(root, args.output),
            n_replicates=args.replicates, n_starts=args.starts,
        )
    elif args.stage == "plan":
        plan = build_plan(
            root, args.data, args.manifest, n_starts=args.starts,
            chunk_size=args.chunk_size, regimes=args.regime,
            families=args.family, replicates=args.replicate,
        )
        if args.output.exists() and common.read_json(args.output) != plan:
            raise ValueError("Existing plan differs; use a new output path")
        common.write_json(args.output, plan)
        result = summary(plan)
    else:
        plan = common.read_json(args.plan)
        if plan.get("workflow") != WORKFLOW:
            raise ValueError("This command requires an all-model plan")
        if args.stage == "submit":
            missing = None
            if args.tasks_from:
                missing = common.read_json(args.tasks_from)
                if missing["plan_id"] != plan["id"]:
                    raise ValueError("Resubmission report belongs to a different plan")
            result = submit_plan(
                root, args.plan, plan, execute=args.execute,
                datacluster=args.datacluster, lab_user=args.lab_user,
                receipt=args.receipt, preview_output=args.preview_output,
                machines=args.machines, cpus=args.cpus, memory_gb=args.memory,
                timeout=args.timeout,
                task_ids=missing["missing_tasks"] if missing else None,
                starts_by_task=missing["missing_starts"] if missing else None,
            )
        elif args.stage == "worker":
            params = json.loads(os.environ.get("UTSC_PARAMS", "{}"))
            if params and params.get("plan_id") != plan["id"]:
                raise ValueError("Assigned plan ID does not match")
            output = args.output or os.environ.get("OUT_DIR")
            if not output:
                raise ValueError("Provide OUT_DIR or --output")
            result = run_worker(
                root, plan, args.task_id or params.get("task_id"), output,
                workers=args.workers, start_ids=params.get("start_ids"),
            )
        elif args.stage == "collect":
            result = common.collect_results(root, plan, args.results, args.output)
        else:
            result = evaluate_results(root, plan, args.output,
                                      density=not args.skip_density)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
