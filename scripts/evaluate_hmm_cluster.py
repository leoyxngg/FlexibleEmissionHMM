import argparse
import ast
import csv
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import flow_cluster as common
import hmm_cluster as training

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("cluster_runs/hmm_100x100")
DEFAULT_DIRECTORY = ROOT / RUN / "evaluation_cluster"
FORMAT = "distributed_hmm_evaluation_v1"
PAYLOAD_PLAN = Path("evaluation/evaluation_plan.json")
SCRIPT = Path("scripts/evaluate_hmm_cluster.py")
GROUP_COLUMNS = {"group_id", "regime", "family", "replicate", "n_train", "model"}
INTEGER_FIELDS = {
    "start",
    "seed",
    "iterations",
    "likelihood_decreases",
    "convergence_patience",
    "consecutive_small_changes",
}
FLOAT_FIELDS = {"train_ll", "fit_seconds", "final_gain_per_observation"}
BOOLEAN_FIELDS = {
    "selected",
    "converged",
    "iteration_limit_reached",
    "monitor_converged",
}

def safe_path(root, relative):
    """Resolve a relative artifact path without allowing directory traversal."""
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Unsafe artifact path: {relative}")
    return path

def condition_groups(plan):
    """Index the original four model groups by dataset condition."""
    groups = defaultdict(list)
    for group in plan["groups"]:
        groups[group["slug"]].append(group)
    for slug, values in groups.items():
        if {group["base"] for group in values} != set(training.MODELS):
            raise ValueError(f"Expected all four model types: {slug}")
    return dict(groups)

def evaluation_request(plan, density=True):
    """Match the original evaluator's method and cache identity."""
    return {
        "plan_id": plan["id"],
        "density_diagnostics": density,
        "selection": plan["selection"],
        "baselines": plan["baselines"],
    }

def decode_summary(row):
    """Restore typed metadata from the validated collection's attempt CSV."""
    result = {}
    for key, value in row.items():
        if key in GROUP_COLUMNS or value == "":
            continue
        if key in INTEGER_FIELDS:
            result[key] = int(value)
        elif key in FLOAT_FIELDS:
            result[key] = float(value)
        elif key in BOOLEAN_FIELDS:
            if value not in {"True", "False"}:
                raise ValueError(f"Invalid boolean summary field: {key}")
            result[key] = value == "True"
        elif key == "warnings":
            result[key] = ast.literal_eval(value)
        else:
            result[key] = value
    return result

def validate_tables(tables, groups, conditions, n_starts, selected=None):
    """Validate evaluation identities and the complete per-condition fit budget."""
    if set(tables) != set(common.TABLES):
        raise ValueError("Unexpected evaluation table set")
    identity = {key: groups[0][key] for key in common_key_names()}
    names = {group["model"] for group in groups}
    seen = set()
    for row in tables["results"]:
        if any(row.get(key) != value for key, value in identity.items()):
            raise ValueError("Evaluation dataset identity mismatch")
        key = row["model"], row["condition"]
        if key in seen:
            raise ValueError("Duplicate evaluation result")
        seen.add(key)
        if row.get("evaluation_status") not in {"ok", "failed"}:
            raise ValueError("Missing evaluation outcome")
        if selected and row["selected_start"] != selected[row["model"]]:
            raise ValueError("Evaluation selected the wrong initialization")
    if seen != {(model, condition) for model in names for condition in conditions}:
        raise ValueError("Incomplete model/test-condition coverage")
    attempts = tables["attempts"]
    keys = {(row["model"], row["start"]) for row in attempts}
    if len(keys) != len(attempts) or keys != {
        (model, start) for model in names for start in range(n_starts)
    }:
        raise ValueError("Incomplete or duplicate initialization summaries")
    for table in ["attempts", "parameters", "per_state", "confusion", "coverage"]:
        for row in tables[table]:
            if any(row.get(key) != value for key, value in identity.items()):
                raise ValueError(f"Dataset identity mismatch in {table}")
    parameters = tables["parameters"]
    if len(parameters) != len(names) or {r["model"] for r in parameters} != names:
        raise ValueError("Incomplete selected-model parameters")
    for row in parameters:
        if sorted(row["mapping_learned_to_true"]) != [0, 1, 2]:
            raise ValueError("Invalid state alignment")
        if selected and row["selected_start"] != selected[row["model"]]:
            raise ValueError("Selected-checkpoint identity mismatch")

def common_key_names():
    """Return the fixed dataset identity columns used by the original evaluator."""
    return ["regime", "family", "n_train", "replicate"]

def test_conditions(root, plan, group):
    """Read the frozen clean and noisy test-condition names."""
    source = common.read_json(
        Path(root) / plan["sources"][group["regime"]] / "manifest.json"
    )
    levels = source["settings"].get("noise_levels", [0.1, 0.25, 0.5])
    return ["clean"] + [f"noise_{level}" for level in levels]

def completed_conditions(root, comparison, plan):
    """Recognize only fully committed, method-compatible local evaluations."""
    request = comparison / "evaluation_request.json"
    if request.exists() and common.read_json(request) != evaluation_request(plan):
        raise ValueError("Local evaluation uses different settings")
    completed = []
    for slug, groups in condition_groups(plan).items():
        marker = comparison / slug / "tables.json"
        if not marker.exists():
            continue
        validate_tables(
            common.read_json(marker),
            groups,
            test_conditions(root, plan, groups[0]),
            plan["n_starts"],
        )
        completed.append(slug)
    return completed

def prepare(root, plan_path, comparison, directory):
    """Package only unevaluated datasets and their selected trained checkpoints."""
    root, comparison, directory = (
        Path(p).resolve() for p in (root, comparison, directory)
    )
    if directory.exists():
        raise FileExistsError("Preparation already exists; use a new --directory")
    plan = common.read_json(plan_path)
    common.validate_plan(root, plan)
    index = common.read_json(comparison / "cluster_collection.json")
    report = common.read_json(comparison / "collection_report.json")
    if index["plan_id"] != plan["id"] or report["plan_id"] != plan["id"]:
        raise ValueError("Collection belongs to a different training plan")
    expected = len(plan["groups"]) * plan["n_starts"]
    if (
        report["missing"]
        or report["recorded"] != expected
        or not report["ready_for_evaluation"]
    ):
        raise ValueError("Complete validated training collection is required")
    grouped = condition_groups(plan)
    done = completed_conditions(root, comparison, plan)
    pending = {
        slug: groups for slug, groups in grouped.items() if slug not in set(done)
    }
    if not pending:
        raise ValueError(
            "Every condition is already evaluated; no remote jobs are needed"
        )
    wanted = {group["id"] for groups in pending.values() for group in groups}
    summaries = defaultdict(list)
    csv_path = comparison / "cluster_attempts.csv"
    summary_hash = common.digest(csv_path)
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["group_id"] in wanted:
                summaries[row["group_id"]].append(decode_summary(row))
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="evaluation-preparation-", dir=directory.parent)
    )
    try:
        payload = temporary / "payload" / "CSCD94"
        shared = {}

        def copy_input(relative, expected_hash=None, *, shared_file=True):
            source = safe_path(root, relative)
            digest = common.digest(source)
            if expected_hash and digest != expected_hash:
                raise ValueError(f"Frozen input changed: {relative}")
            target = safe_path(payload, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if shared_file:
                shared[relative] = digest

        for relative, expected_hash in plan["implementation_hashes"].items():
            copy_input(relative, expected_hash)
        for relative, expected_hash in plan["source_hashes"].items():
            if not relative.endswith(".npz"):
                copy_input(relative, expected_hash)
        copy_input(plan["settings_manifest"], plan["settings_manifest_hash"])
        relative_plan = Path(plan_path).resolve().relative_to(root).as_posix()
        copy_input(relative_plan)
        copy_input("README.md")
        target = payload / SCRIPT
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), target)
        shared[SCRIPT.as_posix()] = common.digest(target)
        tasks = []
        for slug, groups in pending.items():
            records, winners = {}, {}
            relative = f"evaluation/inputs/{slug}/attempts.json"
            hashes = {}
            for group in groups:
                attempts = sorted(summaries[group["id"]], key=lambda row: row["start"])
                if [row["start"] for row in attempts] != list(range(plan["n_starts"])):
                    raise ValueError(
                        f"Missing or duplicate attempt summaries: {group['id']}"
                    )
                for row in attempts:
                    item = index["records"][f"{group['id']}/{row['start']}"]
                    if (
                        row["seed"] != common.start_seed(group, row["start"])
                        or row["status"] != item["status"]
                    ):
                        raise ValueError(
                            "Attempt summaries do not match the frozen collection"
                        )
                valid = [
                    row
                    for row in attempts
                    if row["status"] == "ok" and math.isfinite(row["train_ll"])
                ]
                if not valid:
                    raise ValueError(f"No successful fit: {group['id']}")
                chosen = max(valid, key=lambda row: (row["train_ll"], -row["start"]))
                cache = common._cache_directory(comparison, group)
                record_path = cache / f"start_{chosen['start']}.json"
                original = common.read_json(record_path)
                artifact = {
                    "start": chosen["start"],
                    "record_hash": common.digest(record_path),
                    "checkpoint_hash": common.digest(record_path.with_suffix(".pt")),
                }
                indexed = index["records"][f"{group['id']}/{chosen['start']}"]
                if any(artifact[key] != indexed[key] for key in artifact):
                    raise ValueError(
                        "Selected checkpoint failed collection integrity checks"
                    )
                for key in ["seed", "train_ll", "iterations", "fit_seconds", "status"]:
                    if chosen[key] != original[key]:
                        raise ValueError(
                            "Winner summary differs from original fit record"
                        )
                attempts[chosen["start"]] = {
                    k: v for k, v in original.items() if k != "history"
                }
                records[group["base"]] = attempts
                winners[group["model"]] = chosen["start"]
                checkpoint = f"evaluation/inputs/{slug}/{group['base']}.pt"
                destination = payload / checkpoint
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache / f"start_{chosen['start']}.pt", destination)
                hashes[checkpoint] = artifact["checkpoint_hash"]
            common.write_json(payload / relative, records)
            hashes[relative] = common.digest(payload / relative)
            dataset = groups[0]["dataset"]
            copy_input(dataset, plan["source_hashes"][dataset], shared_file=False)
            hashes[dataset] = plan["source_hashes"][dataset]
            tasks.append(
                {
                    "slug": slug,
                    "groups": groups,
                    "bundle": relative,
                    "hashes": hashes,
                    "selected": winners,
                    "conditions": test_conditions(root, plan, groups[0]),
                }
            )
        evaluation = {
            "format": FORMAT,
            "training_plan_id": plan["id"],
            "training_plan": relative_plan,
            "request": evaluation_request(plan),
            "source_comparison": str(comparison),
            "n_starts": plan["n_starts"],
            "shared_hashes": shared,
            "tasks": tasks,
            "completed_at_prepare": done,
            "summary_source_hash": summary_hash,
            "summary_note": "Nonwinning histories omitted; original selected records and checkpoint hashes verified",
        }
        evaluation["id"] = common.object_digest(evaluation)
        common.write_json(payload / PAYLOAD_PLAN, evaluation)
        common.write_json(temporary / "evaluation_plan.json", evaluation)
        files = list(payload.rglob("*"))
        summary = {
            "evaluation_plan_id": evaluation["id"],
            "completed_locally": len(done),
            "jobs_prepared": len(tasks),
            "selected_models": len(tasks) * 4,
            "payload_files": sum(path.is_file() for path in files),
            "payload_mb": round(
                sum(path.stat().st_size for path in files if path.is_file()) / 1e6, 1
            ),
            "directory": str(directory),
            "submitted": False,
        }
        common.write_json(temporary / "preparation_summary.json", summary)
        temporary.rename(directory)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

def load_evaluation(path):
    """Validate the distributed evaluation plan's immutable identity."""
    plan = common.read_json(path)
    if plan.get("format") != FORMAT or common.object_digest(
        {k: v for k, v in plan.items() if k != "id"}
    ) != plan.get("id"):
        raise ValueError("Invalid distributed evaluation plan")
    if len({task["slug"] for task in plan["tasks"]}) != len(plan["tasks"]):
        raise ValueError("Duplicate evaluation assignments")
    return plan

def verify_payload(root, evaluation, slugs):
    """Verify frozen shared inputs and every requested evaluation assignment."""
    expected = dict(evaluation["shared_hashes"])
    wanted = set(slugs)
    for task in evaluation["tasks"]:
        if task["slug"] in wanted:
            expected.update(task["hashes"])
    for relative, digest in expected.items():
        if common.digest(safe_path(root, relative)) != digest:
            raise ValueError(f"Evaluation input changed: {relative}")

def worker(root, evaluation, slug, output):
    """Evaluate one dataset using frozen winners without any model fitting."""
    root, output = Path(root), Path(output)
    task = next((task for task in evaluation["tasks"] if task["slug"] == slug), None)
    if task is None:
        raise ValueError("Unassigned evaluation condition")
    plan = common.read_json(root / evaluation["training_plan"])
    if plan["id"] != evaluation["training_plan_id"]:
        raise ValueError("Training plan mismatch")
    training.validate_worker(root, plan, task["groups"][0])
    verify_payload(root, evaluation, [slug])
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "evaluation_result.json"
    if marker.exists():
        result = common.read_json(marker)
        if result["evaluation_plan_id"] != evaluation["id"] or result["slug"] != slug:
            raise ValueError("Output belongs to another evaluation assignment")
        for relative, expected in result["files"].items():
            if common.digest(safe_path(output, relative)) != expected:
                raise ValueError("Existing evaluation output changed")
        return result
    if any(output.iterdir()):
        raise ValueError("Use an empty OUT_DIR for a new evaluation")
    implementation = training.runtime(root)
    sources = implementation.load_sources(root, plan["sources"])
    records = common.read_json(root / task["bundle"])
    with tempfile.TemporaryDirectory(prefix="hmm-evaluation-") as temporary:
        temporary = Path(temporary)
        for group in task["groups"]:
            cache = common._cache_directory(temporary, group)
            cache.mkdir(parents=True, exist_ok=True)
            attempts = records[group["base"]]
            if [row["start"] for row in attempts] != list(range(plan["n_starts"])):
                raise ValueError("Invalid bundled initialization summaries")
            for row in attempts:
                common.write_json(cache / f"start_{row['start']}.json", row)
            chosen = task["selected"][group["model"]]
            checkpoint = root / f"evaluation/inputs/{slug}/{group['base']}.pt"
            shutil.copy2(checkpoint, cache / f"start_{chosen}.pt")
        with implementation.experiment._cpu_executor(1, 1):
            tables = training._evaluate_condition(
                implementation,
                sources[task["groups"][0]["regime"]],
                task["groups"],
                temporary,
                plan["n_starts"],
                density=evaluation["request"]["density_diagnostics"],
            )
        validate_tables(
            tables,
            task["groups"],
            task["conditions"],
            plan["n_starts"],
            task["selected"],
        )
        produced = temporary / slug
        destination = output / slug
        destination.mkdir()
        for path in produced.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
    files = {
        path.relative_to(output).as_posix(): common.digest(path)
        for path in (output / slug).iterdir()
        if path.is_file()
    }
    result = {
        "format": FORMAT,
        "evaluation_plan_id": evaluation["id"],
        "training_plan_id": plan["id"],
        "slug": slug,
        "files": files,
        "versions": {
            name: version(name)
            for name in ["numpy", "torch", "hmmlearn", "scipy", "pandas"]
        },
        "evaluation_status": dict(
            Counter(row["evaluation_status"] for row in tables["results"])
        ),
        "recovery_status": dict(
            Counter(row["recovery_status"] for row in tables["results"])
        ),
    }
    common.write_json(marker, result)
    return result

def submit(
    directory,
    comparison,
    *,
    execute=False,
    lab_user=None,
    machines=220,
    cpus=1,
    memory=6,
    timeout="4h",
):
    """Preview or explicitly submit only conditions still missing locally."""
    directory, comparison = Path(directory).resolve(), Path(comparison).resolve()
    evaluation = load_evaluation(directory / "evaluation_plan.json")
    payload = directory / "payload/CSCD94"
    original = common.read_json(payload / evaluation["training_plan"])
    done = set(completed_conditions(payload, comparison, original))
    selected = [task for task in evaluation["tasks"] if task["slug"] not in done]
    if min(machines, cpus, memory) <= 0:
        raise ValueError("Request positive machine, CPU, and memory limits")
    preview = {
        "evaluation_plan_id": evaluation["id"],
        "jobs": len(selected),
        "already_complete": len(done),
        "machines": machines,
        "cpus_per_job": cpus,
        "memory_gb_per_job": memory,
        "project": str(payload),
        "execute": execute,
    }
    if not execute or not selected:
        return preview
    receipt = directory / "submission.json"
    if receipt.exists():
        raise FileExistsError(
            "Evaluation receipt exists; do not submit this batch twice"
        )
    if not lab_user:
        raise ValueError("Provide --lab-user for explicit remote submission")
    verify_payload(payload, evaluation, [task["slug"] for task in selected])
    from utsc_cluster import Client, Env

    client = Client()
    if client.cfg.utorid != lab_user:
        raise ValueError("Lab account does not match the configured user")
    command = [
        "python",
        "-u",
        SCRIPT.as_posix(),
        "worker",
        "--plan",
        PAYLOAD_PLAN.as_posix(),
        "--slug",
        "{slug}",
    ]
    params = [
        {"slug": task["slug"], "evaluation_plan_id": evaluation["id"]}
        for task in selected
    ]
    batch = client.map(
        payload,
        command,
        params,
        env=Env.python(extras=["flows", "cpu"]),
        cpus=cpus,
        memory_gb=memory,
        timeout=timeout,
        machines=machines,
        gpus=0,
        vars={
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        },
    )
    result = {
        "evaluation_plan_id": evaluation["id"],
        "batch_id": batch.id,
        "jobs": [
            {
                "slug": parameter["slug"],
                "job_id": job.id,
                "results_dir": str(job.results_dir),
            }
            for parameter, job in zip(params, batch.jobs)
        ],
    }
    common.write_json(receipt, result)
    return {**preview, "batch_id": batch.id, "receipt": str(receipt)}

def aggregate(comparison, plan, evaluation_id, source_root=ROOT):
    """Write the original aggregate CSV layout only after all conditions exist."""
    import pandas as pd

    tables = {name: [] for name in common.TABLES}
    for slug, groups in condition_groups(plan).items():
        path = comparison / slug / "tables.json"
        if not path.exists():
            raise ValueError("Cannot aggregate incomplete evaluation")
        result = common.read_json(path)
        validate_tables(
            result,
            groups,
            test_conditions(source_root, plan, groups[0]),
            plan["n_starts"],
        )
        for name, values in tables.items():
            values.extend(result[name])
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        if name == "baseline_replay":
            frame = pd.DataFrame(
                columns=[
                    "regime",
                    "family",
                    "n_train",
                    "replicate",
                    "model",
                    "condition",
                    "metric",
                    "delta",
                    "passed",
                ]
            )
        target = comparison / f"{name}.csv"
        temporary = target.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
    common.write_json(
        comparison / "manifest.json",
        {
            **evaluation_request(plan),
            "workflow": training.WORKFLOW,
            "sources": plan["sources"],
            "selections": plan["selections"],
            "n_starts": plan["n_starts"],
            "n_replicates": plan["n_replicates"],
            "persistence": plan["persistence"],
            "distributed_evaluation": evaluation_id,
            "baseline_replay": "Not applicable: both baselines are retrained",
        },
    )
    return {name: len(rows) for name, rows in tables.items()}

def collect(directory, results, comparison, *, local_stopped=False):
    """Validate remote artifacts and preserve completed local evaluations."""
    if not local_stopped:
        raise ValueError(
            "Stop the local evaluator and pass --local-evaluation-stopped before importing"
        )
    directory, comparison = Path(directory).resolve(), Path(comparison).resolve()
    evaluation = load_evaluation(directory / "evaluation_plan.json")
    original = common.read_json(
        directory / "payload/CSCD94" / evaluation["training_plan"]
    )
    if (
        common.read_json(comparison / "cluster_collection.json")["plan_id"]
        != original["id"]
    ):
        raise ValueError("Destination collection belongs to another training plan")
    request_path = comparison / "evaluation_request.json"
    if (
        request_path.exists()
        and common.read_json(request_path) != evaluation["request"]
    ):
        raise ValueError("Evaluation method differs from destination")
    tasks = {task["slug"]: task for task in evaluation["tasks"]}
    imported, preserved = [], []
    environments = []
    candidates = set()
    for root in results:
        if not Path(root).is_dir():
            raise FileNotFoundError(root)
        candidates.update(Path(root).rglob("evaluation_result.json"))
    for path in sorted(candidates):
        envelope = common.read_json(path)
        if (
            envelope.get("format") != FORMAT
            or envelope.get("evaluation_plan_id") != evaluation["id"]
        ):
            raise ValueError("Remote output belongs to another evaluation plan")
        if (
            envelope.get("training_plan_id") != original["id"]
            or envelope["slug"] not in tasks
        ):
            raise ValueError("Unexpected remote evaluation assignment")
        task = tasks[envelope["slug"]]
        for relative, expected in envelope["files"].items():
            if common.digest(safe_path(path.parent, relative)) != expected:
                raise ValueError(f"Remote evaluation artifact changed: {relative}")
        source = path.parent / task["slug"]
        tables = common.read_json(source / "tables.json")
        validate_tables(
            tables,
            task["groups"],
            task["conditions"],
            original["n_starts"],
            task["selected"],
        )
        required = {f"model_{group['base']}.pt" for group in task["groups"]} | {
            "tables.json",
            "data.npz",
        }
        by_model = {group["model"]: group for group in task["groups"]}
        for row in tables["results"]:
            if row["evaluation_status"] == "ok":
                required.add(f"{by_model[row['model']]['base']}_{row['condition']}.npz")
        actual_files = {p.name for p in source.iterdir() if p.is_file()}
        listed_files = {f"{task['slug']}/{name}" for name in actual_files}
        if not required.issubset(actual_files) or listed_files != set(
            envelope["files"]
        ):
            raise ValueError(
                "Remote output omitted required checkpoints or predictions"
            )
        for group in task["groups"]:
            relative = f"evaluation/inputs/{task['slug']}/{group['base']}.pt"
            if (
                common.digest(source / f"model_{group['base']}.pt")
                != task["hashes"][relative]
            ):
                raise ValueError("Remote evaluator returned a different selected model")
        target = comparison / task["slug"]
        if (target / "tables.json").exists():
            validate_tables(
                common.read_json(target / "tables.json"),
                task["groups"],
                task["conditions"],
                original["n_starts"],
                task["selected"],
            )
            preserved.append(task["slug"])
            continue
        target.mkdir(parents=True, exist_ok=True)
        for source_file in source.iterdir():
            if not source_file.is_file() or source_file.name == "tables.json":
                continue
            temporary = target / ("." + source_file.name + ".importing")
            shutil.copy2(source_file, temporary)
            temporary.replace(target / source_file.name)
        for row in tables["parameters"]:
            row["checkpoint"] = str(
                target / f"model_{by_model[row['model']]['base']}.pt"
            )
        common.write_json(target / "tables.json", tables)
        imported.append(task["slug"])
        environments.append(envelope["versions"])
    common.write_json(request_path, evaluation["request"])
    completed = completed_conditions(directory / "payload/CSCD94", comparison, original)
    missing = sorted(set(condition_groups(original)) - set(completed))
    report = {
        "evaluation_plan_id": evaluation["id"],
        "imported": imported,
        "preserved_local": preserved,
        "completed_conditions": len(completed),
        "missing_conditions": missing,
        "worker_environments": environments,
        "ready_for_reporting": not missing,
    }
    if not missing:
        report["tables"] = aggregate(
            comparison, original, evaluation["id"], directory / "payload/CSCD94"
        )
    common.write_json(directory / "collection_report.json", report)
    return {
        key: value
        for key, value in report.items()
        if key
        not in {
            "imported",
            "preserved_local",
            "missing_conditions",
            "worker_environments",
        }
    }

def main(argv=None):
    """Expose safe preparation, explicit submission, worker, and collection stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    preparing = sub.add_parser("prepare")
    preparing.add_argument(
        "--training-plan", type=Path, default=ROOT / RUN / "plan.json"
    )
    preparing.add_argument("--comparison", type=Path, default=ROOT / RUN / "comparison")
    preparing.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    submission = sub.add_parser("submit")
    submission.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    submission.add_argument(
        "--comparison", type=Path, default=ROOT / RUN / "comparison"
    )
    submission.add_argument("--execute", action="store_true")
    submission.add_argument("--lab-user")
    submission.add_argument("--machines", type=int, default=220)
    submission.add_argument("--cpus", type=int, default=1)
    submission.add_argument("--memory", type=float, default=6)
    submission.add_argument("--timeout", default="4h")
    working = sub.add_parser("worker")
    working.add_argument("--plan", type=Path, default=ROOT / PAYLOAD_PLAN)
    working.add_argument("--slug")
    working.add_argument("--output", type=Path)
    collection = sub.add_parser("collect")
    collection.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    collection.add_argument("--results", action="append", type=Path, required=True)
    collection.add_argument(
        "--comparison", type=Path, default=ROOT / RUN / "comparison"
    )
    collection.add_argument("--local-evaluation-stopped", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        result = prepare(ROOT, args.training_plan, args.comparison, args.directory)
    elif args.stage == "submit":
        result = submit(
            args.directory,
            args.comparison,
            execute=args.execute,
            lab_user=args.lab_user,
            machines=args.machines,
            cpus=args.cpus,
            memory=args.memory,
            timeout=args.timeout,
        )
    elif args.stage == "worker":
        evaluation = load_evaluation(args.plan)
        params = json.loads(os.environ.get("UTSC_PARAMS", "{}"))
        if params and params.get("evaluation_plan_id") != evaluation["id"]:
            raise ValueError("Assigned evaluation plan mismatch")
        output = args.output or os.environ.get("OUT_DIR")
        if not output:
            raise ValueError("Provide OUT_DIR or --output")
        result = worker(ROOT, evaluation, args.slug or params.get("slug"), output)
    else:
        result = collect(
            args.directory,
            args.results,
            args.comparison,
            local_stopped=args.local_evaluation_stopped,
        )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
