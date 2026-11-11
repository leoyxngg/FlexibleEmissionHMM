import argparse
from concurrent.futures import as_completed
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "mean": "data/synthetic/gmm_synthetic_experiments/20261024T212804Z-4r1by0vo",
    "volatility": (
        "data/synthetic/volatility_markov_experiments/20261020T110835Z-8lz5ptfh"
    ),
}
SAVED_RUN = (
    "data/synthetic/nf_spline_experiments/"
    "four_emissions_p995_hmmlearn_m10_l2_b8-16_s2_i150"
    "_batch2048_float64_cpu8x1_v2_patience5_v2"
)
BASES = {"normal": "RQ-NSF Gaussian base", "student_t": "RQ-NSF Student-t base"}
TABLES = (
    "results", "attempts", "per_state", "confusion", "baseline_replay",
    "parameters", "coverage",
)


def read_json(path):
    """Read a JSON document."""
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Publish a JSON document atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def digest(path):
    """Hash a file without reading the whole file into memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    """Hash a JSON value independently of dictionary insertion order."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def project_path(root, relative):
    """Resolve a path and require it to stay inside the project."""
    path = (Path(root) / relative).resolve()
    path.relative_to(Path(root).resolve())
    return path


def start_seed(group, start):
    """Match the existing notebook's final-training initialization seeds."""
    payload = json.dumps([*group["seed_parts"], start]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def build_plan(
    root, manifest, *, n_starts=100, chunk_size=10, regimes=None,
    families=None, replicates=None, sources=None,
):
    """Freeze data, implementations, settings, and global initialization IDs."""
    if n_starts < 1 or chunk_size < 1:
        raise ValueError("Start count and chunk size must be positive")
    root = Path(root).resolve()
    manifest_path = project_path(root, manifest)
    selections = read_json(manifest_path)["selections"]
    sources = dict(SOURCES if sources is None else sources)
    if set(sources) != {"mean", "volatility"}:
        raise ValueError("Provide both saved sources, as required by load_sources")
    source_hashes = {}
    groups = []
    seen_regimes, seen_families, seen_replicates = set(), set(), set()
    for regime, relative in sources.items():
        directory = project_path(root, relative)
        saved = read_json(directory / "manifest.json")
        settings = saved["settings"]
        index = read_json(directory / "dataset_index.json")
        names = {
            "manifest.json", "dataset_index.json", "parameters.json",
            "results.json", "attempts.json", "per_state.json",
            *(row["file"] for row in index),
        }
        perturbation = "mean_regime_perturbation/results.csv"
        if (directory / perturbation).is_file():
            names.add(perturbation)
        for name in sorted(names):
            path = project_path(root, Path(relative) / name)
            source_hashes[path.relative_to(root).as_posix()] = digest(path)
        if regimes and regime not in regimes:
            continue
        seen_regimes.add(regime)
        for family in saved["families"]:
            if families and family not in families:
                continue
            seen_families.add(family)
            for n_train in settings["train_sizes"]:
                for replicate in range(settings["n_replicates"]):
                    if replicates is not None and replicate not in replicates:
                        continue
                    seen_replicates.add(replicate)
                    slug = (
                        f"{regime}_{family.replace(' ', '_')}_"
                        f"{n_train}_{replicate}"
                    )
                    for base, model in BASES.items():
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
                        chosen = matches[0]
                        if (chosen["device"] != "cpu"
                                or chosen["dtype"] != "float64"):
                            raise ValueError(
                                "This workflow requires CPU float64 settings"
                            )
                        groups.append({
                            "id": f"{slug}/{base}", "slug": slug,
                            "regime": regime, "family": family,
                            "n_train": n_train, "replicate": replicate,
                            "base": base, "model": model, "settings": chosen,
                            "max_iter": settings["max_iter"],
                            "tolerance": (
                                n_train * settings["tol_per_observation"]
                            ),
                            "seed_parts": [
                                settings["master_seed"], "final_flow", regime,
                                family, replicate, n_train,
                            ],
                        })
    for requested, found, label in (
        (regimes, seen_regimes, "regime"),
        (families, seen_families, "family"),
        (replicates, seen_replicates, "replicate"),
    ):
        if requested is not None and set(requested) - found:
            raise ValueError(f"Unknown {label}: {set(requested) - found}")
    if not groups or len({group["id"] for group in groups}) != len(groups):
        raise ValueError("The plan must contain distinct, nonempty flow groups")
    implementation = [
        "scripts/flow_cluster.py", "notebooks/nf_hmmlearn_comparison.py",
        "notebooks/synthetic_emission_comparison.py",
        "notebooks/volatility_markov_comparison.py",
        "pyproject.toml", "uv.lock", ".python-version",
    ]
    implementation += [
        path.relative_to(root).as_posix()
        for path in sorted((root / "src").rglob("*.py"))
    ]
    plan = {
        "format_version": 1, "n_starts": n_starts, "chunk_size": chunk_size,
        "sources": sources, "source_hashes": source_hashes,
        "implementation_hashes": {
            name: digest(project_path(root, name)) for name in implementation
        },
        "settings_manifest": manifest_path.relative_to(root).as_posix(),
        "settings_manifest_hash": digest(manifest_path),
        "selections": selections, "groups": groups,
        "selection": "Highest finite training LL; earliest start breaks ties",
        "baselines": "Replay saved Gaussian/GMM fits; do not retrain baselines",
    }
    plan["id"] = object_digest(plan)
    return plan


def validate_plan(root, plan):
    """Reject changed inputs or implementations before using a frozen plan."""
    content = {key: value for key, value in plan.items() if key != "id"}
    if plan.get("format_version") != 1 or object_digest(content) != plan["id"]:
        raise ValueError("Invalid or modified plan; create a fresh plan")
    hashes = {
        **plan["source_hashes"], **plan["implementation_hashes"],
        plan["settings_manifest"]: plan["settings_manifest_hash"],
    }
    for relative, expected in hashes.items():
        path = project_path(root, relative)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Input changed or missing: {relative}")


def tasks(plan):
    """Return disjoint chunks covering every requested start exactly once."""
    output = []
    for number, group in enumerate(plan["groups"]):
        for begin in range(0, plan["n_starts"], plan["chunk_size"]):
            output.append({
                "id": f"g{number:04d}-s{begin:06d}", "group": group,
                "begin": begin,
                "end": min(begin + plan["chunk_size"], plan["n_starts"]),
            })
    return output


def summary(plan):
    """Describe the planned fitting budget without accessing the cluster."""
    return {
        "plan_id": plan["id"], "flow_groups": len(plan["groups"]),
        "starts_per_group": plan["n_starts"],
        "total_initializations": len(plan["groups"]) * plan["n_starts"],
        "chunks": len(tasks(plan)), "starts_per_chunk": plan["chunk_size"],
        "baselines": plan["baselines"],
    }


def _experiment(root):
    for directory in (Path(root) / "src", Path(root) / "notebooks"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import nf_hmmlearn_comparison
    return nf_hmmlearn_comparison


def _attempt_artifact(directory, group, start):
    path = Path(directory) / "starts" / f"start_{start}.json"
    row = read_json(path)
    if (row.get("start") != start or row.get("seed") != start_seed(group, start)
            or row.get("status") not in ("ok", "failed")):
        raise ValueError(f"Inconsistent start record: {path}")
    artifact = {"start": start, "record_hash": digest(path)}
    if row["status"] == "ok":
        if not math.isfinite(row.get("train_ll", float("nan"))):
            raise ValueError(f"Nonfinite likelihood: {path}")
        artifact["checkpoint_hash"] = digest(path.with_suffix(".pt"))
    return artifact, row


def run_worker(root, plan, task_id, output, *, workers=1, start_ids=None):
    """Train one assigned chunk and checkpoint each finished initialization."""
    validate_plan(root, plan)
    task = next((item for item in tasks(plan) if item["id"] == task_id), None)
    if task is None or workers < 1:
        raise ValueError("Provide a valid task ID and positive worker count")
    assigned = list(range(task["begin"], task["end"]))
    requested = assigned if start_ids is None else sorted(set(start_ids))
    if not requested or not set(requested).issubset(assigned):
        raise ValueError("Requested starts must belong to the assigned chunk")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "chunk.json"
    identity = {
        "plan_id": plan["id"], "task_id": task_id,
        "requested_starts": requested,
    }
    if metadata.exists():
        previous = read_json(metadata)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Output directory belongs to a different task")
    elif any(output.iterdir()):
        raise ValueError("Use an empty output directory for a new chunk")
    experiment = _experiment(root)
    sources = experiment.load_sources(root, plan["sources"])
    group = task["group"]
    frame = sources[group["regime"]].data(
        group["replicate"], group["family"], ("train",),
    )["train"]
    train = frame[["x"]].to_numpy()[:group["n_train"]]
    if len(train) != group["n_train"]:
        raise ValueError("Saved training prefix is too short")
    starts = output / "starts"
    starts.mkdir(exist_ok=True)
    report = {
        **identity, "group_id": group["id"], "begin": task["begin"],
        "end": task["end"], "complete": False, "artifacts": [],
        "versions": {name: version(name) for name in ("torch", "numpy", "hmmlearn")},
        "job_id": os.environ.get("UTSC_JOB_ID"),
    }
    write_json(metadata, report)
    jobs = []
    for start in requested:
        saved = starts / f"start_{start}.json"
        if saved.exists():
            _attempt_artifact(output, group, start)
        jobs.append((
            train, experiment.FlowSettings(**group["settings"]), start,
            group["max_iter"], group["tolerance"], group["seed_parts"],
            str(starts),
        ))
    with experiment._cpu_executor(workers, 1) as executor:
        if executor is None:
            iterator = map(experiment._fit_single_start, jobs)
        else:
            futures = [executor.submit(experiment._fit_single_start, job)
                       for job in jobs]
            iterator = (future.result() for future in as_completed(futures))
        for row in iterator:
            artifact, row = _attempt_artifact(output, group, row["start"])
            report["artifacts"].append(artifact)
            write_json(metadata, report)
            print(
                f"{task_id}: start {row['start']} {row['status']} "
                f"({len(report['artifacts'])}/{len(jobs)})", flush=True,
            )
    report["complete"] = True
    write_json(metadata, report)
    return report


def submission_spec(root, plan_path, plan, *, cpus=4, memory_gb=6, timeout="4h",
                    machines=2, task_ids=None, starts_by_task=None):
    """Prepare a DataCluster submission without importing its client."""
    if cpus < 1 or memory_gb <= 0 or machines < 1:
        raise ValueError("Request positive CPU, memory, and machine limits")
    relative = Path(plan_path).resolve().relative_to(Path(root).resolve()).as_posix()
    all_tasks = tasks(plan)
    if task_ids is not None:
        unknown = set(task_ids) - {task["id"] for task in all_tasks}
        if unknown:
            raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
        all_tasks = [task for task in all_tasks if task["id"] in task_ids]
    if not all_tasks:
        raise ValueError("No tasks selected for submission")
    # DataCluster uses gitignore syntax: anchor names such as pyproject.toml
    # so they cannot also match files inside the excluded local .venv.
    includes = [f"/{name}" for name in sorted({
        relative, plan["settings_manifest"], *plan["source_hashes"],
        *plan["implementation_hashes"],
    })]
    return {
        "project": str(Path(root).resolve()),
        "command": [
            "python", "-u", "scripts/flow_cluster.py", "worker",
            "--plan", relative, "--workers", str(cpus),
        ],
        "params": [{
            "task_id": task["id"], "plan_id": plan["id"],
            **({"start_ids": starts_by_task[task["id"]]}
               if starts_by_task is not None else {}),
        } for task in all_tasks],
        "cpus": cpus, "memory_gb": memory_gb, "timeout": timeout,
        "machines": machines, "gpus": 0, "include": includes,
        "exclude": [
            "data/", "notebooks/", "docs/", "leo_slides/", "tests/",
            "cluster_runs/",
        ],
        "vars": {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        },
    }


def submit_plan(root, plan_path, plan, *, execute=False, datacluster=None,
                receipt=None, lab_user=None, **resources):
    """Preview a batch or explicitly submit it through DataCluster."""
    validate_plan(root, plan)
    spec = submission_spec(root, plan_path, plan, **resources)
    preview = {**summary(plan), "submission": spec, "execute": execute}
    if not execute:
        return preview
    if not lab_user:
        raise ValueError("Provide --lab-user to check the intended lab account")
    receipt = Path(receipt or Path(plan_path).with_name("submission.json"))
    if receipt.exists():
        raise FileExistsError("Use a new --receipt for another submission")
    if datacluster:
        sys.path.insert(0, str(Path(datacluster).expanduser().resolve() / "src"))
    try:
        from utsc_cluster import Client, Env
    except ImportError as exc:
        raise RuntimeError("Provide --datacluster /path/to/DataCluster") from exc
    client = Client()
    # Refuse a copied account configuration before its first network operation.
    if client.cfg.utorid != lab_user:
        raise ValueError("--lab-user does not match DataCluster's configured account")
    selected_params = spec["params"]
    project = spec.pop("project")
    command = spec.pop("command")
    batch = client.map(project, command, env=Env.python(extras=["flows", "cpu"]),
                       **spec)
    result = {
        "plan_id": plan["id"], "batch_id": batch.id,
        "jobs": [{"task_id": params["task_id"], "job_id": job.id,
                  "results_dir": str(job.results_dir)}
                 for params, job in zip(selected_params, batch.jobs)],
    }
    write_json(receipt, result)
    return result


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cache_directory(output, group):
    return project_path(output, f"{group['slug']}/starts_{group['base']}")


def _verify_merged(output, plan, records):
    groups = {group["id"]: group for group in plan["groups"]}
    for key, item in records.items():
        group = groups[item["group_id"]]
        directory = _cache_directory(output, group)
        record = directory / f"start_{item['start']}.json"
        if digest(record) != item["record_hash"]:
            raise ValueError(f"Collected record changed: {record}")
        if item.get("checkpoint_hash"):
            if digest(record.with_suffix(".pt")) != item["checkpoint_hash"]:
                raise ValueError(f"Collected checkpoint changed: {record}")
        if key != f"{group['id']}/{item['start']}":
            raise ValueError("Invalid merged record identity")


def collect_results(root, plan, result_roots, output):
    """Validate downloaded chunks and merge checkpoints into notebook layout."""
    validate_plan(root, plan)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "cluster_collection.json"
    index = {"plan_id": plan["id"], "records": {}, "versions": None}
    if index_path.exists():
        index = read_json(index_path)
        if index["plan_id"] != plan["id"]:
            raise ValueError("Collection directory belongs to another plan")
        _verify_merged(output, plan, index["records"])
    elif any(output.iterdir()):
        raise ValueError("Use an empty directory for a new collection")
    task_table = {task["id"]: task for task in tasks(plan)}
    candidates = set()
    for location in result_roots:
        location = Path(location).expanduser()
        if not location.is_dir():
            raise FileNotFoundError(f"Results directory not found: {location}")
        candidates.update(location.rglob("chunk.json"))
    for metadata in sorted(candidates):
        chunk = read_json(metadata)
        if chunk.get("plan_id") != plan["id"]:
            raise ValueError(f"Chunk belongs to another plan: {metadata}")
        task = task_table.get(chunk.get("task_id"))
        if task is None:
            raise ValueError(f"Unknown chunk: {metadata}")
        group = task["group"]
        if (chunk.get("group_id"), chunk.get("begin"), chunk.get("end")) != (
            group["id"], task["begin"], task["end"],
        ):
            raise ValueError(f"Wrong chunk identity: {metadata}")
        if index["versions"] not in (None, chunk["versions"]):
            raise ValueError("Worker package versions differ across chunks")
        index["versions"] = chunk["versions"]
        assigned = set(range(task["begin"], task["end"]))
        requested = set(chunk.get("requested_starts", assigned))
        if not requested or not requested.issubset(assigned):
            raise ValueError(f"Invalid requested starts: {metadata}")
        seen = set()
        for expected in chunk["artifacts"]:
            start = expected["start"]
            if start in seen or start not in requested:
                raise ValueError(f"Duplicate or out-of-range start: {metadata}")
            seen.add(start)
            artifact, row = _attempt_artifact(metadata.parent, group, start)
            if artifact != expected:
                raise ValueError(f"Chunk artifact is missing or changed: {metadata}")
            key = f"{group['id']}/{start}"
            item = {**artifact, "group_id": group["id"], "status": row["status"]}
            previous = index["records"].get(key)
            if previous and previous != item:
                raise ValueError(f"Conflicting duplicate initialization: {key}")
            if previous:
                continue
            directory = _cache_directory(output, group)
            directory.mkdir(parents=True, exist_ok=True)
            source = metadata.parent / "starts" / f"start_{start}.json"
            shutil.copy2(source, directory / source.name)
            if row["status"] == "ok":
                shutil.copy2(source.with_suffix(".pt"), directory / f"start_{start}.pt")
            index["records"][key] = item
        if chunk.get("complete") and seen != requested:
            raise ValueError(f"Completed chunk has missing records: {metadata}")
    write_json(index_path, index)
    rows, winners = [], []
    for group in plan["groups"]:
        valid = []
        for start in range(plan["n_starts"]):
            if f"{group['id']}/{start}" not in index["records"]:
                continue
            record = read_json(_cache_directory(output, group) / f"start_{start}.json")
            row = {
                "group_id": group["id"], "regime": group["regime"],
                "family": group["family"], "replicate": group["replicate"],
                "n_train": group["n_train"], "model": group["model"],
                **{key: value for key, value in record.items() if key != "history"},
            }
            rows.append(row)
            if row["status"] == "ok":
                valid.append(row)
        if valid:
            best = max(valid, key=lambda row: (row["train_ll"], -row["start"]))
            best["selected"] = True
            winners.append(dict(best))
    missing_starts = {}
    for task in tasks(plan):
        missing = [start for start in range(task["begin"], task["end"])
                   if f"{task['group']['id']}/{start}" not in index["records"]]
        if missing:
            missing_starts[task["id"]] = missing
    missing_tasks = list(missing_starts)
    expected = len(plan["groups"]) * plan["n_starts"]
    report = {
        "plan_id": plan["id"], "expected": expected, "recorded": len(rows),
        "successful": sum(row["status"] == "ok" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "missing": expected - len(rows), "missing_tasks": missing_tasks,
        "missing_starts": missing_starts,
        "ready_for_evaluation": (
            not missing_tasks and len(winners) == len(plan["groups"])
        ),
    }
    write_json(output / "collection_report.json", report)
    _write_csv(output / "cluster_attempts.csv", rows)
    _write_csv(output / "cluster_winners.csv", winners)
    return report


def evaluate_results(root, plan, output):
    """Run the notebook evaluation while forbidding any new training."""
    validate_plan(root, plan)
    output = Path(output)
    index = read_json(output / "cluster_collection.json")
    if index["plan_id"] != plan["id"]:
        raise ValueError("Collected results belong to another plan")
    _verify_merged(output, plan, index["records"])
    expected = {f"{group['id']}/{start}" for group in plan["groups"]
                for start in range(plan["n_starts"])}
    if set(index["records"]) != expected:
        raise ValueError("Collect every requested start before evaluating")
    for group in plan["groups"]:
        if not any(index["records"][f"{group['id']}/{start}"]["status"] == "ok"
                   for start in range(plan["n_starts"])):
            raise ValueError(f"All initializations failed for {group['id']}")
    experiment = _experiment(root)
    sources = experiment.load_sources(root, plan["sources"])
    regimes = {group["regime"] for group in plan["groups"]}
    run = experiment.run_comparison(
        {key: source for key, source in sources.items() if key in regimes},
        output, plan["selections"], n_starts=plan["n_starts"],
        replicates=sorted({group["replicate"] for group in plan["groups"]}),
        families=sorted({group["family"] for group in plan["groups"]}),
        n_jobs=1, torch_threads=1, cached_only=True,
    )
    return {name: len(table) for name, table in run.items()}


def main(argv=None):
    """Run a local planning command or an explicitly selected execution stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="Freeze a local plan; no cluster access")
    plan_parser.add_argument("--output", required=True, type=Path)
    plan_parser.add_argument("--manifest", default=f"{SAVED_RUN}/manifest.json")
    plan_parser.add_argument("--starts", type=int, default=100)
    plan_parser.add_argument("--chunk-size", type=int, default=10)
    plan_parser.add_argument("--regime", action="append", choices=list(SOURCES))
    plan_parser.add_argument("--family", action="append")
    plan_parser.add_argument("--replicate", action="append", type=int)
    for command in ("submit", "worker", "collect", "evaluate"):
        stage = sub.add_parser(command)
        stage.add_argument("--plan", required=True, type=Path)
        if command == "submit":
            stage.add_argument("--execute", action="store_true")
            stage.add_argument("--datacluster", type=Path)
            stage.add_argument(
                "--lab-user", help="Expected lab username; required to execute",
            )
            stage.add_argument("--receipt", type=Path)
            stage.add_argument("--machines", type=int, default=2)
            stage.add_argument("--cpus", type=int, default=4)
            stage.add_argument("--memory", type=float, default=6)
            stage.add_argument("--timeout", default="4h")
            stage.add_argument(
                "--tasks-from", type=Path,
                help="Resubmit missing tasks from collection_report.json",
            )
        elif command == "worker":
            stage.add_argument("--task-id")
            stage.add_argument("--workers", type=int, default=1)
            stage.add_argument("--output", type=Path)
        elif command == "collect":
            stage.add_argument("--results", action="append", required=True, type=Path)
            stage.add_argument("--output", required=True, type=Path)
        else:
            stage.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = PROJECT_ROOT
    if args.command == "plan":
        plan = build_plan(
            root, args.manifest, n_starts=args.starts, chunk_size=args.chunk_size,
            regimes=args.regime, families=args.family, replicates=args.replicate,
        )
        if args.output.exists() and read_json(args.output) != plan:
            raise ValueError(
                "Plan already exists with different settings; use a new path"
            )
        write_json(args.output, plan)
        result = summary(plan)
    else:
        plan = read_json(args.plan)
        if args.command == "submit":
            task_ids = None
            starts_by_task = None
            if args.tasks_from:
                report = read_json(args.tasks_from)
                if report["plan_id"] != plan["id"]:
                    raise ValueError("Resubmission report belongs to another plan")
                task_ids = report["missing_tasks"]
                starts_by_task = report["missing_starts"]
            result = submit_plan(
                root, args.plan, plan, execute=args.execute,
                datacluster=args.datacluster, receipt=args.receipt,
                lab_user=args.lab_user,
                cpus=args.cpus, memory_gb=args.memory, timeout=args.timeout,
                machines=args.machines, task_ids=task_ids,
                starts_by_task=starts_by_task,
            )
        elif args.command == "worker":
            params = json.loads(os.environ.get("UTSC_PARAMS", "{}"))
            if params and params.get("plan_id") != plan["id"]:
                raise ValueError("Worker params belong to another plan")
            task_id = args.task_id or params.get("task_id")
            output = args.output or os.environ.get("OUT_DIR")
            if not output:
                raise ValueError("Provide OUT_DIR or --output")
            result = run_worker(
                root, plan, task_id, output, workers=args.workers,
                start_ids=params.get("start_ids"),
            )
        elif args.command == "collect":
            result = collect_results(root, plan, args.results, args.output)
        else:
            result = evaluate_results(root, plan, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
