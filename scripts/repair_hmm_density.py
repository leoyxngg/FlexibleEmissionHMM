import argparse  # noqa: I001
import json
import shutil
import warnings
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

import flow_cluster as common
import hmm_cluster as training

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "cluster_runs/hmm_100x100"
METHOD = "analytic_mixture_moments_component_resolved_kl_v1"
IDENTITY = ["regime", "family", "n_train", "replicate", "model"]
METRICS = [
    "emission_kl",
    "kl_quad_error_max",
    "mean_rmse",
    "variance_rmse",
    "moment_quad_error_max",
]
OLD_ERROR = "FloatingPointError: Fitted flow density failed normalization check"

def mixture_moments(weights, means, variances):
    """Compute exact Gaussian-mixture moments after validating each component."""
    weights, means, variances = (
        np.asarray(value, dtype=float) for value in (weights, means, variances)
    )
    if (
        weights.ndim != 2
        or means.shape != weights.shape
        or variances.shape != weights.shape
        or not all(np.isfinite(value).all() for value in (weights, means, variances))
        or (weights < 0).any()
        or (variances <= 0).any()
        or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Invalid Gaussian-mixture parameters")
    fitted_means = np.sum(weights * means, axis=1)
    fitted_variances = np.sum(
        weights * (variances + (means - fitted_means[:, None]) ** 2), axis=1
    )
    return fitted_means, fitted_variances

def mixture_recovery(source, family, weights, means, variances):
    """Integrate KL with component-aware intervals and use analytic moments."""
    weights, means, variances = (
        np.asarray(value, dtype=float) for value in (weights, means, variances)
    )
    fitted_means, fitted_variances = mixture_moments(weights, means, variances)
    true_means, true_variances = source.true_moments()
    options = {
        "epsabs": source.settings["quad_epsabs"],
        "epsrel": source.settings["quad_epsrel"],
        "limit": source.settings["quad_limit"],
    }
    kl, errors = [], []
    for state in range(len(true_means)):
        active = weights[state] > 0
        log_weights = np.log(weights[state, active])
        centers = means[state, active]
        scales = np.sqrt(variances[state, active])

        def integrand(
            x, state=state, log_weights=log_weights, centers=centers, scales=scales
        ):
            """Weight the fitted log-density ratio by the true density."""
            log_p = float(source.true_log_density(family, state, x))
            if not np.isfinite(log_p) or log_p < -745:
                return 0.0
            log_q = special.logsumexp(
                log_weights + stats.norm.logpdf(x, centers, scales)
            )
            return float(np.exp(log_p) * (log_p - log_q))

        lower = -np.inf
        if family == "Skewed Gamma":
            lower = true_means[state] - np.sqrt(
                true_variances[state] * source.shape_parameters["gamma_shape"]
            )
        std = np.sqrt(true_variances[state])
        central = np.linspace(
            true_means[state] - 12 * std, true_means[state] + 12 * std, 49
        )
        components = (
            centers[:, None]
            + scales[:, None] * np.array([-12, -8, -4, -1, 0, 1, 4, 8, 12])
        ).ravel()
        breaks = sorted({float(x) for x in np.r_[central, components] if x > lower})
        breaks = [lower, *breaks, np.inf]
        with warnings.catch_warnings():
            warnings.simplefilter("error", integrate.IntegrationWarning)
            pieces = [
                integrate.quad(integrand, a, b, **options) for a, b in pairwise(breaks)
            ]
        value, error = np.sum(pieces, axis=0)
        if (
            not np.isfinite([value, error]).all()
            or value < -max(1e-6, 10 * error)
            or error > max(1e-7, 100 * options["epsrel"] * abs(value))
        ):
            raise FloatingPointError("Repaired KL did not meet integration checks")
        kl.append(float(value))
        errors.append(float(error))
    metrics = {
        "emission_kl": float(np.mean(kl)),
        "kl_quad_error_max": float(max(errors)),
        "mean_rmse": float(np.sqrt(np.mean((fitted_means - true_means) ** 2))),
        "variance_rmse": float(
            np.sqrt(np.mean((fitted_variances - true_variances) ** 2))
        ),
        # No numerical quadrature is used for fitted moments in this method.
        "moment_quad_error_max": 0.0,
    }
    if not np.isfinite(list(metrics.values())).all():
        raise FloatingPointError("Nonfinite repaired recovery metrics")
    return metrics, kl

def identity(row):
    """Return a unique selected-model identity."""
    return tuple(row[key] for key in IDENTITY)

def repair_tables(tables, updates):
    """Update only failed recovery fields and their per-state KL measurements."""
    for row in tables["results"]:
        key = identity(row)
        if key not in updates:
            continue
        if (
            row["recovery_status"] != "failed"
            or OLD_ERROR not in row["evaluation_warnings"]
        ):
            raise ValueError("Expected the original density-normalization failure")
        row["density_repair_previous_warnings"] = row["evaluation_warnings"]
        row["evaluation_warnings"] = [
            value for value in row["evaluation_warnings"] if value != OLD_ERROR
        ]
        row.update(updates[key][0], recovery_status="ok", recovery_method=METHOD)
    for row in tables["per_state"]:
        if identity(row) in updates:
            row["emission_kl"] = updates[identity(row)][1][row["state"]]
    return tables

def publish(directory, audit, report):
    """Publish staged tables safely and resume an interrupted publication."""
    for relative, hashes in report["changed_files"].items():
        current = common.digest(directory / relative)
        if current not in {hashes["before"], hashes["after"]}:
            raise ValueError(f"Destination changed since preparation: {relative}")
    for relative, expected in report["protected_files"].items():
        if common.digest(directory / relative) != expected:
            raise ValueError(f"Protected artifact changed: {relative}")
    for relative, hashes in report["changed_files"].items():
        staged = audit / "staged" / relative
        if common.digest(staged) != hashes["after"]:
            raise ValueError(f"Staged repair changed: {relative}")
        target = directory / relative
        temporary = target.with_name("." + target.name + ".density-repair")
        shutil.copy2(staged, temporary)
        temporary.replace(target)
    report["status"] = "complete"
    common.write_json(audit / "report.json", report)

def repair(directory, plan_path, *, apply=False):
    """Back up and repair failed baseline diagnostics while preserving fit results."""
    directory = Path(directory).resolve()
    plan = common.read_json(plan_path)
    common.validate_plan(ROOT, plan)
    if common.read_json(directory / "manifest.json")["plan_id"] != plan["id"]:
        raise ValueError("Comparison belongs to another training plan")
    audit = directory / "density_repair"
    if (audit / "report.json").exists():
        report = common.read_json(audit / "report.json")
        if report["plan_id"] != plan["id"]:
            raise ValueError("Repair belongs to another training plan")
        if apply:
            publish(directory, audit, report)
        return {
            "status": report["status"],
            "models": len(report["repairs"]),
            "audit": str(audit),
        }
    groups = {identity(group): group for group in plan["groups"]}
    affected = {}
    for slug in dict.fromkeys(group["slug"] for group in plan["groups"]):
        tables = common.read_json(directory / slug / "tables.json")
        keys = {
            identity(row)
            for row in tables["results"]
            if row["recovery_status"] == "failed"
        }
        if keys:
            affected[slug] = (tables, keys)
    if not apply:
        return {
            "status": "preview",
            "models": sum(len(k) for _, k in affected.values()),
            "conditions": len(affected),
            "apply": False,
        }
    if not affected:
        return {"status": "nothing_to_repair", "models": 0}
    if audit.exists():
        raise FileExistsError("Uncommitted repair exists; inspect it before continuing")
    implementation = training.runtime(ROOT)
    sources = implementation.load_sources(ROOT, plan["sources"])
    collection = common.read_json(directory / "cluster_collection.json")
    if collection["plan_id"] != plan["id"]:
        raise ValueError("Training collection identity mismatch")
    updates, protected, repairs = {}, {}, []
    staged_tables = {}
    with implementation.experiment._cpu_executor(1, 1):
        for slug, (tables, keys) in affected.items():
            for key in sorted(keys):
                group = groups[key]
                if group["base"] not in {"gaussian", "gmm"}:
                    raise ValueError("This repair only supports Gaussian baselines")
                parameter = next(
                    row for row in tables["parameters"] if identity(row) == key
                )
                checkpoint = directory / slug / f"model_{group['base']}.pt"
                index_key = f"{group['id']}/{parameter['selected_start']}"
                if (
                    common.digest(checkpoint)
                    != collection["records"][index_key]["checkpoint_hash"]
                ):
                    raise ValueError("Selected checkpoint failed its training hash")
                model = implementation.restore_model(checkpoint, None, group)
                mapping = np.asarray(parameter["mapping_learned_to_true"])
                if sorted(mapping) != [0, 1, 2]:
                    raise ValueError("Invalid saved state alignment")
                components = implementation.experiment.reference.fitted_components(
                    model, np.argsort(mapping)
                )
                for name, value in zip(["weights", "means", "variances"], components):
                    if not np.array_equal(value, parameter[name]):
                        raise ValueError("Saved parameters differ from checkpoint")
                updates[key] = mixture_recovery(
                    sources[group["regime"]], group["family"], *components
                )
                repairs.append(
                    {
                        **dict(zip(IDENTITY, key)),
                        "selected_start": parameter["selected_start"],
                        "metrics": updates[key][0],
                        "state_kl": updates[key][1],
                    }
                )
                print(f"Repaired {slug}: {group['model']}", flush=True)
            staged_tables[slug] = repair_tables(tables, updates)
            for path in (directory / slug).iterdir():
                if path.is_file() and path.name != "tables.json":
                    protected[path.relative_to(directory).as_posix()] = common.digest(
                        path
                    )
    for name in common.TABLES:
        if name not in {"results", "per_state"}:
            relative = f"{name}.csv"
            protected[relative] = common.digest(directory / relative)
    audit.mkdir()
    changed = {}

    def stage(relative, writer):
        original = directory / relative
        backup, staged = audit / "backup" / relative, audit / "staged" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, backup)
        writer(staged)
        changed[relative] = {
            "before": common.digest(backup),
            "after": common.digest(staged),
        }

    for slug, tables in staged_tables.items():
        stage(
            f"{slug}/tables.json",
            lambda path, value=tables: common.write_json(path, value),
        )
    for name in ["results", "per_state"]:
        old = pd.read_csv(directory / f"{name}.csv", float_precision="round_trip")
        if name == "results":
            failed = {
                identity(row)
                for row in old.to_dict("records")
                if row["recovery_status"] == "failed"
            }
            if failed != set(updates):
                raise ValueError("Aggregate failures differ from per-condition tables")
        frame = old.copy()
        expected_rows = set()
        for slug, tables in staged_tables.items():
            for row in tables[name]:
                key = identity(row)
                if key not in updates:
                    continue
                mask = np.ones(len(frame), dtype=bool)
                for column in IDENTITY + ["condition"]:
                    mask &= frame[column].eq(row[column]).to_numpy()
                if name == "per_state":
                    mask &= frame.state.eq(row["state"]).to_numpy()
                if mask.sum() != 1:
                    raise ValueError("Aggregate row identity is missing or duplicated")
                index = frame.index[mask][0]
                expected_rows.add(index)
                fields = (
                    METRICS
                    + [
                        "recovery_status",
                        "recovery_method",
                        "evaluation_warnings",
                        "density_repair_previous_warnings",
                    ]
                    if name == "results"
                    else ["emission_kl"]
                )
                for field in fields:
                    if field not in frame:
                        frame[field] = np.nan if field in METRICS else ""
                    value = row[field]
                    frame.at[index, field] = (
                        repr(value) if isinstance(value, list) else value
                    )
        untouched = old.index.difference(list(expected_rows))
        pd.testing.assert_frame_equal(
            old.loc[untouched], frame.loc[untouched, old.columns]
        )
        keep = old.columns.difference(
            METRICS + ["recovery_status", "evaluation_warnings"]
        )
        pd.testing.assert_frame_equal(old[keep], frame[keep])
        stage(f"{name}.csv", lambda path, value=frame: value.to_csv(path, index=False))
        written = pd.read_csv(
            audit / "staged" / f"{name}.csv", float_precision="round_trip"
        )
        pd.testing.assert_frame_equal(old[keep], written[keep], check_exact=True)
    manifest = common.read_json(directory / "manifest.json")
    manifest["density_repair"] = {
        "method": METHOD,
        "models": len(repairs),
        "audit": "density_repair/report.json",
    }
    stage("manifest.json", lambda path: common.write_json(path, manifest))
    report = {
        "status": "prepared",
        "plan_id": plan["id"],
        "method": METHOD,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repairs": repairs,
        "changed_files": changed,
        "protected_files": protected,
        "script_hash": common.digest(__file__),
        "notes": [
            "Moment quadrature error is zero because moments are analytic.",
            "Prediction metrics, trained checkpoints, and fit records are unchanged.",
        ],
    }
    common.write_json(audit / "report.json", report)
    publish(directory, audit, report)
    return {
        "status": "complete",
        "models": len(repairs),
        "result_rows": len(repairs) * 4,
        "audit": str(audit),
    }

def main():
    """Preview or apply the local diagnostic repair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RUN / "comparison")
    parser.add_argument("--plan", type=Path, default=RUN / "plan.json")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args.input, args.plan, apply=args.apply), indent=2))

if __name__ == "__main__":
    main()
