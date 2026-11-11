import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from utsc_cluster import hosts


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "cluster_runs/hmm_100x100"
AUDIT = RUN / "storage_recovery"
FINAL = {"SUCCESS", "FAILURE", "REVOKED"}


def save_status(value):
    """Publish the recovery monitor status atomically."""
    value["checked_at"] = datetime.now().astimezone().isoformat()
    target = AUDIT / "completion.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(target)


def main():
    """Wait for the retry batch, then validate and merge its downloaded output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=float, default=30)
    args = parser.parse_args()
    if args.poll < 5:
        raise ValueError("Poll at least five seconds apart")
    receipt = json.loads((RUN / "submission_storage_retry.json").read_text())
    if len(receipt["jobs"]) != 1766:
        raise ValueError("Unexpected recovery batch size")
    r = hosts.redis_client()
    status = {"batch_id": receipt["batch_id"], "stage": "waiting_for_training"}
    while True:
        try:
            pipe = r.pipeline(transaction=False)
            for job in receipt["jobs"]:
                pipe.hget(f"utsc:job:{job['job_id']}", "phase")
            counts = Counter(phase or "UNKNOWN" for phase in pipe.execute())
        except Exception as exc:
            save_status({**status, "connection_error": str(exc)[:300]})
            time.sleep(args.poll)
            continue
        status["counts"] = dict(counts)
        save_status(status.copy())
        if sum(value for phase, value in counts.items() if phase in FINAL) == 1766:
            break
        time.sleep(args.poll)

    # The existing index already preserves the validated original outputs.
    # The collector revalidates it and adds only the new recovery directory.
    status["stage"] = "validating_and_merging"
    save_status(status.copy())
    result_root = hosts.get_config().results_path / receipt["batch_id"]
    command = [
        sys.executable, str(ROOT / "scripts/hmm_cluster.py"), "collect",
        "--results", str(result_root), "--output", str(RUN / "comparison"),
    ]
    with (AUDIT / "final_collection.log").open("a") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    status["collection_exit_code"] = result.returncode
    if result.returncode:
        status["stage"] = "collection_needs_attention"
    else:
        report = json.loads((RUN / "comparison/collection_report.json").read_text())
        status["collection"] = {
            key: report[key] for key in [
                "expected", "recorded", "successful", "failed", "missing",
                "ready_for_evaluation",
            ]
        }
        status["stage"] = (
            "training_complete" if report["missing"] == 0
            and report["ready_for_evaluation"] else "recovery_needs_attention"
        )
    save_status(status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
