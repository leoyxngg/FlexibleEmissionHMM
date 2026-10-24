"""Date-validated signal merging and source checkpoint collection."""

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


def merge_results(sources, output, legacy_gmm_components=None):
    """Merge matching signal histories and collect unchanged checkpoints.

    Args:
        sources: Result folders containing signals.csv and checkpoint.pkl.
        output: New destination folder; existing folders are never overwritten.
        legacy_gmm_components: Explicit GMM count for older signal schemas.

    Returns:
        The merged signal table.

    Raises:
        ValueError: Sources contain duplicate branches or mismatched dates.
        FileExistsError: The destination already exists.
    """
    sources = [Path(source).resolve() for source in sources]
    output = Path(output).resolve()
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError("Provide at least two distinct source folders")
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("Source folder names must be unique")

    frames, manifest = [], []
    reference_dates = None
    seen_specs = set()
    for source in sources:
        checkpoint = source / "checkpoint.pkl"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        frame = pd.read_csv(source / "signals.csv", parse_dates=["date"])
        required = {"date", "model", "pca", "bull_probability", "bull_raw"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f"Missing required columns or empty signals: {source}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"Missing required signal values: {source}")
        if not frame["pca"].isin([True, False]).all():
            raise ValueError(f"Invalid PCA flags: {source}")

        # Older paper runs omit spec and use two components for GMM emissions.
        if "gmm_components" not in frame:
            if frame["model"].eq("GMM").any() and (
                legacy_gmm_components is None or legacy_gmm_components < 1
            ):
                raise ValueError(
                    f"Unknown GMM component count in {source}; supply "
                    "--legacy-gmm-components (2 for the original paper run)"
                )
            frame["gmm_components"] = [
                legacy_gmm_components if model == "GMM" else 1
                for model in frame["model"]
            ]
        if frame["gmm_components"].isna().any():
            raise ValueError(f"Missing component counts: {source}")
        if "spec" not in frame:
            names = {"Gaussian": "gaussian", "Student-t": "student_t"}
            frame["spec"] = [
                (f"gmm{int(count)}" if model == "GMM" else names[model])
                + ("_pca" if pca else "_no_pca")
                for model, count, pca in zip(
                    frame["model"], frame["gmm_components"], frame["pca"]
                )
            ]
        if frame["spec"].isna().any():
            raise ValueError(f"Missing branch identifiers: {source}")
        if frame.duplicated(["date", "spec"]).any():
            raise ValueError(f"Duplicate date/spec rows: {source}")
        specs = set(frame["spec"])
        if seen_specs & specs:
            raise ValueError(f"Repeated branches: {sorted(seen_specs & specs)}")
        seen_specs.update(specs)
        for spec, group in frame.groupby("spec"):
            dates = pd.DatetimeIndex(group["date"].sort_values())
            if reference_dates is None:
                reference_dates = dates
            elif not dates.equals(reference_dates):
                raise ValueError(f"Decision dates do not match: {source.name}/{spec}")

        # Distinguish mixture sizes for existing model/PCA plotting helpers.
        is_gmm = frame["model"].eq("GMM")
        frame.loc[is_gmm, "model"] = frame.loc[is_gmm, "gmm_components"].map(
            lambda count: f"GMM (M={int(count)})"
        )
        frame["source_run"] = source.name
        frames.append(frame)
        manifest.append({"source": str(source), "rows": len(frame)})

    merged = pd.concat(frames, ignore_index=True).sort_values(["date", "spec"])
    if merged.duplicated(["date", "model", "pca"]).any():
        raise ValueError("Model labels collide; use distinct model names")

    # Validate everything before creating the destination; never load pickle data.
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    for source in sources:
        shutil.copy2(source / "checkpoint.pkl", checkpoints / f"{source.name}.pkl")
    merged.to_csv(output / "signals.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps({"sources": manifest, "decision_dates": len(reference_dates),
                    "note": "Analysis bundle, not a resumable training checkpoint."},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    return merged


def main():
    """Merge result folders supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--legacy-gmm-components", type=int,
        help="Component count for older signals without gmm_components (e.g. 2)",
    )
    args = parser.parse_args()
    merged = merge_results(
        args.sources, args.output, args.legacy_gmm_components
    )
    print(f"Merged {len(merged):,} rows / {merged['spec'].nunique()} branches")
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
