"""Export auditable, source-level metrics from V2 five-policy evaluations.

The raw JSON reports remain the source of truth.  This module validates that
the five reports for each seed are paired, then writes:

* ``metrics_long.csv``: one row per seed, policy, and metric;
* ``metrics_wide.csv``: one row per seed and metric, with five policy values;
* ``source_manifest.csv``: report provenance and pairing hashes; and
* ``analysis_validation.json``: deterministic completeness checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


POLICIES = (
    "earliest_feasible",
    "lowest_cost",
    "highest_green",
    "equal_weight",
    "candidate_dqn",
)

PAIRED_METADATA_FIELDS = (
    "system_version",
    "requirements_version",
    "algorithm_version",
    "task_schema_version",
    "candidate_schema_version",
    "model_schema_version",
    "metric_schema_version",
    "aggregation_schema_version",
    "candidate_mode",
    "seed",
    "arrival_cutoff_sim",
    "code_hash",
    "config_hash",
    "topology_hash",
    "task_trace_hash",
    "exogenous_trace_hash",
    "dependency_lock_hash",
    "tariff_mode",
    "gamma_per_second",
)


class EvaluationDataError(ValueError):
    """Raised when reports are incomplete or cannot be compared fairly."""


def _read_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"cannot read evaluation report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise EvaluationDataError(f"evaluation report must be an object: {path}")
    return report


def _policy_from_name(path: Path, seed: int) -> str:
    suffix = f"_seed{seed}"
    stem = path.stem
    if not stem.endswith(suffix):
        raise EvaluationDataError(
            f"report name must end with {suffix}.json: {path}"
        )
    policy = stem[: -len(suffix)]
    if policy not in POLICIES:
        raise EvaluationDataError(f"unknown policy in report name: {path}")
    return policy


def discover_reports(input_dir: Path) -> list[Path]:
    """Return recognized policy reports beneath ``input_dir``."""

    found = []
    for path in sorted(input_dir.rglob("*.json")):
        if any(path.stem.startswith(f"{policy}_seed") for policy in POLICIES):
            found.append(path)
    if not found:
        raise EvaluationDataError(f"no five-policy evaluation reports found in {input_dir}")
    return found


def _finite_scalar(value, *, path: str):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise EvaluationDataError(f"non-finite metric at {path}")
        return value
    raise EvaluationDataError(f"metric value at {path} is not numeric or null")


def flatten_metrics(metrics: Mapping[str, object]) -> list[dict]:
    """Flatten all numeric metric leaves while preserving MetricValue status."""

    rows: list[dict] = []

    def visit(value, parts: tuple[str, ...]) -> None:
        metric_path = ".".join(parts)
        if isinstance(value, Mapping):
            if "value" in value and "status" in value:
                rows.append({
                    "metric_path": metric_path,
                    "value": _finite_scalar(value.get("value"), path=metric_path),
                    "status": value.get("status"),
                    "reason": value.get("reason"),
                    "numerator": _finite_scalar(
                        value.get("numerator"), path=f"{metric_path}.numerator"
                    ),
                    "denominator": _finite_scalar(
                        value.get("denominator"), path=f"{metric_path}.denominator"
                    ),
                })
                return
            for key, child in value.items():
                visit(child, parts + (str(key),))
            return
        if isinstance(value, (int, float, bool)) or value is None:
            rows.append({
                "metric_path": metric_path,
                "value": _finite_scalar(value, path=metric_path),
                "status": "VALID" if value is not None else "NOT_APPLICABLE",
                "reason": None if value is not None else "null scalar metric",
                "numerator": None,
                "denominator": None,
            })
            return
        # Descriptive leaves such as sla_type are not evaluation measures.
        if not isinstance(value, str):
            raise EvaluationDataError(
                f"unsupported metric value type at {metric_path}: {type(value).__name__}"
            )

    visit(metrics, ())
    return rows


def _validate_and_group(report_paths: Iterable[Path]):
    grouped: dict[int, dict[str, tuple[Path, dict]]] = {}
    for path in report_paths:
        report = _read_report(path)
        metadata = report.get("metadata")
        if not isinstance(metadata, dict) or "seed" not in metadata:
            raise EvaluationDataError(f"missing metadata.seed: {path}")
        seed = int(metadata["seed"])
        policy = _policy_from_name(path, seed)
        if policy in grouped.setdefault(seed, {}):
            raise EvaluationDataError(f"duplicate report for seed {seed}, policy {policy}")
        grouped[seed][policy] = (path, report)

    for seed, reports in grouped.items():
        missing = [policy for policy in POLICIES if policy not in reports]
        if missing:
            raise EvaluationDataError(
                f"seed {seed} is missing policies: {', '.join(missing)}"
            )
        reference = reports[POLICIES[0]][1]
        reference_metadata = reference["metadata"]
        for policy in POLICIES:
            path, report = reports[policy]
            if report.get("status") != "VALID":
                raise EvaluationDataError(f"report is not VALID: {path}")
            if report.get("unsettled_task_ids"):
                raise EvaluationDataError(f"report has unsettled tasks: {path}")
            if not isinstance(report.get("metrics"), dict):
                raise EvaluationDataError(f"report has no metrics object: {path}")
            metadata = report.get("metadata", {})
            mismatches = [
                field
                for field in PAIRED_METADATA_FIELDS
                if metadata.get(field) != reference_metadata.get(field)
            ]
            if mismatches:
                raise EvaluationDataError(
                    f"seed {seed}, policy {policy} has mismatched paired metadata: "
                    + ", ".join(mismatches)
                )
    return grouped


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_reports(report_paths: Iterable[Path], output_dir: Path) -> dict:
    """Validate reports and create long, wide, and provenance source tables."""

    grouped = _validate_and_group(Path(path) for path in report_paths)
    long_rows = []
    manifest_rows = []

    for seed in sorted(grouped):
        for policy in POLICIES:
            source_path, report = grouped[seed][policy]
            metadata = report["metadata"]
            metrics = report["metrics"]
            source_text = source_path.as_posix()
            for metric in flatten_metrics(metrics):
                long_rows.append({
                    "seed": seed,
                    "policy": policy,
                    **metric,
                    "source_file": source_text,
                })
            manifest_rows.append({
                "seed": seed,
                "policy": policy,
                "source_file": source_text,
                "status": report["status"],
                "system_version": metadata.get("system_version"),
                "arrival_cutoff_sim": metadata.get("arrival_cutoff_sim"),
                "model_hash": metadata.get("model_hash"),
                "code_hash": metadata.get("code_hash"),
                "config_hash": metadata.get("config_hash"),
                "topology_hash": metadata.get("topology_hash"),
                "task_trace_hash": metadata.get("task_trace_hash"),
                "exogenous_trace_hash": metadata.get("exogenous_trace_hash"),
                "dependency_lock_hash": metadata.get("dependency_lock_hash"),
                "arrival_count": metrics.get("arrival_count"),
                "completed_count": metrics.get("completed_count"),
                "unsettled_count": len(report.get("unsettled_task_ids", ())),
                "paired_metadata_match": True,
            })

    indexed = {
        (row["seed"], row["metric_path"], row["policy"]): row
        for row in long_rows
    }
    metric_keys = sorted({(row["seed"], row["metric_path"]) for row in long_rows})
    wide_rows = []
    for seed, metric_path in metric_keys:
        row = {"seed": seed, "metric_path": metric_path}
        for policy in POLICIES:
            metric = indexed.get((seed, metric_path, policy))
            row[policy] = None if metric is None else metric["value"]
            row[f"{policy}_status"] = "MISSING" if metric is None else metric["status"]
        wide_rows.append(row)

    long_fields = (
        "seed", "policy", "metric_path", "value", "status", "reason",
        "numerator", "denominator", "source_file",
    )
    wide_fields = (
        "seed", "metric_path", *POLICIES,
        *(f"{policy}_status" for policy in POLICIES),
    )
    manifest_fields = (
        "seed", "policy", "source_file", "status", "system_version",
        "arrival_cutoff_sim", "model_hash", "code_hash", "config_hash",
        "topology_hash", "task_trace_hash", "exogenous_trace_hash",
        "dependency_lock_hash", "arrival_count", "completed_count",
        "unsettled_count", "paired_metadata_match",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics_long.csv", long_fields, long_rows)
    _write_csv(output_dir / "metrics_wide.csv", wide_fields, wide_rows)
    _write_csv(output_dir / "source_manifest.csv", manifest_fields, manifest_rows)

    validation = {
        "status": "PASS",
        "seeds": sorted(grouped),
        "seed_count": len(grouped),
        "policies": list(POLICIES),
        "reports_per_seed": len(POLICIES),
        "source_report_count": len(manifest_rows),
        "long_metric_row_count": len(long_rows),
        "wide_metric_row_count": len(wide_rows),
        "all_reports_valid": True,
        "all_unsettled_counts_zero": True,
        "paired_metadata_fields": list(PAIRED_METADATA_FIELDS),
        "paired_metadata_match": True,
    }
    (output_dir / "analysis_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all metrics from paired V2 five-policy reports"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/v2/evaluation/five_policy"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir / "source_data"
    validation = export_reports(discover_reports(args.input_dir), output_dir)
    print(json.dumps({
        "status": validation["status"],
        "output_dir": str(output_dir),
        "source_report_count": validation["source_report_count"],
        "long_metric_row_count": validation["long_metric_row_count"],
        "wide_metric_row_count": validation["wide_metric_row_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
