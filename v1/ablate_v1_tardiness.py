"""Frozen-trace ablation for Soft/Flexible preferred-start penalties."""

import argparse
import csv
import json
from pathlib import Path
import statistics

from v1.domain.models import SlaType
from v1.evaluate_v1 import _jsonable, run_evaluation


def _value(metric):
    return None if metric is None or metric.value is None else float(metric.value)


def _parse_weights(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("weights must be SOFT,FLEXIBLE")
    try:
        soft, flexible = (float(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    if soft < 0.0 or flexible < 0.0:
        raise argparse.ArgumentTypeError("weights must be non-negative")
    return soft, flexible


def _row(report, soft_weight, flexible_weight):
    metrics = report.metrics
    soft = metrics.sla_metrics[SlaType.SOFT]
    flexible = metrics.sla_metrics[SlaType.FLEXIBLE]
    return {
        "seed": report.metadata.seed,
        "soft_weight": soft_weight,
        "flexible_weight": flexible_weight,
        "arrival_count": metrics.arrival_count,
        "completion_rate": _value(metrics.completion_rate),
        "expired_count": metrics.expired_count,
        "failed_count": metrics.failed_count,
        "total_economic_cost_yuan": metrics.total_economic_cost_yuan,
        "cost_yuan_per_completed_cpu_hour": _value(
            metrics.cost_yuan_per_completed_cpu_hour
        ),
        "completed_task_green_coverage": _value(
            metrics.completed_task_green_coverage
        ),
        "system_green_absorption_rate": _value(
            metrics.system_green_absorption_rate
        ),
        "active_wait_count": metrics.active_wait_metrics.count,
        "soft_count": soft.count,
        "soft_preferred_on_time_rate": _value(soft.preferred_on_time_rate),
        "soft_acceptable_tardy_rate": _value(soft.acceptable_tardy_rate),
        "soft_tardiness_p95": _value(soft.preferred_start_tardiness_p95),
        "flexible_count": flexible.count,
        "flexible_preferred_on_time_rate": _value(
            flexible.preferred_on_time_rate
        ),
        "flexible_acceptable_tardy_rate": _value(
            flexible.acceptable_tardy_rate
        ),
        "flexible_tardiness_p95": _value(
            flexible.preferred_start_tardiness_p95
        ),
    }


def _mean(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return statistics.fmean(values) if values else None


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(
            (row["soft_weight"], row["flexible_weight"]), []
        ).append(row)
    fields = (
        "completion_rate",
        "expired_count",
        "failed_count",
        "total_economic_cost_yuan",
        "cost_yuan_per_completed_cpu_hour",
        "completed_task_green_coverage",
        "system_green_absorption_rate",
        "active_wait_count",
        "soft_preferred_on_time_rate",
        "soft_acceptable_tardy_rate",
        "soft_tardiness_p95",
        "flexible_preferred_on_time_rate",
        "flexible_acceptable_tardy_rate",
        "flexible_tardiness_p95",
    )
    summaries = []
    for weights, items in sorted(grouped.items()):
        summary = {
            "soft_weight": weights[0],
            "flexible_weight": weights[1],
            "seed_count": len(items),
        }
        summary.update({field: _mean(items, field) for field in fields})
        on_time = [
            value
            for value in (
                summary["soft_preferred_on_time_rate"],
                summary["flexible_preferred_on_time_rate"],
            )
            if value is not None
        ]
        summary["mean_soft_flexible_preferred_on_time_rate"] = (
            statistics.fmean(on_time) if on_time else None
        )
        summaries.append(summary)

    baseline = next(
        (
            item for item in summaries
            if item["soft_weight"] == 0.0
            and item["flexible_weight"] == 0.0
        ),
        None,
    )
    valid = [
        item for item in summaries
        if baseline is None
        or (
            item["completion_rate"] >= baseline["completion_rate"] - 1e-12
            and item["expired_count"] <= baseline["expired_count"] + 1e-12
            and item["failed_count"] <= baseline["failed_count"] + 1e-12
        )
    ]
    recommended = max(
        valid,
        key=lambda item: (
            -1.0
            if item["mean_soft_flexible_preferred_on_time_rate"] is None
            else item["mean_soft_flexible_preferred_on_time_rate"],
            -item["total_economic_cost_yuan"],
            -(item["soft_weight"] + item["flexible_weight"]),
        ),
    )
    return summaries, {
        "soft_weight": recommended["soft_weight"],
        "flexible_weight": recommended["flexible_weight"],
        "selection_rule": (
            "Preserve baseline completion/expired/failed gates; maximize the "
            "mean Soft/Flexible preferred-on-time rate; then minimize cost and "
            "total penalty weight."
        ),
    }


def run_ablation(seeds, weights, cutoff, safety_cap, output_dir):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for soft_weight, flexible_weight in weights:
        weight_id = f"soft_{soft_weight:g}_flex_{flexible_weight:g}"
        for seed in seeds:
            report = run_evaluation(
                "equal_weight",
                cutoff,
                seed,
                safety_cap,
                soft_tardiness_weight=soft_weight,
                flexible_tardiness_weight=flexible_weight,
            )
            if report.status.value != "VALID":
                raise RuntimeError(
                    f"invalid ablation report for weights {weight_id}, seed {seed}"
                )
            report_path = target / f"{weight_id}_seed_{seed}.json"
            report_path.write_text(
                json.dumps(
                    _jsonable(report), ensure_ascii=False, indent=2, allow_nan=False
                ),
                encoding="utf-8",
            )
            row = _row(report, soft_weight, flexible_weight)
            rows.append(row)
            print(json.dumps({"ablation_progress": row}, ensure_ascii=False), flush=True)

    summaries, recommendation = summarize(rows)
    result = {
        "status": "VALID_TARDINESS_ABLATION",
        "seeds": list(seeds),
        "arrival_cutoff_sim": cutoff,
        "weights": [
            {"soft": soft, "flexible": flexible}
            for soft, flexible in weights
        ],
        "per_seed": rows,
        "summary": summaries,
        "recommendation": recommendation,
    }
    json_path = target / "tardiness_ablation.json"
    csv_path = target / "tardiness_ablation.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description="Ablate v1 tardiness weights")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument(
        "--weights",
        type=_parse_weights,
        action="append",
        default=None,
        help="repeat SOFT,FLEXIBLE pairs",
    )
    parser.add_argument("--arrival-cutoff", type=float, default=1.0)
    parser.add_argument("--safety-cap", type=int, default=1000000)
    parser.add_argument(
        "--output-dir", default="artifacts/v1/ablation/tardiness"
    )
    args = parser.parse_args()
    weights = args.weights or [(0.0, 0.0), (0.5, 0.25), (1.0, 0.5)]
    result = run_ablation(
        tuple(args.seed), tuple(weights), args.arrival_cutoff,
        args.safety_cap, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "recommendation": result["recommendation"],
        "output_dir": args.output_dir,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
