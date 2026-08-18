"""Run the five frozen V2 policies on identical per-seed workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from shared import config
from v2.export_five_policy_metrics import POLICIES, discover_reports, export_reports


DEFAULT_MODEL_PATH = Path(
    "artifacts/v2/formal/"
    "candidate_dqn_seed7_layered_pool_600000_from_scratch.pt"
)


def evaluation_command(
    *,
    policy: str,
    seed: int,
    output: Path,
    model_path: Path,
    arrival_cutoff: float,
    safety_cap: int,
    device: str,
    candidate_chunk_size: int,
    audit: str,
    audit_interval: int,
    report_mode: str,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "v2.evaluate_v2",
        "--policy",
        policy,
        "--arrival-cutoff",
        str(arrival_cutoff),
        "--seed",
        str(seed),
        "--safety-cap",
        str(safety_cap),
        "--device",
        device,
        "--candidate-chunk-size",
        str(candidate_chunk_size),
        "--audit",
        audit,
        "--audit-interval",
        str(audit_interval),
        "--report-mode",
        report_mode,
        "--output",
        str(output),
    ]
    if policy == "candidate_dqn":
        command.extend(("--model-path", str(model_path)))
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_reusable_report(
    path: Path,
    *,
    policy: str,
    seed: int,
    arrival_cutoff: float,
    model_hash: str,
) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = report.get("metadata", {})
    reusable = (
        report.get("status") == "VALID"
        and not report.get("unsettled_task_ids")
        and metadata.get("system_version") == "2.0"
        and metadata.get("seed") == seed
        and metadata.get("arrival_cutoff_sim") == arrival_cutoff
    )
    if policy == "candidate_dqn":
        reusable = reusable and metadata.get("model_hash") == model_hash
    return reusable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all five frozen policies with paired V2 workloads"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(42,))
    parser.add_argument(
        "--arrival-cutoff",
        type=float,
        default=config.TRAFFIC_DAY_DURATION_IN_SIM,
    )
    parser.add_argument("--safety-cap", type=int, default=1_000_000)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate-chunk-size", type=int, default=65_536)
    parser.add_argument("--audit", choices=("full", "periodic", "final"), default="periodic")
    parser.add_argument("--audit-interval", type=int, default=500)
    parser.add_argument("--report-mode", choices=("compact", "full"), default="compact")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2/evaluation/five_policy"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing VALID reports with matching seed and cutoff",
    )
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = tuple(dict.fromkeys(args.seeds))
    if not seeds:
        parser.error("at least one seed is required")
    if args.arrival_cutoff <= 0.0:
        parser.error("--arrival-cutoff must be positive")
    if args.safety_cap <= 0:
        parser.error("--safety-cap must be positive")
    if args.candidate_chunk_size <= 0:
        parser.error("--candidate-chunk-size must be positive")
    if args.audit_interval <= 0:
        parser.error("--audit-interval must be positive")
    if not args.model_path.is_file():
        parser.error(f"model does not exist: {args.model_path}")
    model_hash = _sha256(args.model_path)

    commands = []
    for seed in seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        for policy in POLICIES:
            output = seed_dir / f"{policy}_seed{seed}.json"
            commands.append((seed, policy, output, evaluation_command(
                policy=policy,
                seed=seed,
                output=output,
                model_path=args.model_path,
                arrival_cutoff=args.arrival_cutoff,
                safety_cap=args.safety_cap,
                device=args.device,
                candidate_chunk_size=args.candidate_chunk_size,
                audit=args.audit,
                audit_interval=args.audit_interval,
                report_mode=args.report_mode,
            )))

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN",
            "evaluation_count": len(commands),
            "commands": [command for _, _, _, command in commands],
        }, ensure_ascii=False, indent=2))
        return

    for seed, policy, output, command in commands:
        if output.exists():
            if args.resume and _is_reusable_report(
                output,
                policy=policy,
                seed=seed,
                arrival_cutoff=args.arrival_cutoff,
                model_hash=model_hash,
            ):
                print(json.dumps({
                    "status": "REUSED",
                    "seed": seed,
                    "policy": policy,
                    "output": str(output),
                }, ensure_ascii=False))
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"output already exists: {output}; use --resume or --overwrite"
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({
            "status": "RUNNING",
            "seed": seed,
            "policy": policy,
            "output": str(output),
        }, ensure_ascii=False), flush=True)
        subprocess.run(command, check=True)

    result = {
        "status": "VALID",
        "seeds": list(seeds),
        "evaluation_count": len(commands),
        "output_dir": str(args.output_dir),
    }
    if not args.no_export:
        source_dir = args.output_dir / "source_data"
        validation = export_reports(discover_reports(args.output_dir), source_dir)
        result["source_data_dir"] = str(source_dir)
        result["long_metric_row_count"] = validation["long_metric_row_count"]
        result["wide_metric_row_count"] = validation["wide_metric_row_count"]
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
