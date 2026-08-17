"""V2 evaluation entry point with periodic audits and compact reports."""

from v1.evaluate_v1 import main as _evaluation_main


def main():
    _evaluation_main(
        system_version="2.0",
        default_output="artifacts/v2/evaluation/report.json",
        default_audit_mode="periodic",
        default_report_mode="compact",
        default_candidate_chunk_size=65536,
    )


if __name__ == "__main__":
    main()
