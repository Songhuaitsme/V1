import csv
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import unittest
import uuid

from v2.export_five_policy_metrics import (
    EvaluationDataError,
    PAIRED_METADATA_FIELDS,
    POLICIES,
    export_reports,
)


class V2EvaluationToolsTest(unittest.TestCase):
    @contextmanager
    def _temporary_directory(self):
        path = Path(__file__).resolve().parent / f"_tmp_{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path)

    def _report(self, seed, policy, *, task_hash="tasks"):
        metadata = {field: f"shared-{field}" for field in PAIRED_METADATA_FIELDS}
        metadata.update({
            "system_version": "2.0",
            "seed": seed,
            "arrival_cutoff_sim": 1.0,
            "task_trace_hash": task_hash,
            "model_hash": f"model-{policy}",
        })
        return {
            "status": "VALID",
            "metadata": metadata,
            "metrics": {
                "arrival_count": 10,
                "completed_count": 9,
                "completion_rate": {
                    "value": 0.9,
                    "status": "VALID",
                    "reason": None,
                    "numerator": 9,
                    "denominator": 10,
                },
                "sla_metrics": {
                    "Hard": {
                        "sla_type": "Hard",
                        "preferred_on_time_rate": {
                            "value": None,
                            "status": "NOT_APPLICABLE",
                            "reason": "Hard has no preferred start limit",
                            "numerator": None,
                            "denominator": None,
                        },
                    },
                },
            },
            "unsettled_task_ids": [],
        }

    def _write_reports(self, root: Path, *, bad_policy=None):
        paths = []
        seed_dir = root / "seed_42"
        seed_dir.mkdir()
        for policy in POLICIES:
            task_hash = "different" if policy == bad_policy else "tasks"
            path = seed_dir / f"{policy}_seed42.json"
            path.write_text(
                json.dumps(self._report(42, policy, task_hash=task_hash)),
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    def test_export_writes_all_metric_and_provenance_tables(self):
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            output = root / "source_data"
            validation = export_reports(self._write_reports(root), output)

            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["source_report_count"], 5)
            self.assertEqual(validation["long_metric_row_count"], 20)
            self.assertEqual(validation["wide_metric_row_count"], 4)

            with (output / "metrics_long.csv").open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                long_rows = list(csv.DictReader(handle))
            self.assertTrue(any(
                row["policy"] == "candidate_dqn"
                and row["metric_path"] == "completion_rate"
                and row["numerator"] == "9"
                for row in long_rows
            ))
            self.assertFalse(any(
                row["metric_path"].endswith("sla_type") for row in long_rows
            ))

            with (output / "metrics_wide.csv").open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                wide_rows = list(csv.DictReader(handle))
            completion = next(
                row for row in wide_rows if row["metric_path"] == "completion_rate"
            )
            self.assertEqual(completion["candidate_dqn"], "0.9")
            self.assertEqual(completion["candidate_dqn_status"], "VALID")

    def test_export_rejects_unpaired_trace_hash(self):
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(EvaluationDataError, "task_trace_hash"):
                export_reports(
                    self._write_reports(root, bad_policy="candidate_dqn"),
                    root / "source_data",
                )


if __name__ == "__main__":
    unittest.main()
