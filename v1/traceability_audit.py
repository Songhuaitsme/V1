"""Executable v1.0 requirement, invariant, and acceptance-ID audit."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import FrozenSet, Iterable, Tuple


TEST_ID_PATTERN = re.compile(r"\b[A-Z]+-\d{3}\b")
REQUIREMENT_PATTERN = re.compile(r"\bR-\d{2}\b")
INVARIANT_PATTERN = re.compile(r"\bI-\d{2}\b")
NON_TARGET_TEST_PREFIXES = ("SCENARIO-", "TRACE-")


@dataclass(frozen=True)
class TraceabilityReport:
    requirement_ids: FrozenSet[str]
    invariant_ids: FrozenSet[str]
    target_test_ids: FrozenSet[str]
    implemented_test_ids: FrozenSet[str]
    missing_test_ids: Tuple[str, ...]
    duplicate_spec_test_ids: Tuple[str, ...]
    implemented_invariant_ids: FrozenSet[str]
    missing_invariant_ids: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            len(self.requirement_ids) == 50
            and len(self.invariant_ids) == 19
            and len(self.target_test_ids) == 258
            and not self.missing_test_ids
            and not self.duplicate_spec_test_ids
            and not self.missing_invariant_ids
        )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _duplicates(values: Iterable[str]) -> Tuple[str, ...]:
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def audit_repository(root: Path) -> TraceabilityReport:
    start = Path(root).resolve()
    repository_root = next(
        (
            candidate
            for candidate in (start, *start.parents)
            if (candidate / "文档" / "验收测试规格_v1.0.md").exists()
        ),
        None,
    )
    if repository_root is None:
        raise FileNotFoundError(
            f"cannot locate repository 文档 directory from {start}"
        )
    document_root = repository_root / "文档"
    spec_text = _read(document_root / "验收测试规格_v1.0.md")
    raw_test_ids = TEST_ID_PATTERN.findall(spec_text)
    defining_ids = re.findall(r"(?m)^\|\s*([A-Z]+-\d{3})\s*\|", spec_text)
    target_ids = {
        value
        for value in raw_test_ids
        if not value.startswith(NON_TARGET_TEST_PREFIXES)
    }
    implemented = set()
    implemented_invariants = set()
    for path in (repository_root / "tests").rglob("test_*.py"):
        test_text = _read(path)
        implemented.update(TEST_ID_PATTERN.findall(test_text))
        implemented_invariants.update(INVARIANT_PATTERN.findall(test_text))
    requirements = set(REQUIREMENT_PATTERN.findall(
        _read(document_root / "需求追踪矩阵_v1.0.md")
    ))
    invariants = set(INVARIANT_PATTERN.findall(
        _read(document_root / "调度算法设计_v1.0.md")
    ))
    return TraceabilityReport(
        frozenset(requirements),
        frozenset(invariants),
        frozenset(target_ids),
        frozenset(implemented),
        tuple(sorted(target_ids - implemented)),
        _duplicates(defining_ids),
        frozenset(implemented_invariants),
        tuple(sorted(invariants - implemented_invariants)),
    )
