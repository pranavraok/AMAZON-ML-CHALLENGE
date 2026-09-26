"""Phase 6 final validation for the candidate deliverable.

Every check is independent and reports PASS or FAIL with the observed
value, so a failure names the exact condition rather than a generic error.

No check reads ground truth or any label file. Label leakage is checked
separately by :func:`check_no_label_leakage`, which inspects the source of
the generation modules.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

EXPECTED_HEADER = "source1_entity_id\tcandidate_entity_ids"

# Any module that participates in generating final test candidates.
GENERATION_MODULES = (
    "candidate_generation.py",
    "compact_index.py",
    "chunked_generation.py",
    "retrieval.py",
    "blocking.py",
    "normalization.py",
    "run_chunked_generation.py",
    "generate_submission_candidates.py",
)

FORBIDDEN_TOKENS = (
    "ground_truth",
    "train_ground_truth",
    "matched_entity_ids",
    "label",
    "dev_pairs",
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


def _read_ids(path: Path, column: str = "entity_id") -> set[str]:
    import pandas as pd

    ids: set[str] = set()

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[column],
        chunksize=200_000,
    )

    for block in reader:
        ids.update(block[column].tolist())

    return ids


# ==============================================================
# A. SCHEMA
# ==============================================================


def check_schema(path: Path) -> CheckResult:
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").rstrip("\r")

    if header == EXPECTED_HEADER:
        return CheckResult(
            "E. schema",
            True,
            f"header is exactly {EXPECTED_HEADER!r}",
        )

    return CheckResult(
        "E. schema",
        False,
        f"header is {header!r}, expected {EXPECTED_HEADER!r}",
    )


# ==============================================================
# B. ROW COUNT AND S1 UNIQUENESS
# ==============================================================


def check_rows_and_uniqueness(
    path: Path,
    s1_path: Path,
    expected_rows: int | None = None,
) -> list[CheckResult]:
    import pandas as pd

    expected_ids = _read_ids(s1_path)

    seen: dict[str, int] = {}
    total = 0
    bad_columns = 0

    with open(path, "r", encoding="utf-8") as handle:
        handle.readline()

        for number, line in enumerate(handle, start=2):
            line = line.rstrip("\n").rstrip("\r")

            if line.count("\t") != 1:
                bad_columns += 1
                if bad_columns <= 3:
                    continue

            entity_id, _, _payload = line.partition("\t")
            total += 1
            seen[entity_id] = seen.get(entity_id, 0) + 1

    results = []

    duplicates = [
        entity_id
        for entity_id, count in seen.items()
        if count > 1
    ]

    results.append(
        CheckResult(
            "B. row count",
            total == len(expected_ids)
            and (expected_rows is None or total == expected_rows),
            f"{total:,} data rows, expected {len(expected_ids):,}"
            + (
                f" (declared {expected_rows:,})"
                if expected_rows is not None
                else ""
            ),
        )
    )

    results.append(
        CheckResult(
            "B. S1 row uniqueness",
            not duplicates,
            "no duplicate Source 1 rows"
            if not duplicates
            else f"{len(duplicates):,} duplicated, first {duplicates[:3]}",
        )
    )

    missing = expected_ids - set(seen)
    extra = set(seen) - expected_ids

    results.append(
        CheckResult(
            "B. S1 coverage",
            not missing and not extra,
            "every Source 1 record present exactly once"
            if not missing and not extra
            else (
                f"{len(missing):,} missing, {len(extra):,} unexpected"
            ),
        )
    )

    results.append(
        CheckResult(
            "E. field count",
            bad_columns == 0,
            "every row has exactly two tab-separated fields"
            if bad_columns == 0
            else f"{bad_columns:,} malformed rows",
        )
    )

    del pd

    return results


# ==============================================================
# C. CANDIDATE ID VALIDITY
# ==============================================================


def check_candidate_ids(
    path: Path,
    s2_path: Path,
    s3_path: Path,
) -> CheckResult:
    allowed = _read_ids(s2_path) | _read_ids(s3_path)

    total = 0
    invalid: set[str] = set()

    with open(path, "r", encoding="utf-8") as handle:
        handle.readline()

        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            _, _, payload = line.partition("\t")

            if not payload:
                continue

            for candidate_id in payload.split(","):
                if not candidate_id:
                    continue
                total += 1
                if candidate_id not in allowed:
                    invalid.add(candidate_id)

    if not invalid:
        return CheckResult(
            "C. candidate id validity",
            True,
            f"all {total:,} candidate ids exist in S2 or S3",
        )

    return CheckResult(
        "C. candidate id validity",
        False,
        f"{len(invalid):,} invalid ids, first {sorted(invalid)[:3]}",
    )


# ==============================================================
# D. DUPLICATES WITHIN A ROW
# ==============================================================


def check_within_row_duplicates(path: Path) -> CheckResult:
    rows_with_duplicates = 0
    total = 0

    with open(path, "r", encoding="utf-8") as handle:
        handle.readline()

        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            _, _, payload = line.partition("\t")

            if not payload:
                continue

            total += 1
            values = payload.split(",")

            if len(values) != len(set(values)):
                rows_with_duplicates += 1

    if not rows_with_duplicates:
        return CheckResult(
            "D. within-row duplicates",
            True,
            f"no duplicate candidate ids in any of {total:,} rows",
        )

    return CheckResult(
        "D. within-row duplicates",
        False,
        f"{rows_with_duplicates:,} rows contain duplicate ids",
    )


# ==============================================================
# F. NO LABEL LEAKAGE
# ==============================================================


def check_no_label_leakage(package_dir: Path) -> CheckResult:
    """Static check: no generation module references any label source.

    This inspects identifiers and string literals in the source, so it
    catches a leaked read even if the code path is not exercised here.
    """

    offences: list[str] = []

    for name in GENERATION_MODULES:

        path = package_dir / name

        if not path.exists():
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):

            if isinstance(node, ast.Name):
                text = node.id
            elif isinstance(node, ast.Attribute):
                text = node.attr
            elif isinstance(node, ast.Constant) and isinstance(
                node.value, str
            ):
                text = node.value
            else:
                continue

            lowered = text.lower()

            for token in FORBIDDEN_TOKENS:
                if token in lowered:
                    offences.append(
                        f"{name}:{node.lineno} {token!r} in {text!r}"
                    )

    if not offences:
        return CheckResult(
            "F. no label leakage",
            True,
            f"no reference to {list(FORBIDDEN_TOKENS)} in "
            f"{len(GENERATION_MODULES)} generation modules",
        )

    return CheckResult(
        "F. no label leakage",
        False,
        f"{len(offences)} references, first {offences[:3]}",
    )


# ==============================================================
# DRIVER
# ==============================================================


def validate(
    candidates_path: Path,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    package_dir: Path,
    expected_rows: int | None = None,
) -> list[CheckResult]:
    results = [check_schema(candidates_path)]

    results.extend(
        check_rows_and_uniqueness(
            candidates_path, s1_path, expected_rows
        )
    )
    results.append(
        check_candidate_ids(candidates_path, s2_path, s3_path)
    )
    results.append(check_within_row_duplicates(candidates_path))
    results.append(check_no_label_leakage(package_dir))

    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate the final candidate deliverable."
    )
    parser.add_argument(
        "--candidates",
        default="output/candidate_pairs.tsv",
    )
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument(
        "--package-dir",
        default="src/business_entity_resolution",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    results = validate(
        Path(args.candidates),
        Path(args.s1),
        Path(args.s2),
        Path(args.s3),
        Path(args.package_dir),
        expected_rows=args.expected_rows,
    )

    print("=" * 60)
    print("PERSON 2 FINAL VALIDATION")
    print("=" * 60)

    for result in results:
        print(result.render())

    failed = [result for result in results if not result.passed]

    print("=" * 60)

    if failed:
        print(f"VALIDATION FAILED: {len(failed)} check(s)")
        return 1

    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
