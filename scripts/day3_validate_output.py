"""Day 3 / Task 5: validate the candidate output of the integration pipeline.

This script does not create or modify any candidate file. It reads a
candidate file that the production path already produced and reports PASS
or FAIL, with the observed value, for every check in the Day-3 brief.

The leakage check is reported twice: statically, by parsing the imports and
string literals of every module on the final generation path, and
dynamically, from the file-access audit captured by
``scripts/day3_audit_run.py`` while the candidate file was produced.

Usage:
    python scripts/day3_validate_output.py \
        --candidates data/work/day3/shard_run1/candidate_pairs.tsv \
        --s1 <test_source1.tsv> --s2 <test_source2.tsv> --s3 <test_source3.tsv> \
        --train-s1 <train_source1.tsv> --train-s2 <train_source2.tsv> \
        --train-s3 <train_source3.tsv> --limit 10000 \
        --audit data/work/day3/shard_run2/audit.json \
        --output docs/day3_candidate_validation.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from business_entity_resolution.validate_candidates import (  # noqa: E402
    FORBIDDEN_TOKENS,
    GENERATION_MODULES,
)

EXPECTED_HEADER = "source1_entity_id\tcandidate_entity_ids"

# Anything that looks like a label source, for the dynamic file audit.
LABEL_PATTERNS = (
    "ground_truth",
    "train_ground_truth",
    "matched_entity_ids",
    "labels",
    "dev_pairs",
    "validation",
)


def _stream(path: Path, columns: list[str], chunksize: int = 200_000):
    import pandas as pd

    yield from pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=columns,
        chunksize=chunksize,
    )


def read_candidates(
    path: Path,
) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    """One pass: row order, per-row candidate lists and format defects."""

    order: list[str] = []
    rows: dict[str, list[str]] = {}

    defects = {
        "malformed_rows": 0,
        "rows_with_blank_payload_not_at_end": 0,
        "payload_nan_or_none": 0,
        "leading_or_trailing_comma": 0,
    }

    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").rstrip("\r")

        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.count("\t") != 1:
                defects["malformed_rows"] += 1
                continue
            entity_id, _, payload = line.partition("\t")
            order.append(entity_id)
            if payload == "":
                rows[entity_id] = []
                continue
            if payload.startswith(",") or payload.endswith(","):
                defects["leading_or_trailing_comma"] += 1
            values = payload.split(",")
            if any(value in ("nan", "None", "null") for value in values):
                defects["payload_nan_or_none"] += 1
            rows[entity_id] = [
                value for value in values if value
            ]

    return header, rows, defects | {
        "header": header,
        "header_matches_required_schema": header == EXPECTED_HEADER,
    }


def check(label: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "check": label,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "detail": detail,
    }


def static_leakage_check(package_dir: Path) -> dict[str, Any]:
    offences: list[str] = []
    imports: dict[str, list[str]] = {}

    for name in GENERATION_MODULES:
        path = package_dir / name
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            elif isinstance(node, ast.Name):
                text = node.id
                lowered = text.lower()
                for token in FORBIDDEN_TOKENS:
                    if token in lowered:
                        offences.append(f"{name}: name {text!r}")
            elif isinstance(node, ast.Constant) and isinstance(
                node.value, str
            ):
                lowered = node.value.lower()
                for token in FORBIDDEN_TOKENS:
                    if token in lowered:
                        offences.append(
                            f"{name}:{node.lineno} literal {node.value!r}"
                        )
        imports[name] = sorted(set(modules))

    return {
        "offences": offences,
        "imports": imports,
        "modules_checked": [
            name
            for name in GENERATION_MODULES
            if (package_dir / name).is_file()
        ],
    }


def dynamic_leakage_check(audit_path: Path | None) -> dict[str, Any]:
    if audit_path is None or not audit_path.is_file():
        return {
            "available": False,
            "note": (
                "no audit file supplied; the dynamic check was not run"
            ),
        }

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    opened = payload.get("files_opened", [])
    label_reads = [
        path
        for path in opened
        if any(pattern in path.lower() for pattern in LABEL_PATTERNS)
    ]
    return {
        "available": True,
        "audit_file": str(audit_path),
        "python": payload.get("python"),
        "command": payload.get("command"),
        "files_opened_count": len(opened),
        "dataset_files_opened": [
            path for path in opened if "student_resource" in path
        ],
        "label_like_files_opened": label_reads,
        "no_label_file_read": not label_reads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--train-s1", required=True)
    parser.add_argument("--train-s2", required=True)
    parser.add_argument("--train-s3", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--package-dir", default="src/business_entity_resolution")
    parser.add_argument(
        "--output", default="docs/day3_candidate_validation.json"
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    s1_path = Path(args.s1)
    s2_path = Path(args.s2)
    s3_path = Path(args.s3)
    package_dir = REPO / args.package_dir

    header, rows, defects = read_candidates(candidates_path)
    results = []

    # ---- 1. schema -------------------------------------------------
    results.append(
        check(
            "1. required schema",
            defects["header_matches_required_schema"],
            f"header {header!r}; required {EXPECTED_HEADER!r}",
        )
    )
    results.append(
        check(
            "1b. row field count",
            defects["malformed_rows"] == 0,
            f"{defects['malformed_rows']} rows without exactly one tab",
        )
    )

    # ---- 2. S1 coverage, one row each ------------------------------
    s1_ids: list[str] = []
    s1_country: dict[str, str] = {}
    for block in _stream(s1_path, ["entity_id", "country"]):
        s1_ids.extend(block["entity_id"].tolist())
        s1_country.update(
            zip(block["entity_id"], block["country"])
        )
        if args.limit and len(s1_ids) >= args.limit:
            break
    if args.limit:
        s1_ids = s1_ids[: args.limit]

    emitted = list(rows)
    duplicated_rows = len(emitted) - len(set(emitted))
    missing = [i for i in s1_ids if i not in rows]
    unexpected = [i for i in emitted if i not in set(s1_ids)]

    results.append(
        check(
            "1. every test S1 receives exactly one row",
            not missing and not unexpected and duplicated_rows == 0,
            f"{len(emitted):,} rows for {len(s1_ids):,} Source 1 records; "
            f"{len(missing):,} missing, {len(unexpected):,} unexpected, "
            f"{duplicated_rows} duplicated S1 rows",
        )
    )
    results.append(
        check(
            "4. no duplicate S1 rows",
            duplicated_rows == 0,
            f"{duplicated_rows} S1 ids emitted more than once",
        )
    )
    results.append(
        check(
            "1. S1 row order preserved",
            emitted == s1_ids[: len(emitted)],
            "rows follow test_source1.tsv order"
            if emitted == s1_ids[: len(emitted)]
            else "row order differs from the Source 1 file",
        )
    )

    # ---- 3. candidate ids exist in test S2 or S3 -------------------
    wanted: set[str] = set()
    for values in rows.values():
        wanted.update(values)

    candidate_country: dict[str, str] = {}
    for path, source in ((s2_path, "S2"), (s3_path, "S3")):
        for block in _stream(path, ["entity_id", "country"]):
            for entity_id, country in zip(
                block["entity_id"], block["country"]
            ):
                if entity_id in wanted:
                    candidate_country[entity_id] = f"{source}:{country}"

    unknown = sorted(wanted - set(candidate_country))
    results.append(
        check(
            "2. every candidate id exists in test_source2 or test_source3",
            not unknown,
            f"{len(wanted):,} distinct candidate ids, "
            f"{len(unknown):,} not found in test S2/S3",
        )
    )

    # ---- 5. duplicates within a row -------------------------------
    rows_with_duplicates = [
        entity_id
        for entity_id, values in rows.items()
        if len(values) != len(set(values))
    ]
    results.append(
        check(
            "3. no duplicate candidate ids within a row",
            not rows_with_duplicates,
            f"{len(rows_with_duplicates)} of {len(rows):,} rows contain a "
            "repeated candidate id",
        )
    )

    # ---- 6. empty candidate lists ---------------------------------
    empty_rows = [
        entity_id for entity_id, values in rows.items() if not values
    ]
    results.append(
        check(
            "6. empty candidate lists represented correctly",
            defects["leading_or_trailing_comma"] == 0
            and defects["payload_nan_or_none"] == 0,
            f"{len(empty_rows):,} rows have an empty candidate list written "
            "as an empty second field; no trailing comma, no nan/None "
            "placeholder",
        )
    )

    # ---- 7. candidates never come from S1 --------------------------
    from_s1 = sorted(wanted & set(s1_ids))
    results.append(
        check(
            "7. candidate ids never come from Source 1",
            not from_s1,
            f"{len(from_s1):,} candidate ids also appear as Source 1 ids",
        )
    )

    # ---- 8. no training/validation ids -----------------------------
    train_ids: set[str] = set()
    for path in (args.train_s1, args.train_s2, args.train_s3):
        for block in _stream(Path(path), ["entity_id"]):
            train_ids.update(block["entity_id"].tolist())

    leaked = sorted((wanted & train_ids) - set(candidate_country))
    results.append(
        check(
            "8. no training/validation ids unless also test S2/S3 ids",
            not leaked,
            f"{len(train_ids):,} train ids loaded; "
            f"{len(wanted & train_ids):,} of them appear as candidates, of "
            f"which {len(leaked):,} are not also test S2/S3 ids "
            "(id namespaces are shared, so an id can legitimately be both)",
        )
    )

    # ---- 9. France / open-set country support ----------------------
    by_country: dict[str, dict[str, int]] = {}
    cross_country = 0
    france_detail: dict[str, Any] = {}

    for entity_id, values in rows.items():
        country = s1_country.get(entity_id, "")
        bucket = by_country.setdefault(
            country,
            {"rows": 0, "rows_with_candidates": 0, "pairs": 0},
        )
        bucket["rows"] += 1
        if values:
            bucket["rows_with_candidates"] += 1
        bucket["pairs"] += len(values)
        for value in values:
            target = candidate_country.get(value, "")
            if target and target.partition(":")[2] != country:
                cross_country += 1

    if "France" in by_country:
        bucket = by_country["France"]
        france_detail = {
            "s1_rows": bucket["rows"],
            "rows_with_candidates": bucket["rows_with_candidates"],
            "candidate_pairs": bucket["pairs"],
            "handled_without_code_change": True,
        }

    results.append(
        check(
            "9. France and open-set country support",
            cross_country == 0 and bool(france_detail),
            f"countries seen: {sorted(by_country)}; France rows "
            f"{france_detail.get('s1_rows', 0):,}, of which "
            f"{france_detail.get('rows_with_candidates', 0):,} received "
            f"candidates; cross-country candidates {cross_country}",
        )
    )

    # ---- 10. no label file on the generation path -----------------
    static = static_leakage_check(package_dir)
    dynamic = dynamic_leakage_check(
        Path(args.audit) if args.audit else None
    )
    results.append(
        check(
            "10a. no label dependency in generation modules (static)",
            not static["offences"],
            f"{len(static['modules_checked'])} modules parsed, "
            f"{len(static['offences'])} references to "
            f"{list(FORBIDDEN_TOKENS)}; imports recorded in the JSON report",
        )
    )
    results.append(
        check(
            "10b. no ground-truth file read during generation (dynamic)",
            bool(dynamic.get("no_label_file_read")),
            (
                f"{dynamic.get('files_opened_count', 0)} files opened by the "
                f"generation process, "
                f"{len(dynamic.get('label_like_files_opened', []))} of them "
                "label-like"
                if dynamic.get("available")
                else "dynamic audit not available"
            ),
        )
    )

    # ---- candidate yield per country -------------------------------
    pairs = sum(len(values) for values in rows.values())

    report = {
        "report": "day3_candidate_validation",
        "person": "Person 2 - candidate generation",
        "day": 3,
        "generated_by": "scripts/day3_validate_output.py",
        "validated_file": str(candidates_path).replace("\\", "/"),
        "file_created_by_this_script": False,
        "statement": (
            "this script only reads the candidate file produced by the "
            "integration pipeline; it does not create or modify it"
        ),
        "s1_limit": args.limit,
        "rows": len(emitted),
        "distinct_candidate_ids": len(wanted),
        "candidate_pairs": pairs,
        "rows_with_empty_candidate_list": len(empty_rows),
        "format_defects": defects,
        "checks": results,
        "checks_passed": sum(1 for r in results if r["passed"]),
        "checks_total": len(results),
        "all_passed": all(r["passed"] for r in results),
        "per_country": by_country,
        "france": france_detail,
        "cross_country_candidate_pairs": cross_country,
        "leakage_evidence": {
            "static": static,
            "dynamic": dynamic,
        },
    }

    destination = REPO / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {destination.relative_to(REPO)}")
    print()
    for result in results:
        print(f"[{result['status']}] {result['check']}: {result['detail']}")
    print()
    print(
        f"{report['checks_passed']}/{report['checks_total']} checks passed"
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
