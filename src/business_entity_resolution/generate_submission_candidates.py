"""Generate the submission-view ``output/candidate_pairs.tsv`` from test sources.

This is the full-test entry point. It is NOT part of Day-2 validation and
must not be run until the chunked architecture described in the README is in
place: it still holds S2 and S3 fully in memory.

Output contract (docs/interfaces.md section 7):

    source1_entity_id<TAB>candidate_entity_ids

* UTF-8 TSV, LF line endings, exactly two columns.
* One row for every test S1 ID, including entities with no candidates.
* Comma-separated S2/S3 IDs, no duplicates.
* This must be the exact candidate set handed to model inference.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

from business_entity_resolution.candidate_generation import (
    CandidateGenerationConfig,
    CandidateGenerator,
)
from business_entity_resolution.config import DATASET_ENV_VAR
from business_entity_resolution.normalization import (
    NormalizedRecord,
    normalize_record,
)
from business_entity_resolution.schemas import CANDIDATE_RESULTS_COLUMNS


DEFAULT_OUTPUT_PATH = Path("output") / "candidate_pairs.tsv"


def load_tsv(path: Path) -> list[dict]:
    records: list[dict] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        required = {
            "entity_id",
            "business_name",
            "business_address",
            "country",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path} missing columns: {sorted(missing)}"
            )

        for row in reader:
            records.append(
                {
                    "entity_id": str(row["entity_id"]),
                    "business_name": row["business_name"] or "",
                    "business_address": row["business_address"] or "",
                    "country": row["country"] or "",
                }
            )

    return records


def normalize_records(records) -> list[NormalizedRecord]:
    return [normalize_record(record) for record in records]


def resolve_dataset_root(explicit: str | None) -> Path:
    """Resolve the dataset root without exploding at import time."""

    if explicit:
        return Path(explicit)

    from_environment = os.environ.get(DATASET_ENV_VAR)

    if not from_environment:
        raise SystemExit(
            f"Dataset root not provided. Pass --dataset-root or set "
            f"{DATASET_ENV_VAR}."
        )

    return Path(from_environment)


def open_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = path.open("w", encoding="utf-8", newline="")

    writer = csv.writer(
        handle,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writerow(list(CANDIDATE_RESULTS_COLUMNS))

    return handle, writer


def write_candidate_row(writer, s1_entity_id: str, evidence_list) -> int:
    """Write one S1 row. Returns the number of candidates written."""

    seen: set[str] = set()
    candidate_ids: list[str] = []

    for evidence in evidence_list:
        candidate_id = str(evidence.candidate_entity_id)

        if candidate_id in seen:
            continue

        seen.add(candidate_id)
        candidate_ids.append(candidate_id)

    # An entity with no candidates still gets a row, with an empty list.
    writer.writerow([s1_entity_id, ",".join(candidate_ids)])

    return len(candidate_ids)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate_pairs.tsv for the test sources. "
            "Full-scale only."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N S1 records (smoke testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:

    args = parse_args(argv)

    dataset_root = resolve_dataset_root(args.dataset_root)
    test_dir = dataset_root / "test"
    output_path = Path(args.output)

    print(f"Dataset root: {dataset_root}")
    print(f"Output:       {output_path}")
    print(f"Config:       {CandidateGenerationConfig()}")

    started = time.perf_counter()

    print("\nLoading test files...")

    s1_raw = load_tsv(test_dir / "test_source1.tsv")
    s2_raw = load_tsv(test_dir / "test_source2.tsv")
    s3_raw = load_tsv(test_dir / "test_source3.tsv")

    print(f"S1: {len(s1_raw):,}")
    print(f"S2: {len(s2_raw):,}")
    print(f"S3: {len(s3_raw):,}")

    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be greater than 0")
        s1_raw = s1_raw[: args.limit]
        print(f"Limiting to first {len(s1_raw):,} S1 records.")

    print("\nNormalizing...")

    source1 = normalize_records(s1_raw)
    del s1_raw
    source2 = normalize_records(s2_raw)
    del s2_raw
    source3 = normalize_records(s3_raw)
    del s3_raw

    print("Normalization complete.")

    print("\nBuilding S2 candidate generator...")
    generator_s2 = CandidateGenerator(target_records=source2)

    print("Building S3 candidate generator...")
    generator_s3 = CandidateGenerator(target_records=source3)

    total = len(source1)
    running_total = 0

    print(f"\nGenerating candidates for {total:,} S1 records...")

    # Rows are streamed straight to disk so the full candidate set is never
    # resident in memory at once.
    handle, writer = open_writer(output_path)

    try:
        for position, record in enumerate(source1, start=1):

            merged = generator_s2.generate_for_one(record)
            merged += generator_s3.generate_for_one(record)

            seen: set[str] = set()
            unique: list = []

            for evidence in merged:
                candidate_id = str(evidence.candidate_entity_id)
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                unique.append(evidence)

            running_total += write_candidate_row(
                writer, record.entity_id, unique
            )

            if position % 5000 == 0 or position == total:
                elapsed = time.perf_counter() - started
                print(
                    f"[{position:,}/{total:,}] "
                    f"avg candidates = {running_total / position:.2f} "
                    f"elapsed = {elapsed:.0f}s "
                    f"eta = "
                    f"{elapsed / position * (total - position):.0f}s"
                )
    finally:
        handle.close()

    print("\n" + "=" * 56)
    print("FINAL CANDIDATE GENERATION COMPLETE")
    print("=" * 56)
    print(f"S1 records written: {total:,}")
    print(f"Candidate pairs:    {running_total:,}")
    print(f"Average candidates/S1: {running_total / max(1, total):.2f}")
    print(f"Runtime:            {time.perf_counter() - started:.0f}s")
    print(f"Output:             {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
