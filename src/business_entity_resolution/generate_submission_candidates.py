from __future__ import annotations

import csv
import os
from pathlib import Path

from business_entity_resolution.candidate_generation import (
    CandidateGenerationConfig,
    CandidateGenerator,
)
from business_entity_resolution.normalization import normalize_record


DATASET_ROOT = Path(os.environ["AMAZON_ML_DATASET_ROOT"])
TEST_DIR = DATASET_ROOT / "test"
OUTPUT_PATH = Path("output") / "candidate_pairs.tsv"


def load_tsv(path: Path):
    records = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

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


def normalize_records(records):
    return [
        normalize_record(record)
        for record in records
    ]


def write_candidates(source1, candidates):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids",
            ]
        )

        for s1 in source1:

            evidence_list = candidates.get(
                s1.entity_id,
                [],
            )

            candidate_ids = []
            seen = set()

            for evidence in evidence_list:

                candidate_id = str(
                    evidence.candidate_entity_id
                )

                if candidate_id not in seen:
                    seen.add(candidate_id)
                    candidate_ids.append(candidate_id)

            writer.writerow(
                [
                    s1.entity_id,
                    ",".join(candidate_ids),
                ]
            )


def main():

    print("Dataset:")
    print(DATASET_ROOT)

    print("\nLoading test files...")

    s1_raw = load_tsv(
        TEST_DIR / "test_source1.tsv"
    )

    s2_raw = load_tsv(
        TEST_DIR / "test_source2.tsv"
    )

    s3_raw = load_tsv(
        TEST_DIR / "test_source3.tsv"
    )

    print(f"S1: {len(s1_raw):,}")
    print(f"S2: {len(s2_raw):,}")
    print(f"S3: {len(s3_raw):,}")

    print("\nNormalizing...")

    s1 = normalize_records(s1_raw)
    s2 = normalize_records(s2_raw)
    s3 = normalize_records(s3_raw)

    print("Normalization complete.")

    config = CandidateGenerationConfig(
        name_top_k=100,
        address_top_k=20,
        name_min_score=0.30,
        address_min_score=0.0,
        rare_token_max_postings=50,
        numeric_token_max_postings=100,
    )

    print("\nBuilding S2 candidate generator...")

    generator_s2 = CandidateGenerator(
        target_records=s2,
        config=config,
    )

    print("Building S3 candidate generator...")

    generator_s3 = CandidateGenerator(
        target_records=s3,
        config=config,
    )

    candidates_by_source1 = {}

    total = len(s1)

    print(
        f"\nGenerating candidates for {total:,} S1 records..."
    )

    for i, record in enumerate(s1, start=1):

        candidates_s2 = generator_s2.generate_for_one(
            record
        )

        candidates_s3 = generator_s3.generate_for_one(
            record
        )

        merged = []
        seen = set()

        for evidence in candidates_s2 + candidates_s3:

            candidate_id = str(
                evidence.candidate_entity_id
            )

            if candidate_id not in seen:
                seen.add(candidate_id)
                merged.append(evidence)

        candidates_by_source1[
            record.entity_id
        ] = merged

        if i % 500 == 0 or i == total:

            total_so_far = sum(
                len(values)
                for values in candidates_by_source1.values()
            )

            print(
                f"[{i:,}/{total:,}] "
                f"avg candidates = "
                f"{total_so_far / i:.2f}"
            )

    print("\nWriting candidate_pairs.tsv...")

    write_candidates(
        s1,
        candidates_by_source1,
    )

    total_pairs = sum(
        len(values)
        for values in candidates_by_source1.values()
    )

    print("\n========================================")
    print("FINAL CANDIDATE GENERATION COMPLETE")
    print("========================================")
    print(f"S1 records: {len(s1):,}")
    print(f"Candidate pairs: {total_pairs:,}")
    print(
        f"Average candidates/S1: "
        f"{total_pairs / len(s1):.2f}"
    )
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()