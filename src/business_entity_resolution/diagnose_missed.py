from __future__ import annotations

from pathlib import Path

import pandas as pd

from .candidate_generation import (
    CandidateGenerationConfig,
    CandidateGenerator,
)
from .normalization import normalize_record
from .metrics import find_missed_pairs


DEV_DIR = Path("data/dev/train")

# Same slice size the benchmark uses by default.
BENCHMARK_LIMIT = 1000


def load_source(path: Path) -> list:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    return [
        normalize_record(row)
        for row in df.to_dict(orient="records")
    ]


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    result = {}

    for row in df.to_dict(orient="records"):
        s1_id = row["source1_entity_id"].strip()
        raw = row["matched_entity_ids"].strip()

        if raw:
            result[s1_id] = {
                x.strip()
                for x in raw.split(",")
                if x.strip()
            }
        else:
            result[s1_id] = set()

    return result


def show_record(label, record):
    print(f"\n{label}")
    print("-" * 70)
    print("ID:")
    print(record.entity_id)

    print("Country:")
    print(record.country)

    print("\nName:")
    print(record.name_unicode)

    print("ASCII name:")
    print(record.name_ascii)

    print("Token-sorted name:")
    print(record.name_token_sorted)

    print("\nAddress:")
    print(record.address_unicode)

    print("ASCII address:")
    print(record.address_ascii)

    print("Token-sorted address:")
    print(record.address_token_sorted)

    print("\nName tokens:")
    print(record.name_tokens)

    print("Address tokens:")
    print(record.address_tokens)

    print("Address numbers:")
    print(record.address_numbers)

    print("Address missing:")
    print(record.address_missing)


def main():

    print("Loading data...")

    source1 = load_source(DEV_DIR / "train_source1.tsv")
    source2 = load_source(DEV_DIR / "train_source2.tsv")
    source3 = load_source(DEV_DIR / "train_source3.tsv")

    ground_truth = load_ground_truth(
        DEV_DIR / "train_ground_truth.tsv"
    )

    # Must be identical to the benchmark configuration, otherwise this tool
    # diagnoses a different system than dev_candidate_generation reports on.
    config = CandidateGenerationConfig()

    print("Config:", config)

    print("Building S2 generator...")
    generator_s2 = CandidateGenerator(
        target_records=source2,
        config=config,
    )

    print("Building S3 generator...")
    generator_s3 = CandidateGenerator(
        target_records=source3,
        config=config,
    )

    # The benchmark evaluates the first BENCHMARK_LIMIT S1 records unless
    # --limit says otherwise, so mirror that here.
    source1 = source1[:BENCHMARK_LIMIT]

    candidates_s2 = generator_s2.generate(source1)
    candidates_s3 = generator_s3.generate(source1)

    all_candidates = {}

    for s1_id, values in candidates_s2.items():
        all_candidates[s1_id] = list(values)

    for s1_id, values in candidates_s3.items():
        all_candidates.setdefault(s1_id, [])
        all_candidates[s1_id].extend(values)

    missed = find_missed_pairs(
        candidate_results=all_candidates,
        ground_truth={
            r.entity_id: ground_truth.get(r.entity_id, set())
            for r in source1
        },
    )

    print("\nMissed S1 entities:", len(missed))

    # Build lookup dictionaries.
    s1_lookup = {
        r.entity_id: r
        for r in source1
    }

    s2_lookup = {
        r.entity_id: r
        for r in source2
    }

    s3_lookup = {
        r.entity_id: r
        for r in source3
    }

    print("\n" + "=" * 80)
    print("FIRST 10 MISSED TRUE PAIRS")
    print("=" * 80)

    shown = 0

    for s1_id, missing_ids in missed.items():

        s1 = s1_lookup[s1_id]

        for target_id in sorted(missing_ids):

            if target_id.startswith("S2-"):
                target = s2_lookup.get(target_id)
            else:
                target = s3_lookup.get(target_id)

            if target is None:
                continue

            show_record("SOURCE 1", s1)
            show_record("TRUE TARGET", target)

            # Check raw route behavior directly.
            generator = (
                generator_s2
                if target_id.startswith("S2-")
                else generator_s3
            )

            blocking = generator.blocking_index.generate(s1)

            blocking_ids = {
                x.candidate_entity_id
                for x in blocking
            }

            name_results = generator.name_retriever.retrieve(s1)

            name_ids = {
                x.candidate_entity_id
                for x in name_results
            }

            address_results = generator.address_retriever.retrieve(s1)

            address_ids = {
                x.candidate_entity_id
                for x in address_results
            }

            print("\nROUTE DIAGNOSTIC")
            print("-" * 70)

            print(
                "Found by blocking:",
                target_id in blocking_ids,
            )

            print(
                "Found by name TF-IDF:",
                target_id in name_ids,
            )

            print(
                "Found by address TF-IDF:",
                target_id in address_ids,
            )

            if name_results:
                target_name_rank = [
                    x
                    for x in name_results
                    if x.candidate_entity_id == target_id
                ]

                if target_name_rank:
                    print(
                        "Name TF-IDF result:",
                        target_name_rank[0],
                    )
                else:
                    print(
                        "Target not in name TF-IDF top-K."
                    )

            if address_results:
                target_address_rank = [
                    x
                    for x in address_results
                    if x.candidate_entity_id == target_id
                ]

                if target_address_rank:
                    print(
                        "Address TF-IDF result:",
                        target_address_rank[0],
                    )
                else:
                    print(
                        "Target not in address TF-IDF top-K."
                    )

            print("\n" + "=" * 80)

            shown += 1

            if shown >= 10:
                return


if __name__ == "__main__":
    main()