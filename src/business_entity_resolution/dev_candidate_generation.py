from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time
import tracemalloc

import pandas as pd

from .candidate_generation import CandidateGenerator, CandidateGenerationConfig
from .normalization import normalize_record
from .metrics import evaluate_candidate_recall, find_missed_pairs


def load_source(path: Path) -> list:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
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

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    ground_truth: dict[str, set[str]] = {}

    for row in df.to_dict(orient="records"):
        source1_id = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matched_entity_ids"]).strip()

        if not raw_matches:
            matches = set()
        else:
            matches = {
                value.strip()
                for value in raw_matches.split(",")
                if value.strip()
            }

        ground_truth[source1_id] = matches

    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Person 2 candidate generation on dev data."
    )

    parser.add_argument(
        "--dev-dir",
        default="data/dev/train",
        help="Development training directory.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of Source 1 records to evaluate.",
    )

    parser.add_argument(
        "--blocking-backend",
        choices=("legacy", "compact"),
        default="legacy",
        help=(
            "Blocking index implementation. 'legacy' is the Day 2 "
            "in-memory dictionary index. 'compact' is the hashed CSR "
            "index and produces identical candidates with a much "
            "smaller memory footprint."
        ),
    )

    parser.add_argument(
        "--release-target-records",
        action="store_true",
        help=(
            "Drop this process's reference to the target records once "
            "the indexes are built. Requires --blocking-backend compact, "
            "because the legacy index retains the records."
        ),
    )

    args = parser.parse_args()

    blocking_backend = args.blocking_backend

    dev_dir = Path(args.dev_dir)

    started = time.perf_counter()
    tracemalloc.start()

    s1_path = dev_dir / "train_source1.tsv"
    s2_path = dev_dir / "train_source2.tsv"
    s3_path = dev_dir / "train_source3.tsv"
    gt_path = dev_dir / "train_ground_truth.tsv"

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print("Loading Source 1...")
    source1 = load_source(s1_path)

    print("Loading Source 2...")
    source2 = load_source(s2_path)

    print("Loading Source 3...")
    source3 = load_source(s3_path)

    print(f"S1 records loaded: {len(source1):,}")
    print(f"S2 records:        {len(source2):,}")
    print(f"S3 records:        {len(source3):,}")

    # ---------------------------------------------------------
    # LIMIT S1 FOR BENCHMARK
    # ---------------------------------------------------------

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than 0")

        source1 = source1[:args.limit]

        print(
            f"\nUsing first {len(source1):,} Source 1 records "
            f"for evaluation."
        )

    # ---------------------------------------------------------
    # GROUND TRUTH
    # ---------------------------------------------------------

    print("\nLoading ground truth...")

    full_ground_truth = load_ground_truth(gt_path)

    ground_truth = {
        record.entity_id: full_ground_truth.get(
            record.entity_id,
            set(),
        )
        for record in source1
    }

    print(
        f"Ground-truth S1 entities used: "
        f"{len(ground_truth):,}"
    )

    # ---------------------------------------------------------
    # CONFIG
    #
    # Use the dataclass defaults so the benchmark always measures the
    # configuration the pipeline actually ships with.
    # ---------------------------------------------------------

    config = CandidateGenerationConfig()

    print("\nConfiguration:")
    print(f"  {config}")
    print(f"  blocking_backend = {blocking_backend}")

    build_started = time.perf_counter()
    # ---------------------------------------------------------
    # S2
    # ---------------------------------------------------------

    print("\nBuilding S2 candidate generator...")

    generator_s2 = CandidateGenerator(
        target_records=source2,
        config=config,
        blocking_backend=blocking_backend,
    )

    # ---------------------------------------------------------
    # S3
    # ---------------------------------------------------------

    print("Building S3 candidate generator...")

    generator_s3 = CandidateGenerator(
        target_records=source3,
        config=config,
        blocking_backend=blocking_backend,
    )

    # ---------------------------------------------------------
    # FREE TARGET RECORDS
    #
    # The retrievers keep only entity ids, so their records can be
    # released. The legacy blocking index still holds them, so this is
    # only possible with the compact backend.
    # ---------------------------------------------------------

    if args.release_target_records:

        if blocking_backend != "compact":
            raise ValueError(
                "--release-target-records requires "
                "--blocking-backend compact; the legacy index keeps a "
                "reference to the target records"
            )

        del source2
        del source3

        print("Released target records after index build.")

    # ---------------------------------------------------------
    # GENERATE
    # ---------------------------------------------------------

    print("Generating S2 candidates...")

    generate_started = time.perf_counter()

    candidates_s2 = generator_s2.generate(source1)

    print("Generating S3 candidates...")

    candidates_s3 = generator_s3.generate(source1)

    generate_seconds = time.perf_counter() - generate_started

    build_seconds = generate_started - build_started

    # ---------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------

    all_candidates = {}

    for source1_id, evidence_list in candidates_s2.items():
        all_candidates[source1_id] = list(evidence_list)

    for source1_id, evidence_list in candidates_s3.items():
        all_candidates.setdefault(source1_id, [])
        all_candidates[source1_id].extend(evidence_list)

    print("\nCandidate generation completed.")

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    report = evaluate_candidate_recall(
        candidate_results=all_candidates,
        ground_truth=ground_truth,
    )

    print("\n" + "=" * 60)
    print("PERSON 2 CANDIDATE GENERATION REPORT")
    print("=" * 60)

    print(
        f"Total true pairs:              "
        f"{report.total_true_pairs:,}"
    )

    print(
        f"Retrieved true pairs:          "
        f"{report.retrieved_true_pairs:,}"
    )

    print(
        f"Missed true pairs:             "
        f"{report.missed_true_pairs:,}"
    )

    print(
        f"Pair recall:                   "
        f"{report.pair_recall:.6%}"
    )

    print()

    print(
        f"Total Source 1 entities:       "
        f"{report.total_source1_entities:,}"
    )

    print(
        f"Entities with all links:       "
        f"{report.entities_with_all_links_retrieved:,}"
    )

    print(
        f"Entity/all-links coverage:     "
        f"{report.entity_coverage:.6%}"
    )

    print()

    print(
        f"Average candidates/S1:         "
        f"{report.average_candidate_count:.2f}"
    )

    print(
        f"Median candidates/S1:          "
        f"{report.median_candidate_count:.2f}"
    )

    print(
        f"P95 candidates/S1:             "
        f"{report.p95_candidate_count:.2f}"
    )

    print(
        f"P99 candidates/S1:             "
        f"{report.p99_candidate_count:.2f}"
    )

    print(
        f"Maximum candidates/S1:         "
        f"{report.maximum_candidate_count}"
    )

    # ---------------------------------------------------------
    # ROUTE CONTRIBUTION
    # ---------------------------------------------------------

    route_pairs = defaultdict(int)
    route_entities = defaultdict(int)

    for evidence_list in all_candidates.values():
        seen_routes = set()
        for evidence in evidence_list:
            for route in evidence.routes:
                route_pairs[route] += 1
            seen_routes.update(evidence.routes)
        for route in seen_routes:
            route_entities[route] += 1

    print()
    print("Route contribution (candidate pairs / S1 entities touched):")
    for route, pairs in sorted(
        route_pairs.items(), key=lambda kv: -kv[1]
    ):
        print(
            f"  {route:26s} {pairs:>12,}  "
            f"{route_entities[route]:>8,}"
        )

    # ---------------------------------------------------------
    # COST
    # ---------------------------------------------------------

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()

    print()
    print(
        f"S1 entities evaluated:         "
        f"{len(source1):,}"
    )
    print(
        f"Index build time:              "
        f"{build_seconds:.1f}s"
    )
    print(
        f"Candidate generation time:     "
        f"{generate_seconds:.1f}s "
        f"({generate_seconds / max(1, len(source1)) * 1000:.1f} ms/S1)"
    )
    print(
        f"Total wall clock:              "
        f"{time.perf_counter() - started:.1f}s"
    )
    print(
        f"Python peak memory:            "
        f"{peak_bytes / 1024 ** 3:.2f} GB"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # MISSED PAIRS
    # ---------------------------------------------------------

    missed = find_missed_pairs(
        candidate_results=all_candidates,
        ground_truth=ground_truth,
    )

    if missed:
        print("\nSample missed true pairs:")

        shown = 0

        for source1_id, missing_ids in missed.items():

            for candidate_id in sorted(missing_ids):

                print(
                    f"  S1={source1_id} -> "
                    f"missing candidate={candidate_id}"
                )

                shown += 1

                if shown >= 20:
                    break

            if shown >= 20:
                break

    else:
        print("\nNo missed true pairs.")


if __name__ == "__main__":
    main()
