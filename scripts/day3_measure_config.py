"""Day 3 / Task 2: measure one candidate-generation configuration.

One configuration per process, so the reported runtime and peak RSS belong
to that configuration alone and are not contaminated by another
configuration's index still being resident.

The retrieval route set is never touched. Only the numeric budget knobs on
the frozen ``CandidateGenerationConfig`` differ between configurations.

Usage:
    python scripts/day3_measure_config.py --config baseline \
        --output data/work/day3/baseline.json
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from business_entity_resolution.candidate_generation import (  # noqa: E402
    CandidateGenerationConfig,
    CandidateGenerator,
)
from business_entity_resolution.chunked_generation import (  # noqa: E402
    PeakMemoryMonitor,
)
from business_entity_resolution.dev_candidate_generation import (  # noqa: E402
    load_ground_truth,
    load_source,
)
from business_entity_resolution.metrics import (  # noqa: E402
    evaluate_candidate_recall,
)

# ==============================================================
# CONFIGURATION GRID
# ==============================================================
#
# A-E are the configurations required by the Day 3 brief. Every one of them
# is derived from the frozen baseline by a stated rule, so no value is
# chosen by hand: the top-K variants halve a budget, and the cap variants
# multiply every posting cap by a fixed factor. Route semantics, score
# floors, n-gram range and the route set are identical in all of them.

CAP_KEYS = (
    "rare_token_max_postings",
    "numeric_token_max_postings",
    "translit_token_max_postings",
    "address_token_max_postings",
    "address_pair_max_postings",
)

# address_pair must stay >= 1; below that the route can never fire.
CAP_FLOOR = {"address_pair_max_postings": 1}


def _scaled_caps(factor: float) -> dict[str, int]:
    baseline = CandidateGenerationConfig()
    result: dict[str, int] = {}
    for key in CAP_KEYS:
        raw = int(getattr(baseline, key) * factor)
        result[key] = max(CAP_FLOOR.get(key, 1), raw)
    return result


def build_configs() -> dict[str, dict[str, Any]]:
    baseline = CandidateGenerationConfig()

    definitions: list[dict[str, Any]] = [
        {
            "name": "A_baseline",
            "label": "A. Baseline (frozen Day-2 configuration)",
            "rule": "frozen CandidateGenerationConfig() defaults, unchanged",
            "overrides": {},
        },
        {
            "name": "B_name_top_k_50",
            "label": "B. Reduced name retrieval top-K (100 -> 50)",
            "rule": "halve name_top_k",
            "overrides": {"name_top_k": 50},
        },
        {
            "name": "C_address_top_k_50",
            "label": "C. Reduced address retrieval top-K (100 -> 50)",
            "rule": "halve address_top_k",
            "overrides": {"address_top_k": 50},
        },
        {
            "name": "D_posting_caps_x0.6",
            "label": "D. Reduced posting caps (x0.6)",
            "rule": "multiply every posting cap by 0.6, floor 1",
            "overrides": _scaled_caps(0.6),
        },
        {
            "name": "E_combined_conservative",
            "label": "E. Combined conservative reduction "
            "(caps x0.8, both top-K x0.75)",
            "rule": "posting caps x0.8; name_top_k and address_top_k x0.75",
            "overrides": {
                **_scaled_caps(0.8),
                "name_top_k": int(baseline.name_top_k * 0.75),
                "address_top_k": int(baseline.address_top_k * 0.75),
            },
        },
    ]

    return {
        definition["name"]: {
            "configuration": definition["name"],
            "label": definition["label"],
            "derivation_rule": definition["rule"],
            "overrides": definition["overrides"],
        }
        for definition in definitions
    }


# ==============================================================
# MEASUREMENT
# ==============================================================

CHUNK = 2_000


def measure(
    name: str,
    dev_dir: Path,
    limit: int | None,
) -> dict[str, Any]:
    grid = build_configs()
    if name not in grid:
        raise SystemExit(
            f"unknown configuration {name!r}; "
            f"choose from {sorted(grid)}"
        )

    spec = grid[name]
    baseline = CandidateGenerationConfig()
    config = CandidateGenerationConfig(
        **{
            **{
                field.name: getattr(baseline, field.name)
                for field in dataclasses.fields(baseline)
            },
            **spec["overrides"],
        }
    )

    parameters = dataclasses.asdict(config)
    parameters["ngram_range"] = list(config.ngram_range)

    baseline_parameters = dataclasses.asdict(baseline)
    baseline_parameters["ngram_range"] = list(baseline.ngram_range)

    changed = {
        key: {"baseline": baseline_parameters[key],
              "candidate": parameters[key]}
        for key in parameters
        if parameters[key] != baseline_parameters[key]
    }

    monitor = PeakMemoryMonitor(interval=0.2)
    monitor.__enter__()

    wall_started = time.perf_counter()

    load_started = time.perf_counter()
    source1 = load_source(dev_dir / "train_source1.tsv")
    source2 = load_source(dev_dir / "train_source2.tsv")
    source3 = load_source(dev_dir / "train_source3.tsv")
    source2_records = len(source2)
    source3_records = len(source3)
    full_ground_truth = load_ground_truth(
        dev_dir / "train_ground_truth.tsv"
    )
    load_seconds = time.perf_counter() - load_started

    if limit is not None:
        source1 = source1[:limit]

    ground_truth = {
        record.entity_id: full_ground_truth.get(record.entity_id, set())
        for record in source1
    }

    print(
        f"[{name}] S1={len(source1):,} S2={len(source2):,} "
        f"S3={len(source3):,} ground_truth={len(ground_truth):,}",
        flush=True,
    )
    print(f"[{name}] load {load_seconds:.1f}s", flush=True)

    build_started = time.perf_counter()
    generator_s2 = CandidateGenerator(
        target_records=source2, config=config,
        blocking_backend="compact",
    )
    generator_s3 = CandidateGenerator(
        target_records=source3, config=config,
        blocking_backend="compact",
    )
    build_seconds = time.perf_counter() - build_started
    print(f"[{name}] index build {build_seconds:.1f}s", flush=True)

    route_pairs: dict[str, int] = {}
    route_entities: dict[str, int] = {}

    total_true_pairs = 0
    retrieved_true_pairs = 0
    entities_with_all_links = 0
    candidate_counts: list[int] = []
    rows_with_candidates = 0
    generation_seconds = 0.0

    generate_started = time.perf_counter()

    for offset in range(0, len(source1), CHUNK):
        block = source1[offset:offset + CHUNK]

        chunk_started = time.perf_counter()

        results_s2 = generator_s2.generate(block)
        results_s3 = generator_s3.generate(block)

        merged: dict[str, list] = {
            record.entity_id: [] for record in block
        }
        for mapping in (results_s2, results_s3):
            for entity_id, evidence_list in mapping.items():
                merged[entity_id].extend(evidence_list)

        for evidence_list in merged.values():
            if evidence_list:
                rows_with_candidates += 1
            seen_routes: set[str] = set()
            for evidence in evidence_list:
                for route in evidence.routes:
                    route_pairs[route] = route_pairs.get(route, 0) + 1
                seen_routes.update(evidence.routes)
            for route in seen_routes:
                route_entities[route] = route_entities.get(route, 0) + 1

        report = evaluate_candidate_recall(
            candidate_results=merged,
            ground_truth={
                record.entity_id: ground_truth[record.entity_id]
                for record in block
            },
        )

        total_true_pairs += report.total_true_pairs
        retrieved_true_pairs += report.retrieved_true_pairs
        entities_with_all_links += (
            report.entities_with_all_links_retrieved
        )
        candidate_counts.extend(
            len({
                evidence.candidate_entity_id
                for evidence in evidence_list
            })
            for evidence_list in merged.values()
        )

        generation_seconds += time.perf_counter() - chunk_started

        done = min(offset + CHUNK, len(source1))
        elapsed = time.perf_counter() - generate_started
        print(
            f"[{name}] {done:,}/{len(source1):,} "
            f"{elapsed:.0f}s "
            f"eta {elapsed / max(1, done) * (len(source1) - done):.0f}s",
            flush=True,
        )

    generate_seconds = time.perf_counter() - generate_started
    total_seconds = time.perf_counter() - wall_started

    del generator_s2
    del generator_s3
    del source2
    del source3
    gc.collect()

    monitor.__exit__()

    s1_count = len(source1)
    total_pairs = sum(candidate_counts)
    target_counts = {
        "s2_target_count": source2_records,
        "s3_target_count": source3_records,
    }

    ordered = sorted(candidate_counts)

    def percentile(fraction: float) -> float:
        if not ordered:
            return 0.0
        if len(ordered) == 1:
            return float(ordered[0])
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered))
        weight = position - lower
        return (
            ordered[lower] * (1.0 - weight)
            + ordered[upper] * weight
        )

    s1_with_truth = sum(
        1 for record in source1 if ground_truth[record.entity_id]
    )

    return {
        "configuration": name,
        "label": spec["label"],
        "derivation_rule": spec["derivation_rule"],
        "parameters": parameters,
        "parameters_changed_from_baseline": changed,
        "routes_changed": [],
        "route_semantics_changed": False,
        "blocking_backend": "compact",
        "evaluated_on": "development subset (data/dev/train)",
        "ground_truth_file": "data/dev/train/train_ground_truth.tsv",
        "s1_count": s1_count,
        "s1_count_with_true_links": s1_with_truth,
        "s2_target_count": target_counts["s2_target_count"],
        "s3_target_count": target_counts["s3_target_count"],
        "candidate_row_count": rows_with_candidates,
        "candidate_row_count_note": (
            "Source 1 rows that received at least one candidate; the "
            "deliverable itself always emits one row per Source 1 record"
        ),
        "total_candidate_pairs": total_pairs,
        "average_candidates_per_s1": (
            total_pairs / s1_count if s1_count else 0.0
        ),
        "median_candidates_per_s1": percentile(0.50),
        "p95_candidates_per_s1": percentile(0.95),
        "p99_candidates_per_s1": percentile(0.99),
        "maximum_candidates_per_s1": max(candidate_counts)
        if candidate_counts
        else 0,
        "total_true_pairs": total_true_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
        "missed_true_pairs": (
            total_true_pairs - retrieved_true_pairs
        ),
        "pair_recall": (
            retrieved_true_pairs / total_true_pairs
            if total_true_pairs
            else 1.0
        ),
        "full_entity_recall": (
            entities_with_all_links / s1_count if s1_count else 0.0
        ),
        "entities_with_all_links_retrieved": entities_with_all_links,
        "runtime": {
            "total_wall_seconds": total_seconds,
            "data_load_seconds": load_seconds,
            "index_build_seconds": build_seconds,
            "candidate_generation_seconds": generate_seconds,
            "measured_generation_seconds": generation_seconds,
            "ms_per_s1": (
                generation_seconds / s1_count * 1000.0
                if s1_count
                else 0.0
            ),
        },
        "peak_ram": {
            "process_peak_rss_gb": monitor.peak_gb,
            "method": "psutil RSS sampled every 200 ms on a background thread",
        },
        "route_contribution": {
            "candidate_pairs": route_pairs,
            "s1_entities_touched": route_entities,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        choices=sorted(build_configs()),
    )
    parser.add_argument("--dev-dir", default="data/dev/train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = measure(args.config, Path(args.dev_dir), args.limit)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"configuration        {result['configuration']}")
    print(
        f"candidate pairs      "
        f"{result['total_candidate_pairs']:,}"
    )
    print(
        f"average/S1           "
        f"{result['average_candidates_per_s1']:.2f}"
    )
    print(f"pair recall          {result['pair_recall']:.6%}")
    print(
        f"full-entity recall   "
        f"{result['full_entity_recall']:.4%}"
    )
    print(
        f"wall clock           "
        f"{result['runtime']['total_wall_seconds']:.1f}s"
    )
    print(
        f"peak RSS             "
        f"{result['peak_ram']['process_peak_rss_gb']:.2f} GB"
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
