"""Day 3 / Tasks 3 and 4: shard scalability measurement and full-test estimate.

Reads the artifacts of a real test-data shard run (country partitions,
per-chunk manifests, the merged output and the process log) and reports what
was actually measured, kept strictly separate from what is extrapolated.

The full-test numbers are labelled ESTIMATE everywhere. They are produced by
scaling the measured shard, and the scaling rule is written into the report
so the arithmetic can be checked.

Usage:
    python scripts/day3_scalability.py \
        --work-dir data/work/day3/shard_run1 \
        --s1 6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv \
        --output docs/day3_scalability_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

DEFAULT_CHUNK_SIZE = 50_000

# Full test set size, quoted in the Day-3 brief.
FULL_TEST_S1 = 1_732_544

WALL_RE = re.compile(r"total wall clock ([\d.]+)s")
RSS_RE = re.compile(r"process peak RSS ([\d.]+) GB")
PARTITION_RE = re.compile(r"partitioned in ([\d.]+)s")
INDEX_RE = re.compile(
    r"blocking index: ([\d,]+) targets in ([\d.]+)s"
)
FINALIZE_RE = re.compile(r"candidate pairs ([\d,]+)")
EMPTY_RE = re.compile(r"empty rows\s+([\d,]+)")


def percentile(ordered: list[int], fraction: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def directory_bytes(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def read_manifests(work_dir: Path) -> list[dict[str, Any]]:
    manifests = []
    for target in ("S2", "S3"):
        path = work_dir / target / "manifest.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["target_source"] = payload["target_source"]
        manifests.append(payload)
    return manifests


def read_output_counts(path: Path) -> dict[str, Any]:
    """One streaming pass over the deliverable: counts and the S1 order."""

    counts: list[int] = []
    empty_rows = 0
    rows = 0
    malformed = 0
    blank_payload_rows = 0

    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.count("\t") != 1:
                malformed += 1
                continue
            entity_id, _, payload = line.partition("\t")
            rows += 1
            if payload == "":
                blank_payload_rows += 1
                empty_rows += 1
                counts.append(0)
                continue
            values = [value for value in payload.split(",") if value]
            if any(value in ("nan", "None") for value in values):
                malformed += 1
            counts.append(len(values))

    ordered = sorted(counts)

    return {
        "header": header,
        "rows": rows,
        "malformed_rows": malformed,
        "rows_with_empty_candidate_list": blank_payload_rows,
        "empty_rows": empty_rows,
        "total_candidate_pairs": sum(counts),
        "average_candidates_per_s1": (
            sum(counts) / rows if rows else 0.0
        ),
        "median_candidates_per_s1": percentile(ordered, 0.50),
        "p95_candidates_per_s1": percentile(ordered, 0.95),
        "p99_candidates_per_s1": percentile(ordered, 0.99),
        "maximum_candidates_per_s1": max(counts) if counts else 0,
        "_counts": counts,
    }


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def find(pattern: re.Pattern[str], cast=float) -> Any:
        match = pattern.search(text)
        return cast(match.group(1)) if match else None

    return {
        "total_wall_seconds": find(WALL_RE),
        "process_peak_rss_gb": find(RSS_RE),
        "partition_seconds": [
            float(value) for value in PARTITION_RE.findall(text)
        ],
        "blocking_index_builds": [
            {"targets": int(targets.replace(",", "")), "seconds": seconds}
            for targets, seconds in INDEX_RE.findall(text)
        ],
        "finalized_candidate_pairs": find(FINALIZE_RE, int),
        "finalized_empty_rows": find(EMPTY_RE, int),
    }


def build_report(
    work_dir: Path,
    s1_path: Path,
    log_path: Path,
    chunk_size: int,
    limit: int,
) -> dict[str, Any]:
    output_path = work_dir / "candidate_pairs.tsv"
    if not output_path.is_file():
        raise SystemExit(f"missing shard output {output_path}")

    counts = read_output_counts(output_path)
    per_row = counts.pop("_counts")

    manifests = read_manifests(work_dir)
    log = parse_log(log_path)

    partition_files, partition_bytes = directory_bytes(
        work_dir / "partition_S2"
    )
    partition_files += directory_bytes(work_dir / "partition_S3")[0]
    partition_bytes += directory_bytes(work_dir / "partition_S3")[1]

    chunk_files, chunk_bytes = 0, 0
    for target in ("S2", "S3"):
        files, size = directory_bytes(work_dir / target)
        chunk_files += files
        chunk_bytes += size

    output_bytes = output_path.stat().st_size
    intermediate_bytes = partition_bytes + chunk_bytes

    chunk_rows = sum(
        record["row_count"]
        for manifest in manifests
        for record in manifest["chunks"]
    )
    generation_seconds = sum(
        record["seconds"]
        for manifest in manifests
        for record in manifest["chunks"]
    )

    per_target = {
        manifest["target_source"]: {
            "chunk_size": manifest["chunk_size"],
            "total_s1_rows": manifest["total_s1_rows"],
            "chunks": len(manifest["chunks"]),
            "candidate_pairs": sum(
                record["candidate_pairs"]
                for record in manifest["chunks"]
            ),
            "s1_with_candidates": sum(
                record["s1_with_candidates"]
                for record in manifest["chunks"]
            ),
            "generation_seconds": round(
                sum(
                    record["seconds"] for record in manifest["chunks"]
                ),
                3,
            ),
        }
        for manifest in manifests
    }

    s1_processed = counts["rows"]
    scale = FULL_TEST_S1 / s1_processed if s1_processed else 0.0
    measured_wall = log["total_wall_seconds"] or 0.0
    ms_per_s1 = (
        generation_seconds / chunk_rows * 1000.0 if chunk_rows else 0.0
    )

    # Index construction and partitioning happen once per target source and
    # do not scale with the number of Source 1 records, so they are carried
    # over unscaled and only the per-S1 generation term is multiplied.
    fixed_seconds = measured_wall - generation_seconds

    return {
        "report": "day3_scalability_report",
        "person": "Person 2 - candidate generation",
        "day": 3,
        "status": "MEASURED shard + ESTIMATE full test",
        "generated_by": "scripts/day3_scalability.py",
        "reproduce": (
            "python -m business_entity_resolution.run_chunked_generation "
            "--s1 <test_source1.tsv> --target-s2 <test_source2.tsv> "
            "--target-s3 <test_source3.tsv> --work-dir <dir> "
            "--output <dir>/candidate_pairs.tsv --chunk-size 2500 "
            "--limit 10000   &&   python scripts/day3_scalability.py ..."
        ),
        "shard": {
            "purpose": (
                "scalability and output integrity only; the test set has "
                "no ground truth, so no recall is computed or claimed here"
            ),
            "uses_real_test_files": True,
            "ground_truth_used": False,
            "s1_source": str(
                s1_path.relative_to(REPO)
                if s1_path.is_absolute()
                else s1_path
            ).replace("\\", "/"),
            "s1_limit": limit,
            "chunk_size_used": chunk_size,
            "target_sources_indexed_in_full": True,
            "note": (
                "the limit applies to Source 1 only; the full test Source 2 "
                "and Source 3 files are indexed, partitioned by country, "
                "so the measured cost is representative of the full run"
            ),
        },
        "measured": {
            "s1_records_processed": s1_processed,
            "s1_rows_written": counts["rows"],
            "malformed_rows": counts["malformed_rows"],
            "s1_rows_with_candidates": (
                s1_processed - counts["empty_rows"]
            ),
            "rows_with_empty_candidate_list": counts[
                "rows_with_empty_candidate_list"
            ],
            "total_candidate_pairs": counts["total_candidate_pairs"],
            "candidates_per_s1": {
                "average": counts["average_candidates_per_s1"],
                "median": counts["median_candidates_per_s1"],
                "p95": counts["p95_candidates_per_s1"],
                "p99": counts["p99_candidates_per_s1"],
                "maximum": counts["maximum_candidates_per_s1"],
            },
            "runtime": {
                "total_wall_seconds": measured_wall,
                "candidate_generation_seconds": round(
                    generation_seconds, 1
                ),
                "index_and_partition_seconds": round(
                    fixed_seconds, 1
                ),
                "partition_seconds_by_source": log[
                    "partition_seconds"
                ],
                "blocking_index_builds": log["blocking_index_builds"],
                "ms_per_s1": round(ms_per_s1, 3),
            },
            "peak_ram": {
                "process_peak_rss_gb": log["process_peak_rss_gb"],
                "method": (
                    "psutil RSS sampled every 250 ms on a background thread"
                ),
                "ram_is_bounded_by_target_index": True,
            },
            "disk": {
                "partition_files": partition_files,
                "partition_bytes": partition_bytes,
                "chunk_output_files": chunk_files,
                "chunk_output_bytes": chunk_bytes,
                "intermediate_bytes": intermediate_bytes,
                "intermediate_mb": round(
                    intermediate_bytes / 1024 ** 2, 1
                ),
                "final_output_bytes": output_bytes,
                "final_output_mb": round(output_bytes / 1024 ** 2, 1),
            },
            "per_target_source": per_target,
        },
        "estimate": {
            "label": "ESTIMATE - NOT MEASURED",
            "basis": (
                f"one measured shard of {s1_processed:,} real test Source 1 "
                f"records, scaled by {scale:.2f}x to the full test set of "
                f"{FULL_TEST_S1:,} Source 1 records"
            ),
            "scaling_rule": (
                "partitioning and target index construction run once per "
                "target source and are carried over unscaled; only the "
                "per-Source-1 generation term is multiplied by the scale "
                "factor, because Source 1 is streamed and never fully "
                "materialised"
            ),
            "full_test_s1_records": FULL_TEST_S1,
            "scale_factor": round(scale, 3),
            "candidate_generation_hours": round(
                generation_seconds * scale / 3600.0, 2
            ),
            "total_wall_hours": round(
                (fixed_seconds + generation_seconds * scale) / 3600.0, 2
            ),
            "total_wall_hours_upper_bound": round(
                (fixed_seconds + generation_seconds * scale) / 3600.0 * 1.25,
                2,
            ),
            "peak_ram_gb": log["process_peak_rss_gb"],
            "peak_ram_note": (
                "unchanged from the shard: peak memory is set by the "
                "largest single-country target index, not by the number of "
                "Source 1 records, so the full run is not expected to need "
                "more RAM than the measured shard"
            ),
            "chunks": {
                "with_default_chunk_size_50000_per_target_source": (
                    math.ceil(FULL_TEST_S1 / DEFAULT_CHUNK_SIZE)
                ),
                "note": (
                    "Source 2 and Source 3 are processed independently, so "
                    "each target source produces this many chunk files"
                ),
            },
            "candidate_pairs": {
                "value": int(
                    counts["total_candidate_pairs"] * scale
                ),
                "basis": (
                    f"{counts['total_candidate_pairs']:,} measured pairs "
                    f"x {scale:.2f}; assumes a per-S1 candidate yield "
                    "similar to the shard, which is an assumption because "
                    "the shard is the first 10,000 rows of the file and is "
                    "not a random sample"
                ),
            },
            "disk": {
                "partition_mb": round(partition_bytes / 1024 ** 2, 1),
                "chunk_output_gb": round(
                    chunk_bytes * scale / 1024 ** 3, 2
                ),
                "intermediate_gb": round(
                    intermediate_bytes * scale / 1024 ** 3, 2
                ),
                "final_output_gb": round(
                    output_bytes * scale / 1024 ** 3, 2
                ),
                "total_gb": round(
                    (intermediate_bytes + output_bytes) * scale
                    / 1024 ** 3,
                    2,
                ),
            },
        },
        "per_row_counts": per_row,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--s1", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument(
        "--output", default="docs/day3_scalability_report.json"
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    log_path = Path(args.log) if args.log else work_dir / "run1.out.log"

    report = build_report(
        work_dir,
        Path(args.s1),
        log_path,
        args.chunk_size,
        args.limit,
    )

    destination = REPO / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    measured = report["measured"]
    estimate = report["estimate"]

    print(f"wrote {destination.relative_to(REPO)}")
    print()
    print("MEASURED (shard)")
    print(
        f"  S1 records          "
        f"{measured['s1_records_processed']:,}"
    )
    print(
        f"  candidate rows      "
        f"{measured['s1_rows_with_candidates']:,} with candidates, "
        f"{measured['rows_with_empty_candidate_list']:,} empty"
    )
    print(
        f"  candidate pairs     "
        f"{measured['total_candidate_pairs']:,}"
    )
    print(
        f"  candidates/S1       "
        f"avg {measured['candidates_per_s1']['average']:.2f}  "
        f"med {measured['candidates_per_s1']['median']:.0f}  "
        f"p95 {measured['candidates_per_s1']['p95']:.0f}  "
        f"p99 {measured['candidates_per_s1']['p99']:.0f}  "
        f"max {measured['candidates_per_s1']['maximum']:,}"
    )
    print(
        f"  runtime             "
        f"{measured['runtime']['total_wall_seconds']:.1f}s total, "
        f"{measured['runtime']['ms_per_s1']:.2f} ms/S1"
    )
    print(
        f"  peak RAM            "
        f"{measured['peak_ram']['process_peak_rss_gb']} GB"
    )
    print(
        f"  disk                "
        f"{measured['disk']['intermediate_mb']} MB intermediate + "
        f"{measured['disk']['final_output_mb']} MB output"
    )
    print()
    print("ESTIMATE (full test, NOT measured)")
    print(
        f"  wall clock          ~{estimate['total_wall_hours']} h "
        f"(upper bound {estimate['total_wall_hours_upper_bound']} h)"
    )
    print(f"  peak RAM            ~{estimate['peak_ram_gb']} GB")
    print(f"  chunks              {estimate['chunks']['with_default_chunk_size_50000_per_target_source']} per target source")
    print(f"  candidate pairs     ~{estimate['candidate_pairs']['value']:,}")
    print(f"  disk                ~{estimate['disk']['total_gb']} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
