"""Phase 3-6 driver: chunked candidate generation and final validation.

Usage:

    python -m business_entity_resolution.run_chunked_generation \
        --s1 data/test/test_source1.tsv \
        --target-s2 data/test/test_source2.tsv \
        --target-s3 data/test/test_source3.tsv \
        --work-dir data/work/chunked \
        --output output/candidate_pairs.tsv

Re-running after an interruption skips completed chunks. S2 and S3 are
processed independently and merged only by the finalize step.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

from .chunked_generation import (
    DEFAULT_CHUNK_SIZE,
    PeakMemoryMonitor,
    finalize,
    load_partition,
    partition_targets_by_country,
    run_chunked,
)

TARGETS = (("S2", "--target-s2"), ("S3", "--target-s3"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restartable chunked candidate generation."
    )

    parser.add_argument("--s1", required=True, help="Source 1 TSV.")
    parser.add_argument(
        "--target-s2", default=None, help="Source 2 TSV."
    )
    parser.add_argument(
        "--target-s3", default=None, help="Source 3 TSV."
    )
    parser.add_argument(
        "--work-dir",
        default="data/work/chunked",
        help="Directory for partitions, chunk outputs and manifests.",
    )
    parser.add_argument(
        "--output",
        default="output/candidate_pairs.tsv",
        help="Final deliverable path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"S1 records per chunk (default {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N Source 1 records.",
    )
    parser.add_argument(
        "--skip-partition",
        action="store_true",
        help="Reuse an existing country partition.",
    )
    parser.add_argument(
        "--index-cache-dir",
        default=None,
        help=(
            "Directory holding the persisted target index. The index is "
            "written there after the first build and reused afterwards, so "
            "a re-run or a resumed run does not rebuild it."
        ),
    )
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help=(
            "Load the target index from --index-cache-dir when it is "
            "already there instead of rebuilding it."
        ),
    )
    parser.add_argument(
        "--cache-tfidf",
        action="store_true",
        help=(
            "Also cache the per-country TF-IDF retrievers under "
            "--index-cache-dir. Saves the vectoriser build on re-runs at "
            "the cost of disk space."
        ),
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Merge existing chunk outputs without generating.",
    )
    parser.add_argument(
        "--tracemalloc",
        action="store_true",
        help=(
            "Also track Python allocations. This roughly triples "
            "runtime, and it under-reports numpy memory, so peak RSS is "
            "reported either way and this is off by default."
        ),
    )

    args = parser.parse_args()

    s1_path = Path(args.s1)
    work_root = Path(args.work_dir)
    output_path = Path(args.output)

    sources = {
        "S2": args.target_s2,
        "S3": args.target_s3,
    }

    started = time.perf_counter()

    if args.tracemalloc:
        tracemalloc.start()

    monitor = PeakMemoryMonitor()
    monitor.__enter__()

    partitions: dict[str, dict[str, Path]] = {}
    manifests = []
    work_dirs = []

    for name, flag in TARGETS:

        path = sources[name]
        if not path:
            continue

        target_path = Path(path)
        work_dir = work_root / name

        print(f"\n{'=' * 60}")
        print(f"{name}: {target_path}")
        print("=" * 60)

        partition_dir = work_root / f"partition_{name}"

        if args.finalize_only:
            from .chunked_generation import Manifest

            manifests.append(Manifest.load(
                work_dir / "manifest.json", name
            ))
            work_dirs.append(work_dir)
            continue

        if not args.skip_partition or not partition_dir.exists():
            print(f"  partitioning {target_path} by country...")
            partition_started = time.perf_counter()
            partitions[name] = partition_targets_by_country(
                target_path, partition_dir
            )
            print(
                f"  partitioned in "
                f"{time.perf_counter() - partition_started:.1f}s"
            )
        else:
            print(f"  reusing partition {partition_dir}")
            partitions[name] = load_partition(partition_dir)

        manifests.append(
            run_chunked(
                s1_path,
                partitions[name],
                name,
                work_dir,
                chunk_size=args.chunk_size,
                limit=args.limit,
                index_dir=(
                    Path(args.index_cache_dir) / name
                    if args.index_cache_dir
                    else None
                ),
                reuse_index=args.reuse_index,
                cache_tfidf=args.cache_tfidf,
            )
        )
        work_dirs.append(work_dir)

    if not manifests:
        raise SystemExit("no target sources given")

    if not args.finalize_only:

        print(f"\n{'=' * 60}")
        print("FINALIZE")
        print("=" * 60)

        summary = finalize(
            s1_path, manifests, work_dirs, output_path,
            limit=args.limit,
        )

        print(f"  wrote {summary['path']}")
        print(f"  rows          {summary['rows']:,}")
        print(f"  candidate pairs {summary['candidate_pairs']:,}")
        print(f"  empty rows    {summary['empty_rows']:,}")

    current, peak = tracemalloc.get_traced_memory()
    monitor.__exit__()

    print()
    print(f"total wall clock {time.perf_counter() - started:.1f}s")
    if args.tracemalloc:
        print(f"python peak (tracemalloc) {peak / 1024 ** 3:.2f} GB")
    print(f"process peak RSS {monitor.peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
