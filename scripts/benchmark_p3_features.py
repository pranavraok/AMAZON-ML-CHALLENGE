"""Final runtime + peak-memory benchmark of the frozen feature path.

Peak memory is sampled every 0.2 s over the main process *and* all worker
processes, so it reflects what the machine actually needs.

Example:
    python scripts/benchmark_p3_features.py \
        --candidates data/work/day1/candidate_pairs_training.tsv \
        --source-dir data/dev/train --prefix train --workers 10 \
        --output data/work/p3d3/benchmark_person1_candidates.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import threading
import time

import polars as pl
import psutil

from business_entity_resolution.pair_labels import READ_KW
from business_entity_resolution.similarity_features import (
    FEATURE_VERSION, FROZEN_MODEL_FEATURES, calculate_pair_features_parallel,
    validate_feature_frame,
)


class PeakMemory:
    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.peak_total = 0
        self.peak_main = 0
        self.peak_children = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        main = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                main_rss = main.memory_info().rss
                children = 0
                for child in main.children(recursive=True):
                    try:
                        children += child.memory_info().rss
                    except psutil.Error:
                        pass
                self.peak_main = max(self.peak_main, main_rss)
                self.peak_children = max(self.peak_children, children)
                self.peak_total = max(self.peak_total, main_rss + children)
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakMemory":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--prefix", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-pairs", type=int, default=50_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--s1-batch", type=int, default=0,
                        help="streaming mode: featurize this many S1 ids at a time and discard")
    args = parser.parse_args()
    mb = 2**20
    with PeakMemory() as memory:
        t0 = time.perf_counter()
        pairs = pl.read_csv(args.candidates, separator="\t", infer_schema=False) \
            .select("source1_entity_id", "candidate_entity_id", "target_source")
        if args.limit:
            pairs = pairs.head(args.limit)
        s1_raw = pl.read_csv(args.source_dir / f"{args.prefix}_source1.tsv", **READ_KW) \
            .filter(pl.col("entity_id").is_in(pairs["source1_entity_id"].unique().implode()))
        sec_raw = pl.concat([
            pl.read_csv(args.source_dir / f"{args.prefix}_source{i}.tsv", **READ_KW)
            .filter(pl.col("entity_id").is_in(pairs["candidate_entity_id"].unique().implode()))
            for i in (2, 3)])
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        batches = [pairs]
        if args.s1_batch:
            order = pairs["source1_entity_id"].unique(maintain_order=True)
            batches = (pairs.filter(pl.col("source1_entity_id").is_in(order.slice(lo, args.s1_batch).implode()))
                       for lo in range(0, order.len(), args.s1_batch))
        n_rows = 0
        largest_batch = 0
        table_mb = 0.0
        for batch in batches:
            features = calculate_pair_features_parallel(
                batch, s1_raw, sec_raw, "artifacts/token_stats.json", "artifacts/name_stats",
                processes=args.workers, chunk_pairs=args.chunk_pairs)
            validate_feature_frame(features)
            n_rows += features.height
            largest_batch = max(largest_batch, features.height)
            table_mb = max(table_mb, features.estimated_size() / mb)
            del features
        t_features = time.perf_counter() - t0
    report = {
        "feature_version": FEATURE_VERSION,
        "candidates_file": str(args.candidates),
        "pairs": n_rows,
        "unique_s1": pairs["source1_entity_id"].n_unique(),
        "s1_batch": args.s1_batch or "all at once",
        "largest_batch_pairs": largest_batch,
        "n_features": len(FROZEN_MODEL_FEATURES),
        "workers": args.workers,
        "chunk_pairs": args.chunk_pairs,
        "load_seconds": round(t_load, 1),
        "feature_seconds_incl_record_prep": round(t_features, 1),
        "pairs_per_second": round(n_rows / t_features),
        "peak_rss_total_mb": round(memory.peak_total / mb),
        "peak_rss_main_mb": round(memory.peak_main / mb),
        "peak_rss_workers_mb": round(memory.peak_children / mb),
        "largest_feature_table_in_memory_mb": round(table_mb),
        "validate_feature_frame": "PASS (no NaN/null/inf, dtypes match frozen contract)",
        "machine": {"platform": platform.platform(), "cpus": os.cpu_count(),
                    "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
