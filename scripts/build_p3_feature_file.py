"""Write Person 3's frozen features for a Person 1/2 candidate TSV.

Input is the shared candidate-pair contract (six columns, one row per pair).
Output keeps Person 1's identifier columns (plus ``label``/``fold`` when truth
and folds are given) followed by the 79 frozen feature columns, in candidate
order. The same command serves training (with truth + folds) and test
inference (without).

Examples:
    # training / validation rows (Person 1 dev pipeline)
    python scripts/build_p3_feature_file.py \
        --candidates data/work/day1/candidate_pairs_training.tsv \
        --source-dir data/dev/train --prefix train \
        --truth data/dev/train/train_ground_truth.tsv --folds data/work/day1/folds.tsv \
        --output data/work/p3d2/p3_features_training.parquet

    # test inference
    python scripts/build_p3_feature_file.py \
        --candidates data/work/test/candidate_pairs_raw.tsv \
        --source-dir ../student_resource/dataset/test --prefix test \
        --output data/work/test/p3_features.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import polars as pl
import psutil

from business_entity_resolution.pair_labels import READ_KW, explode_ground_truth
from business_entity_resolution.schemas import CANDIDATE_PAIR_COLUMNS
from business_entity_resolution.similarity_features import (
    FEATURE_VERSION, FROZEN_MODEL_FEATURES, calculate_pair_features_parallel,
    validate_feature_frame,
)

KEYS = ["source1_entity_id", "candidate_entity_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--prefix", choices=("train", "test"), required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-stats", type=Path, default=Path("artifacts/token_stats.json"))
    parser.add_argument("--name-stats", type=Path, default=Path("artifacts/name_stats"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-pairs", type=int, default=50_000)
    parser.add_argument("--tsv", action="store_true", help="also write a TSV next to the parquet")
    parser.add_argument("--s1-batch", type=int, default=0,
                        help="stream in batches of this many S1 ids; --output becomes a directory")
    args = parser.parse_args()

    candidates = pl.read_csv(args.candidates, separator="\t", infer_schema=False)
    if tuple(candidates.columns) != CANDIDATE_PAIR_COLUMNS:
        raise ValueError(f"{args.candidates} columns {candidates.columns} != {CANDIDATE_PAIR_COLUMNS}")
    if candidates.select(KEYS).is_duplicated().any():
        raise ValueError("candidate file must contain one row per unique pair")
    bad_source = candidates.filter(
        pl.col("target_source").is_null()
        | (pl.col("target_source") != pl.col("candidate_entity_id").str.slice(0, 2))
        | ~pl.col("target_source").is_in(["S2", "S3"])
    )
    if bad_source.height:
        raise ValueError(
            f"{bad_source.height} candidate rows have target_source that is not the S2/S3 "
            f"prefix of candidate_entity_id (e.g. {bad_source.row(0)})"
        )
    pairs = candidates.select(*KEYS, "target_source")
    del candidates

    s1_raw = pl.read_csv(args.source_dir / f"{args.prefix}_source1.tsv", **READ_KW) \
        .filter(pl.col("entity_id").is_in(pairs["source1_entity_id"].unique().implode()))
    sec_raw = pl.concat([
        pl.read_csv(args.source_dir / f"{args.prefix}_source{i}.tsv", **READ_KW)
        .filter(pl.col("entity_id").is_in(pairs["candidate_entity_id"].unique().implode()))
        for i in (2, 3)
    ])
    positives = folds = None
    if args.truth is not None:
        positives = explode_ground_truth(pl.read_csv(args.truth, separator="\t", infer_schema=False)) \
            .with_columns(label=pl.lit(1, pl.Int8))
    if args.folds is not None:
        folds = pl.read_csv(args.folds, separator="\t", infer_schema=False) \
            .with_columns(pl.col("fold").cast(pl.Int8))
    extra = [c for c, v in (("label", positives), ("fold", folds)) if v is not None]

    def featurize(batch: pl.DataFrame) -> pl.DataFrame:
        s1_ids = batch["source1_entity_id"].unique().implode()
        sec_ids = batch["candidate_entity_id"].unique().implode()
        features = calculate_pair_features_parallel(
            batch, s1_raw.filter(pl.col("entity_id").is_in(s1_ids)),
            sec_raw.filter(pl.col("entity_id").is_in(sec_ids)),
            args.token_stats, args.name_stats,
            processes=args.workers, chunk_pairs=args.chunk_pairs,
        )
        validate_feature_frame(features)
        if positives is not None:
            features = features.join(positives, on=KEYS, how="left", maintain_order="left") \
                .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        if folds is not None:
            features = features.join(folds, on="source1_entity_id", how="left", maintain_order="left")
            if features["fold"].null_count():
                raise ValueError("some S1 ids have no fold assignment")
        return features.select(*KEYS, "target_source", *extra, *FROZEN_MODEL_FEATURES)

    start = time.perf_counter()
    peak_rss = 0.0
    rows = 0
    if args.s1_batch:
        # Streaming mode: bounded memory for very large candidate sets. The
        # output is a directory of parquet parts; read it with
        # pl.scan_parquet(output / "*.parquet").
        args.output.mkdir(parents=True, exist_ok=True)
        s1_order = pairs["source1_entity_id"].unique(maintain_order=True)
        for part, lo in enumerate(range(0, s1_order.len(), args.s1_batch)):
            ids = s1_order.slice(lo, args.s1_batch).implode()
            batch = featurize(pairs.filter(pl.col("source1_entity_id").is_in(ids)))
            batch.write_parquet(args.output / f"part-{part:05d}.parquet")
            rows += batch.height
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss / 2**20)
            print(f"part {part}: {batch.height} rows ({rows} total, "
                  f"{time.perf_counter() - start:.0f}s)", flush=True)
            del batch
        report_path = args.output / "report.json"
        unique_s1 = s1_order.len()
    else:
        features = featurize(pairs)
        rows = features.height
        unique_s1 = features["source1_entity_id"].n_unique()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        features.write_parquet(args.output)
        if args.tsv:
            features.write_csv(args.output.with_suffix(".tsv"), separator="\t")
        peak_rss = psutil.Process().memory_info().rss / 2**20
        report_path = args.output.with_suffix(".report.json")
    seconds = time.perf_counter() - start
    report = {
        "feature_version": FEATURE_VERSION,
        "rows": rows,
        "unique_s1": unique_s1,
        "n_features": len(FROZEN_MODEL_FEATURES),
        "seconds": round(seconds, 1),
        "pairs_per_second": round(rows / max(seconds, 1e-9)),
        "workers": args.workers,
        "s1_batch": args.s1_batch,
        "main_process_rss_mb": round(peak_rss, 1),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
