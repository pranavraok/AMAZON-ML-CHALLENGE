"""Compute the frozen feature table for a candidate-pair file.

Works for both training (labelled pairs) and inference (Person 2 candidates):
the same ``calculate_pair_features`` / ``add_context_features`` functions run in
both paths.

Example (training, dev pairs):
    python scripts/compute_features.py --pairs data/work/p3/labelled_pairs.parquet \
        --split train --output data/work/p3/features.parquet --workers 10

Example (benchmark on 100k pairs):
    python scripts/compute_features.py --pairs data/work/p3/labelled_pairs.parquet \
        --limit 100000 --output data/work/p3/bench.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import resource
import sys
import time

import polars as pl

from business_entity_resolution.config import load_config
from business_entity_resolution.similarity_features import (
    FEATURE_VERSION, add_context_features, add_retrieval_features,
    calculate_pair_features, model_feature_columns, prepare_records,
)
from business_entity_resolution.pair_labels import READ_KW
from business_entity_resolution.token_stats import TokenStats

RETRIEVAL_COLUMNS = ("retrieval_route", "retrieval_rank", "retrieval_score")


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def load_records(paths: list[Path], ids: set[str]) -> pl.DataFrame:
    frames = [
        pl.scan_csv(p, **READ_KW).filter(pl.col("entity_id").is_in(ids)).collect() for p in paths
    ]
    return pl.concat(frames).with_columns(pl.all().fill_null(""))


def missing_report(df: pl.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    report = {}
    n = max(df.height, 1)
    for column in columns:
        series = df[column]
        nulls = series.null_count()
        nans = int(series.is_nan().sum()) if series.dtype.is_float() else 0
        infs = int(series.is_infinite().sum()) if series.dtype.is_float() else 0
        report[column] = {
            "dtype": str(series.dtype),
            "missing_rate": round((nulls + nans) / n, 5),
            "inf_count": infs,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--token-stats", type=Path, default=Path("artifacts/token_stats.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N pairs only (benchmark)")
    parser.add_argument("--chunk-pairs", type=int, default=250_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    paths = load_config(args.config).paths
    if args.split == "train":
        s1_paths = [paths.train_source1]
        sec_paths = [paths.train_source2, paths.train_source3]
    else:
        s1_paths = [paths.test_source1]
        sec_paths = [paths.test_source2, paths.test_source3]

    pairs_long = pl.read_parquet(args.pairs)
    if args.limit:
        pairs_long = pairs_long.head(args.limit)
    has_retrieval = all(c in pairs_long.columns for c in RETRIEVAL_COLUMNS)
    passthrough = [c for c in ("label", "fold") if c in pairs_long.columns]
    pairs = (
        pairs_long.unique(["source1_entity_id", "candidate_entity_id"], keep="first",
                          maintain_order=True)
        .select("source1_entity_id", "candidate_entity_id", *passthrough)
        .with_columns(target_source=pl.col("candidate_entity_id").str.slice(0, 2))
        .select("source1_entity_id", "candidate_entity_id", "target_source", *passthrough)
        .sort("source1_entity_id", "candidate_entity_id")
    )
    token_stats = TokenStats.load(args.token_stats)

    t0 = time.time()
    s1_raw = load_records(s1_paths, set(pairs["source1_entity_id"].unique().to_list()))
    sec_raw = load_records(sec_paths, set(pairs["candidate_entity_id"].unique().to_list()))
    t_load = time.time() - t0
    t0 = time.time()
    s1_prep = prepare_records(s1_raw, workers=args.workers)
    sec_prep = prepare_records(sec_raw, workers=args.workers)
    t_prepare = time.time() - t0
    logging.info("loaded %.1fs, prepared %d+%d records in %.1fs",
                 t_load, len(s1_prep), len(sec_prep), t_prepare)

    t0 = time.time()
    parts = []
    for offset in range(0, pairs.height, args.chunk_pairs):
        chunk = pairs.slice(offset, args.chunk_pairs)
        parts.append(calculate_pair_features(chunk, s1_prep, sec_prep, token_stats))
        logging.info("features %d/%d pairs", min(offset + args.chunk_pairs, pairs.height),
                     pairs.height)
    features = pl.concat(parts)
    t_pair = time.time() - t0
    t0 = time.time()
    if has_retrieval:
        features = add_retrieval_features(features, pairs_long)
    if not args.no_context:
        features = add_context_features(features)
    t_context = time.time() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(args.output)
    feature_columns = model_feature_columns(
        include_context=not args.no_context, include_retrieval=has_retrieval
    )
    report = {
        "feature_version": FEATURE_VERSION,
        "pairs": features.height,
        "unique_s1": features["source1_entity_id"].n_unique(),
        "positives": int(features["label"].sum()) if "label" in features.columns else None,
        "n_features": len(feature_columns),
        "seconds": {
            "load": round(t_load, 1), "prepare_records": round(t_prepare, 1),
            "pair_features": round(t_pair, 1), "context": round(t_context, 1),
        },
        "pair_features_per_second": round(features.height / max(t_pair, 1e-9)),
        "end_to_end_pairs_per_second": round(
            features.height / max(t_prepare + t_pair + t_context, 1e-9)
        ),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "workers": args.workers,
        "features": missing_report(features, feature_columns),
    }
    bad = {k: v for k, v in report["features"].items() if v["inf_count"]}
    if bad:
        logging.warning("infinite values found: %s", bad)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k != "features"}
    print(json.dumps(summary, indent=2))
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
