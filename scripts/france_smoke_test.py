"""Smoke-test the frozen feature path on real France test records (unseen country).

No labels exist; the test checks the path runs, dtypes are numeric, there are
no infinities, missing rates are plausible and the best candidates look sane.

Example:
    python scripts/france_smoke_test.py --sample 500 --workers 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import polars as pl

from business_entity_resolution.config import load_config
from business_entity_resolution.pair_labels import READ_KW, interim_candidates
from business_entity_resolution.similarity_features import (
    calculate_pair_features_parallel, model_feature_columns,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--country", default="France")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("data/work/p3d2/france"))
    parser.add_argument("--reference", type=Path, default=Path("data/work/p3d2/combined_features.parquet"),
                        help="training-country feature table for missing-rate comparison")
    args = parser.parse_args()
    test = load_config(args.config).paths.test_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    s1 = pl.read_csv(test / "test_source1.tsv", **READ_KW)
    country_s1 = s1.filter(pl.col("country") == args.country).with_columns(pl.all().fill_null(""))
    sample = country_s1.sample(min(args.sample, country_s1.height), seed=2026)
    sec_paths = []
    for i in (2, 3):
        sec = pl.read_csv(test / f"test_source{i}.tsv", **READ_KW).filter(pl.col("country") == args.country)
        path = out / f"test_source{i}_{args.country}.tsv"
        sec.write_csv(path, separator="\t", quote_style="never")
        sec_paths.append(path)
    candidates, _ = interim_candidates(sample, sec_paths, top_k_per_source=25, max_key_df=5000,
                                       workers=args.workers)
    sec_raw = pl.concat([pl.read_csv(p, **READ_KW) for p in sec_paths])
    pairs = candidates.select("source1_entity_id", "candidate_entity_id", "target_source") \
        .sort("source1_entity_id", "candidate_entity_id")
    start = time.perf_counter()
    features = calculate_pair_features_parallel(
        pairs, sample, sec_raw, "artifacts/token_stats.json", "artifacts/name_stats",
        processes=args.workers, chunk_pairs=5_000)
    seconds = time.perf_counter() - start

    columns = model_feature_columns(include_context=False)
    def missing(frame: pl.DataFrame, column: str) -> float:
        series = frame[column]
        nans = int(series.is_nan().sum()) if series.dtype.is_float() else 0
        return round((series.null_count() + nans) / max(frame.height, 1), 4)

    reference = pl.read_parquet(args.reference, columns=columns) if args.reference.exists() else None
    report = {
        "country": args.country,
        "test_s1_in_country": country_s1.height,
        "sampled_s1": sample.height,
        "s1_with_candidates": features["source1_entity_id"].n_unique(),
        "pairs": features.height,
        "feature_seconds": round(seconds, 1),
        "non_numeric_columns": [c for c in columns if not features[c].dtype.is_numeric()],
        "columns_with_inf": [c for c in columns if features[c].dtype.is_float()
                             and int(features[c].is_infinite().sum())],
        "missing_rate_vs_training": {
            c: {"test_country": missing(features, c),
                "training": missing(reference, c) if reference is not None else None}
            for c in columns if missing(features, c) > 0
            or (reference is not None and missing(reference, c) > 0)
        },
    }
    features.write_parquet(out / "features.parquet")
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    raw = pl.concat([sample, sec_raw]).select("entity_id", "business_name", "business_address")
    info = {r[0]: (r[1], r[2]) for r in raw.iter_rows()}
    best = features.with_columns(
        _s=pl.max_horizontal("name_token_sort_ratio", "name_skeleton_ratio")
        + pl.col("addr_token_set_ratio").fill_nan(0.0)
    ).sort("_s", descending=True).unique("source1_entity_id", keep="first", maintain_order=True)
    for r in best.head(5).iter_rows(named=True):
        print(info[r["source1_entity_id"]], "\n   ~", info[r["candidate_entity_id"]])


if __name__ == "__main__":
    main()
