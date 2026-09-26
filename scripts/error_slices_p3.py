"""Split validation errors into candidate misses vs model mistakes and show examples.

Example:
    python scripts/error_slices_p3.py --scores data/work/p3d2/scores/p3_separate.tsv --threshold 0.61
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from business_entity_resolution.config import load_config
from business_entity_resolution.pair_labels import explode_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--features", type=Path, default=Path("data/work/p3d2/combined_features.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/work/p3d2/error_slices.parquet"))
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    paths = load_config(args.config).paths
    train = paths.development_dir / "train"
    read = dict(separator="\t", quote_char=None, infer_schema=False)
    records = pl.concat([pl.read_csv(train / f"train_source{i}.tsv", **read) for i in (1, 2, 3)])
    scores = pl.read_csv(args.scores, separator="\t")
    valid_ids = scores["source1_entity_id"].unique()
    gt = pl.read_csv(train / "train_ground_truth.tsv", **read)
    truth = explode_ground_truth(gt).filter(pl.col("source1_entity_id").is_in(valid_ids.implode()))
    truth = truth.with_columns(is_true=pl.lit(True))
    joined = scores.join(truth, on=["source1_entity_id", "candidate_entity_id"], how="full",
                         coalesce=True).with_columns(pl.col("is_true").fill_null(False))
    joined = joined.with_columns(
        error_type=pl.when(pl.col("match_probability").is_null()).then(pl.lit("fn_candidate_miss"))
        .when(pl.col("is_true") & (pl.col("match_probability") < args.threshold)).then(pl.lit("fn_model"))
        .when(~pl.col("is_true") & (pl.col("match_probability") >= args.threshold)).then(pl.lit("fp_model"))
        .otherwise(pl.lit("correct"))
    )
    errors = joined.filter(pl.col("error_type") != "correct")
    feats = pl.read_parquet(args.features).drop("target_source", "label", "fold")
    errors = errors.join(feats, on=["source1_entity_id", "candidate_entity_id"], how="left")
    errors.write_parquet(args.output)
    counts = dict(errors.group_by("error_type").len().iter_rows())
    print(json.dumps(counts))
    info = dict(zip(records["entity_id"], zip(records["business_name"], records["business_address"])))
    for kind in ("fp_model", "fn_model"):
        print(f"\n===== {kind}")
        subset = errors.filter(pl.col("error_type") == kind).sort("match_probability", descending=kind == "fp_model")
        for r in subset.head(args.examples).iter_rows(named=True):
            print(f"p={r['match_probability']:.3f} name_ev={max(r['name_token_sort_ratio'], r['name_skeleton_ratio']):.2f} "
                  f"addr_set={r['addr_token_set_ratio']} num_j={r['addr_num_jaccard']} ctx_rank={r['ctx_combo_rank']} rev_n={r['ctx_rev_n_s1']}")
            print("   S1:", info[r["source1_entity_id"]])
            print("   C :", info[r["candidate_entity_id"]])


if __name__ == "__main__":
    main()
