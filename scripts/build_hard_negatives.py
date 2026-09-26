"""Select realistic hard negatives from retrieved (label 0) candidate pairs.

Day 1 ranks wrong candidates by the ``combo_mean`` similarity; once Person 1's
Version 1 model exists, pass ``--score-column match_probability`` on a scored
table to mine the model's own highest-confidence mistakes.

Example:
    python scripts/build_hard_negatives.py --features data/work/p3/features.parquet \
        --output data/work/p3/hard_negatives.parquet --per-s1 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from business_entity_resolution.pair_labels import select_hard_negatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-column", default="combo_mean")
    parser.add_argument("--per-s1", type=int, default=5)
    args = parser.parse_args()
    features = pl.read_parquet(args.features)
    hard = select_hard_negatives(features, score_column=args.score_column, per_s1=args.per_s1)
    hard.write_parquet(args.output)
    counts = dict(hard.group_by("hard_negative_type").len().sort("len", descending=True).iter_rows())
    summary = {"hard_negatives": hard.height, "s1_covered": hard["source1_entity_id"].n_unique(),
               "score_column": args.score_column, "per_s1": args.per_s1, "by_type": counts}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
