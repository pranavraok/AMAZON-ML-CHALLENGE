"""Build labelled development pairs with realistic (retrieved) negatives.

Example:
    python scripts/build_training_pairs.py --config configs/base.json \
        --source1-count 50000 --top-k 25 --workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import polars as pl

from business_entity_resolution.config import load_config
from business_entity_resolution.pair_labels import (
    candidate_recall, explode_ground_truth, interim_candidates, label_candidates,
)
from business_entity_resolution.subset import _priority_sample_source1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--source1-count", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--max-key-df", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("data/work/p3"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    paths = config.paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    sampled = _priority_sample_source1(
        paths.train_source1, count=args.source1_count, seed=config.project.seed
    )
    s1 = pl.DataFrame(sampled).with_columns(pl.all().fill_null(""))
    s1_ids = set(s1["entity_id"].to_list())
    gt = pl.read_csv(
        paths.train_ground_truth, separator="\t", quote_char=None, infer_schema=False
    ).filter(pl.col("source1_entity_id").is_in(s1_ids))

    candidates, info = interim_candidates(
        s1, [paths.train_source2, paths.train_source3],
        top_k_per_source=args.top_k, max_key_df=args.max_key_df, workers=args.workers,
    )
    labelled = label_candidates(candidates, gt)
    stats = candidate_recall(labelled, gt)
    positives = explode_ground_truth(gt)
    missed = positives.join(labelled, on=["source1_entity_id", "candidate_entity_id"], how="anti")

    s1.write_parquet(args.output_dir / "dev_source1.parquet")
    gt.write_parquet(args.output_dir / "dev_ground_truth.parquet")
    labelled.write_parquet(args.output_dir / "labelled_pairs.parquet")
    missed.write_parquet(args.output_dir / "missed_positives.parquet")
    stats.update(
        source1_count=s1.height,
        singletons=int((gt["matched_entity_ids"].fill_null("") == "").sum()),
        top_k_per_source=args.top_k,
        max_key_df=args.max_key_df,
        n_secondary_docs=info["n_secondary_docs"],
        seconds=round(time.time() - start, 1),
    )
    (args.output_dir / "labelled_pairs_manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
