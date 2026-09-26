"""Fit unsupervised token document frequencies for IDF features.

Uses only source records (no labels). By default all six source files (train +
test, S1/S2/S3) are used so unseen-country tokens (France) receive real IDF
values; the same artifact is loaded for training and inference.

Example:
    python scripts/fit_token_stats.py --config configs/base.json --workers 10
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import time

from business_entity_resolution.config import load_config
from business_entity_resolution.pair_labels import _chunk_stats, _init_filter, _iter_rows
from business_entity_resolution.token_stats import TokenStats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/token_stats.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    paths = load_config(args.config).paths
    files = [paths.train_source1, paths.train_source2, paths.train_source3]
    if not args.train_only:
        files += [paths.test_source1, paths.test_source2, paths.test_source3]
    start = time.time()
    name_df: Counter[str] = Counter()
    addr_df: Counter[str] = Counter()
    n_docs = 0
    with ProcessPoolExecutor(args.workers, initializer=_init_filter, initargs=(None,)) as pool:
        for path in files:
            for _, nd, ad, rows in pool.map(_chunk_stats, _iter_rows(path, 50_000)):
                name_df.update(nd)
                addr_df.update(ad)
                n_docs += rows
            logging.info("counted %s (docs so far %d)", path.name, n_docs)
    stats = TokenStats(
        n_docs=n_docs,
        name_df={t: c for t, c in name_df.items() if c >= args.min_df},
        address_df={t: c for t, c in addr_df.items() if c >= args.min_df},
    )
    stats.save(args.output)
    print(f"docs={n_docs} name_vocab={len(stats.name_df)} address_vocab={len(stats.address_df)} "
          f"seconds={time.time() - start:.0f} -> {args.output}")


if __name__ == "__main__":
    main()
