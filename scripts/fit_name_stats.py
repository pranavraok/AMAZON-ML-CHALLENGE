"""Fit unsupervised S1-vs-S2/S3 name statistics (no labels) from all source files.

Example:
    python scripts/fit_name_stats.py --config configs/base.json --workers 10
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import time

from business_entity_resolution.config import load_config
from business_entity_resolution.feature_text import prepare_name
from business_entity_resolution.name_stats import core_hash, save_name_stats
from business_entity_resolution.pair_labels import _iter_rows


def _chunk(rows):
    cores: Counter[int] = Counter()
    tokens: Counter[str] = Counter()
    for _, name, _, _ in rows:
        prepared = prepare_name(name)
        cores[core_hash(prepared["name_core"])] += 1
        tokens.update(set(prepared["name_latin"].split()))
    return cores, tokens, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/name_stats"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    paths = load_config(args.config).paths
    groups = {
        "s1": [paths.train_source1, paths.test_source1],
        "secondary": [paths.train_source2, paths.train_source3,
                      paths.test_source2, paths.test_source3],
    }
    start = time.time()
    totals = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for kind, files in groups.items():
            cores: Counter[int] = Counter()
            tokens: Counter[str] = Counter()
            n = 0
            for path in files:
                for c, t, rows in pool.map(_chunk, _iter_rows(path, 50_000)):
                    cores.update(c)
                    tokens.update(t)
                    n += rows
                logging.info("%s: counted %s (records %d)", kind, path.name, n)
            totals[kind] = (n, cores, tokens)
    save_name_stats(
        args.output,
        n_s1=totals["s1"][0], n_secondary=totals["secondary"][0],
        core_s1=totals["s1"][1], core_secondary=totals["secondary"][1],
        token_s1=totals["s1"][2], token_secondary=totals["secondary"][2],
    )
    print(f"s1={totals['s1'][0]} secondary={totals['secondary'][0]} "
          f"seconds={time.time() - start:.0f} -> {args.output}")


if __name__ == "__main__":
    main()
