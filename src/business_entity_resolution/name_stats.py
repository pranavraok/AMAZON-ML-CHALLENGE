"""Unsupervised name statistics split by source kind (S1 vs S2/S3).

Fitted once from the raw source files only (no labels) and saved as one
artifact that training and inference both load:

* ``core_counts``: how many S1 records and how many S2/S3 records carry each
  exact core name (keyed by a stable 64-bit hash of the core string). A name
  shared by many businesses is ambiguous evidence, especially when an address
  is missing.
* ``token_counts``: per-token document frequency in S1 names vs S2/S3 names.
  Noise words (``services``, ``holdings``, ``www`` ...) are injected into
  secondary-source names, so their S2/S3-vs-S1 log frequency ratio is high.
  The ratio is used as a continuous feature on *unmatched* tokens only (a hard
  vocabulary would also capture OCR/transliteration variants of real words).

The artifact is written by ``scripts/fit_name_stats.py`` and the checksum of
every file is recorded in ``configs/p3_features_frozen.json``.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import polars as pl

NAME_STATS_VERSION = 1
NOISE_MIN_SECONDARY_DF = 50

NAME_FREQUENCY_SCHEMA: dict[str, pl.DataType] = {
    "name_core_freq_s1_a": pl.Float32,
    "name_core_freq_sec_a": pl.Float32,
    "name_core_freq_s1_b": pl.Float32,
    "name_core_freq_sec_b": pl.Float32,
}
HASH_COLUMNS = ("_core_hash_a", "_core_hash_b")


def core_hash(core: str) -> int:
    digest = hashlib.blake2b(core.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def load_token_noise(directory: str | Path) -> dict[str, float]:
    """log(S2/S3 rate / smoothed S1 rate) for tokens seen >= 50 times in S2/S3.

    Positive values mean the word is over-represented in secondary sources,
    which is typical of injected noise words ("services", "holdings", "www").
    Tokens absent from the table get 0.0 (neutral).
    """

    directory = Path(directory)
    meta = pl.read_parquet(directory / "meta.parquet").row(0, named=True)
    if meta["version"] != NAME_STATS_VERSION:
        raise ValueError(f"unsupported name-stats version in {directory}")
    tokens = pl.read_parquet(directory / "token_counts.parquet").filter(
        pl.col("df_secondary") >= NOISE_MIN_SECONDARY_DF
    ).with_columns(
        log_ratio=((pl.col("df_secondary") / meta["n_secondary"])
                   / ((pl.col("df_s1") + 1) / meta["n_s1"])).log()
    )
    return dict(zip(tokens["token"].to_list(), tokens["log_ratio"].cast(pl.Float64).to_list()))


def add_name_frequency_features(features: pl.DataFrame, directory: str | Path) -> pl.DataFrame:
    """Join exact core-name frequencies (log1p counts) and drop the hash helpers.

    Counts come from all six source files, so the same values are used for
    training and inference. The join keeps the input row order.
    """

    cores = pl.scan_parquet(Path(directory) / "core_counts.parquet")
    out = features
    for side in ("a", "b"):
        key = f"_core_hash_{side}"
        wanted = out.select(pl.col(key).unique().alias("hash")).lazy()
        counts = (
            cores.join(wanted, on="hash", how="semi")
            .select(
                pl.col("hash").alias(key),
                pl.col("n_s1").log1p().cast(pl.Float32).alias(f"name_core_freq_s1_{side}"),
                pl.col("n_secondary").log1p().cast(pl.Float32).alias(f"name_core_freq_sec_{side}"),
            )
            .collect()
        )
        out = out.join(counts, on=key, how="left", maintain_order="left").with_columns(
            pl.col(f"name_core_freq_s1_{side}").fill_null(0.0),
            pl.col(f"name_core_freq_sec_{side}").fill_null(0.0),
        )
    return out.drop(*HASH_COLUMNS)


def save_name_stats(
    directory: str | Path,
    *,
    n_s1: int,
    n_secondary: int,
    core_s1: Counter,
    core_secondary: Counter,
    token_s1: Counter,
    token_secondary: Counter,
    min_token_df: int = 5,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = np.fromiter(set(core_s1) | set(core_secondary), dtype=np.int64)
    keys.sort()
    pl.DataFrame({
        "hash": keys,
        "n_s1": np.fromiter((core_s1.get(int(k), 0) for k in keys), np.int32, len(keys)),
        "n_secondary": np.fromiter((core_secondary.get(int(k), 0) for k in keys), np.int32, len(keys)),
    }).write_parquet(directory / "core_counts.parquet")
    vocab = sorted(t for t in set(token_s1) | set(token_secondary)
                   if token_s1.get(t, 0) + token_secondary.get(t, 0) >= min_token_df)
    pl.DataFrame({
        "token": vocab,
        "df_s1": [token_s1.get(t, 0) for t in vocab],
        "df_secondary": [token_secondary.get(t, 0) for t in vocab],
    }).write_parquet(directory / "token_counts.parquet")
    pl.DataFrame({"version": [NAME_STATS_VERSION], "n_s1": [n_s1],
                  "n_secondary": [n_secondary]}).write_parquet(directory / "meta.parquet")
