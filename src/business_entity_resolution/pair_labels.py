"""Labelled development pairs and hard negatives (Person 3).

Person 2 owns production candidate generation. Until their candidates are
available, this module provides an *interim* same-country blocker so that
negatives are realistic retrieved look-alikes rather than random easy pairs.
Once Person 2's long candidate table exists, call ``label_candidates`` on it
instead and keep the rest of the pipeline unchanged.

Blocking keys per record (all prefixed with the country so keys never cross
countries):
  n:<skeleton of a core name token>   (script-neutral; catches Hindi <-> Latin)
  a:<address alphabetic token, len >= 4>
  d:<address number, len >= 2>
Pairs are scored by the summed IDF of shared keys; keys with document
frequency above ``max_key_df`` are ignored as stop keys.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import logging
import math
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl

from business_entity_resolution.feature_text import prepare_address, prepare_name
from business_entity_resolution.similarity_features import name_evidence_expr

LOGGER = logging.getLogger(__name__)
READ_KW = dict(separator="\t", quote_char=None, infer_schema=False)

_KEY_FILTER: frozenset[str] | None = None


def read_source(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(path, **READ_KW).with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    )


def record_keys(name: str, address: str, country: str) -> tuple[set[str], list[str], list[str]]:
    """Blocking keys plus the name/address token lists used for IDF fitting."""

    n = prepare_name(name)
    a = prepare_address(address)
    c = country.strip().casefold()
    keys = {f"{c}|n:{s}" for s in n["name_core_skeleton_tokens"] if len(s) >= 2}
    keys |= {f"{c}|a:{t}" for t in a["address_alpha_tokens"] if len(t) >= 4}
    keys |= {f"{c}|d:{t}" for t in a["address_numbers"] if len(t) >= 2}
    return keys, list(n["name_core_tokens"]), list(a["address_tokens"])


def _init_filter(key_filter: frozenset[str] | None) -> None:
    global _KEY_FILTER
    _KEY_FILTER = key_filter


def _chunk_stats(rows: list[tuple[str, str, str, str]]) -> tuple[Counter, Counter, Counter, int]:
    """Worker: count key DF (restricted to the filter) and name/address token DF."""

    key_df: Counter[str] = Counter()
    name_df: Counter[str] = Counter()
    addr_df: Counter[str] = Counter()
    for _, name, address, country in rows:
        keys, name_tokens, address_tokens = record_keys(name, address, country)
        if _KEY_FILTER is not None:
            key_df.update(k for k in keys if k in _KEY_FILTER)
        name_df.update(set(name_tokens))
        addr_df.update(set(address_tokens))
    return key_df, name_df, addr_df, len(rows)


def _chunk_postings(rows: list[tuple[str, str, str, str]]) -> list[tuple[str, str]]:
    out = []
    for entity_id, name, address, country in rows:
        keys, _, _ = record_keys(name, address, country)
        out.extend((entity_id, k) for k in keys if k in _KEY_FILTER)
    return out


def _iter_rows(path: Path, chunk_size: int) -> Iterator[list[tuple[str, str, str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        chunk: list[tuple[str, str, str, str]] = []
        for row in reader:
            row = (row + ["", "", "", ""])[:4]
            chunk.append((row[0], row[1], row[2], row[3]))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def secondary_postings(
    paths: Iterable[Path],
    query_keys: frozenset[str],
    *,
    max_key_df: int,
    workers: int,
    chunk_size: int = 50_000,
) -> tuple[pl.DataFrame, dict[str, int], Counter, Counter, int]:
    """Two streaming passes: key DF counts, then postings for non-stop keys."""

    paths = list(paths)
    key_df: Counter[str] = Counter()
    name_df: Counter[str] = Counter()
    addr_df: Counter[str] = Counter()
    n_docs = 0
    with ProcessPoolExecutor(workers, initializer=_init_filter, initargs=(query_keys,)) as pool:
        for path in paths:
            for kd, nd, ad, rows in pool.map(_chunk_stats, _iter_rows(path, chunk_size)):
                key_df.update(kd)
                name_df.update(nd)
                addr_df.update(ad)
                n_docs += rows
            LOGGER.info("counted keys in %s", path.name)
    usable = frozenset(k for k, c in key_df.items() if c <= max_key_df)
    LOGGER.info("query keys=%d usable=%d", len(query_keys), len(usable))
    postings: list[tuple[str, str]] = []
    with ProcessPoolExecutor(workers, initializer=_init_filter, initargs=(usable,)) as pool:
        for path in paths:
            for part in pool.map(_chunk_postings, _iter_rows(path, chunk_size)):
                postings.extend(part)
            LOGGER.info("collected postings from %s (%d)", path.name, len(postings))
    frame = pl.DataFrame(postings, schema=["candidate_entity_id", "key"], orient="row")
    return frame, dict(key_df), name_df, addr_df, n_docs


def interim_candidates(
    source1: pl.DataFrame,
    secondary_paths: Iterable[Path],
    *,
    top_k_per_source: int = 25,
    max_key_df: int = 20_000,
    workers: int = 8,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Retrieve top-K S2 and S3 candidates per S1 record with an IDF key blocker."""

    s1_rows = list(zip(
        source1["entity_id"], source1["business_name"],
        source1["business_address"], source1["country"],
    ))
    s1_postings = [(eid, k) for eid, n, a, c in s1_rows for k in record_keys(n, a, c)[0]]
    s1_keys = pl.DataFrame(s1_postings, schema=["source1_entity_id", "key"], orient="row")
    query_keys = frozenset(s1_keys["key"].unique().to_list())
    sec, key_df, name_df, addr_df, n_docs = secondary_postings(
        secondary_paths, query_keys, max_key_df=max_key_df, workers=workers
    )
    idf = pl.DataFrame(
        {"key": list(key_df), "df": list(key_df.values())}
    ).with_columns(idf=(pl.lit(float(n_docs)) / pl.col("df")).log())
    scored = (
        s1_keys.join(sec, on="key")
        .join(idf.select("key", "idf"), on="key")
        .group_by("source1_entity_id", "candidate_entity_id")
        .agg(retrieval_score=pl.col("idf").sum(), shared_keys=pl.len())
        .with_columns(target_source=pl.col("candidate_entity_id").str.slice(0, 2))
        .with_columns(
            retrieval_rank=pl.col("retrieval_score")
            .rank("ordinal", descending=True)
            .over("source1_entity_id", "target_source")
        )
        .filter(pl.col("retrieval_rank") <= top_k_per_source)
        .with_columns(retrieval_route=pl.lit("interim_idf_keys"))
        .select(
            "source1_entity_id", "candidate_entity_id", "target_source",
            "retrieval_route", pl.col("retrieval_rank").cast(pl.Int32),
            "retrieval_score", "shared_keys",
        )
        .sort("source1_entity_id", "target_source", "retrieval_rank")
    )
    info = {"n_secondary_docs": n_docs, "query_keys": len(query_keys),
            "name_df": name_df, "addr_df": addr_df}
    return scored, info


def explode_ground_truth(ground_truth: pl.DataFrame) -> pl.DataFrame:
    return (
        ground_truth.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"matched_entity_ids": "candidate_entity_id"})
        .select("source1_entity_id", "candidate_entity_id")
    )


def label_candidates(candidates: pl.DataFrame, ground_truth: pl.DataFrame) -> pl.DataFrame:
    """Attach label 1/0 from ground truth. Labels are targets, never features."""

    positives = explode_ground_truth(ground_truth).with_columns(label=pl.lit(1, pl.Int8))
    return candidates.join(
        positives, on=["source1_entity_id", "candidate_entity_id"], how="left"
    ).with_columns(pl.col("label").fill_null(0).cast(pl.Int8))


def candidate_recall(labelled: pl.DataFrame, ground_truth: pl.DataFrame) -> dict[str, float]:
    positives = explode_ground_truth(ground_truth)
    found = int(labelled["label"].sum())
    total = positives.height
    per_s1 = labelled.group_by("source1_entity_id").agg(pl.len().alias("n"))
    return {
        "positive_pairs": total,
        "positives_retrieved": found,
        "pair_recall": found / total if total else math.nan,
        "mean_candidates_per_s1": float(per_s1["n"].mean()) if per_s1.height else 0.0,
        "candidate_pairs": labelled.height,
    }


# ---------------------------------------------------------------------------
# Hard negatives
# ---------------------------------------------------------------------------

HARD_NEGATIVE_TYPES = (
    "high_name_low_address",
    "high_address_low_name",
    "common_name_collision",
    "near_duplicate",
    "other_retrieved",
)


def classify_hard_negatives(features: pl.DataFrame) -> pl.DataFrame:
    """Tag each label-0 retrieved pair with a hard-negative family.

    Uses only similarity features (no labels beyond selecting label == 0).
    """

    name = name_evidence_expr()
    addr = pl.when(pl.col("addr_any_missing") == 1).then(None).otherwise(
        pl.col("addr_token_set_ratio").fill_nan(None))
    return (
        features.filter(pl.col("label") == 0)
        .with_columns(_name=name, _addr=addr)
        .with_columns(
            hard_negative_type=pl.when((pl.col("_name") >= 0.85) & (pl.col("_addr") >= 0.85))
            .then(pl.lit("near_duplicate"))
            .when((pl.col("_name") >= 0.85) & (pl.col("name_shared_max_idf") < 6.0))
            .then(pl.lit("common_name_collision"))
            .when((pl.col("_name") >= 0.85) & (pl.col("_addr").is_null() | (pl.col("_addr") < 0.6)))
            .then(pl.lit("high_name_low_address"))
            .when((pl.col("_addr") >= 0.85) & (pl.col("_name") < 0.6))
            .then(pl.lit("high_address_low_name"))
            .otherwise(pl.lit("other_retrieved"))
        )
        .drop("_name", "_addr")
    )


def select_hard_negatives(
    features: pl.DataFrame,
    *,
    score_column: str = "combo_mean",
    per_s1: int = 5,
) -> pl.DataFrame:
    """Top-scoring wrong candidates per S1 (by model score once available)."""

    tagged = classify_hard_negatives(features)
    return (
        tagged.with_columns(
            _r=pl.col(score_column).fill_nan(None).rank("ordinal", descending=True)
            .over("source1_entity_id")
        )
        .filter(pl.col("_r") <= per_s1)
        .drop("_r")
    )
