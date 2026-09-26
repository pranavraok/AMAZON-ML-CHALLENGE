"""Pair-level similarity features (Person 3 module).

Public entry points
-------------------
``prepare_records(df)``
    Build record-level text views once per unique record.
``calculate_pair_features(pairs, source1, secondary, token_stats)``
    Deterministic batch function returning one numeric row per candidate pair.
    Training and inference must both call this function.
``add_context_features(features)``
    Group-level context (rank / gap within an S1 group and within a
    candidate's competing S1 set). Run it once on the *complete* feature table,
    after all chunks are concatenated, because it needs whole groups.
``add_retrieval_features(features, candidates_long)``
    Optional aggregation of Person 2's retrieval route / rank / score columns.

Missing evidence contract: "not applicable" values (a missing address, an
address without numbers, a name whose tokens are all matched) are written as an
out-of-range sentinel, never as a plausible similarity and never as NaN:
-1.0 for features whose valid range is >= 0, and -5.0 for the noise log-ratio
features (valid range is bounded below by log(N_S1 / N_S2S3) ~ -1.65). The
``addr_missing_*`` flags and number counts make every sentinel explicit.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import os
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from business_entity_resolution.feature_text import prepare_address, prepare_name
from business_entity_resolution.name_stats import (
    NAME_FREQUENCY_SCHEMA, add_name_frequency_features, core_hash, load_token_noise,
)
from business_entity_resolution.token_stats import TokenStats

FEATURE_VERSION = "p3-v2.1"

ID_COLUMNS = ("source1_entity_id", "candidate_entity_id", "target_source")

# Ordered feature schema: name -> polars dtype. The order is part of the
# contract; do not reorder or rename after the schema is frozen.
PAIR_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    # --- name: exact / fuzzy ---------------------------------------------
    "name_exact_latin": pl.Int8,
    "name_exact_core": pl.Int8,
    "name_ratio": pl.Float32,
    "name_token_sort_ratio": pl.Float32,
    "name_token_set_ratio": pl.Float32,
    "name_partial_ratio": pl.Float32,
    "name_wratio_latin": pl.Float32,
    "name_jaro_winkler": pl.Float32,
    "name_levenshtein_sim": pl.Float32,
    "name_unicode_ratio": pl.Float32,
    "name_skeleton_ratio": pl.Float32,
    "name_skeleton_token_set": pl.Float32,
    "name_nospace_ratio": pl.Float32,
    "name_nospace_partial": pl.Float32,
    "name_alias_best_ratio": pl.Float32,
    # --- name: token / IDF -----------------------------------------------
    "name_token_jaccard": pl.Float32,
    "name_token_containment": pl.Float32,
    "name_idf_jaccard": pl.Float32,
    "name_soft_idf_min": pl.Float32,
    "name_soft_idf_max": pl.Float32,
    "name_unmatched_max_idf": pl.Float32,
    "name_shared_max_idf": pl.Float32,
    "name_first_token_equal": pl.Int8,
    "name_last_token_equal": pl.Int8,
    "name_prefix3_equal": pl.Int8,
    "name_acronym_match": pl.Int8,
    "name_len_a": pl.Int16,
    "name_len_b": pl.Int16,
    "name_ntok_a": pl.Int8,
    "name_ntok_b": pl.Int8,
    "name_len_ratio": pl.Float32,
    "name_legal_conflict": pl.Int8,
    "name_legal_both_present": pl.Int8,
    "name_non_latin_a": pl.Int8,
    "name_non_latin_b": pl.Int8,
    "name_script_mismatch": pl.Int8,
    "name_a_unmatched_max_noise": pl.Float32,
    "name_b_unmatched_max_noise": pl.Float32,
    # --- address ----------------------------------------------------------
    "addr_missing_a": pl.Int8,
    "addr_missing_b": pl.Int8,
    "addr_any_missing": pl.Int8,
    "addr_ratio": pl.Float32,
    "addr_token_sort_ratio": pl.Float32,
    "addr_token_set_ratio": pl.Float32,
    "addr_partial_ratio": pl.Float32,
    "addr_skeleton_token_set": pl.Float32,
    "addr_token_jaccard": pl.Float32,
    "addr_token_containment": pl.Float32,
    "addr_alpha_containment": pl.Float32,
    "addr_idf_jaccard": pl.Float32,
    "addr_soft_idf_min": pl.Float32,
    "addr_soft_idf_max": pl.Float32,
    "addr_unmatched_max_idf": pl.Float32,
    "addr_ntok_a": pl.Int16,
    "addr_ntok_b": pl.Int16,
    # --- address numbers --------------------------------------------------
    "addr_num_count_a": pl.Int8,
    "addr_num_count_b": pl.Int8,
    "addr_num_jaccard": pl.Float32,
    "addr_num_shared": pl.Int8,
    "addr_num_any_shared": pl.Int8,
    "addr_num_suffix_match": pl.Int8,
    "addr_num_conflict": pl.Int8,
    "addr_longest_num_equal": pl.Int8,
    "addr_primary_equal": pl.Float32,
    "addr_primary_cross": pl.Float32,
    "addr_num_only_a": pl.Int8,
    "addr_num_only_b": pl.Int8,
    # --- combined evidence ------------------------------------------------
    "combo_min": pl.Float32,
    "combo_max": pl.Float32,
    "combo_product": pl.Float32,
    "combo_mean": pl.Float32,
    "combo_name_high_addr_low": pl.Int8,
    "combo_addr_high_name_low": pl.Int8,
    "same_country": pl.Int8,
    "target_is_s3": pl.Int8,
}
PAIR_FEATURE_COLUMNS = tuple(PAIR_FEATURE_SCHEMA)

CONTEXT_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "ctx_n_candidates": pl.Int16,
    "ctx_n_candidates_source": pl.Int16,
    "ctx_name_rank": pl.Int16,
    "ctx_name_gap": pl.Float32,
    "ctx_addr_rank": pl.Int16,
    "ctx_addr_gap": pl.Float32,
    "ctx_combo_rank": pl.Int16,
    "ctx_combo_gap": pl.Float32,
    "ctx_combo_second_gap": pl.Float32,
    "ctx_rev_n_s1": pl.Int16,
    "ctx_rev_combo_rank": pl.Int16,
    "ctx_rev_combo_gap": pl.Float32,
}
CONTEXT_FEATURE_COLUMNS = tuple(CONTEXT_FEATURE_SCHEMA)

RETRIEVAL_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "retr_min_rank": pl.Int16,
    "retr_n_routes": pl.Int8,
    "retr_max_score": pl.Float32,
    "retr_score_gap": pl.Float32,
}
RETRIEVAL_FEATURE_COLUMNS = tuple(RETRIEVAL_FEATURE_SCHEMA)

NA_SENTINEL = -1.0
NA_SENTINEL_OVERRIDES = {
    "name_a_unmatched_max_noise": -5.0,
    "name_b_unmatched_max_noise": -5.0,
}

_SOFT_TOKEN_THRESHOLD = 80.0
_NAME_KEYS = (
    "name_unicode", "name_latin", "name_core", "name_core_sorted", "name_nospace",
    "name_core_skeleton", "name_acronym", "name_alias_parts", "name_legal_families",
    "name_core_tokens", "name_core_skeleton_tokens", "name_non_latin",
)
_ADDRESS_KEYS = (
    "address_latin", "address_skeleton", "address_tokens", "address_alpha_tokens",
    "address_numbers", "address_longest_number", "address_primary_number", "address_missing",
)


# ---------------------------------------------------------------------------
# Record preparation
# ---------------------------------------------------------------------------

def _prepare_chunk(rows: Sequence[tuple[str, str, str]]) -> list[tuple]:
    out = []
    for name, address, country in rows:
        n = prepare_name(name)
        a = prepare_address(address)
        out.append(
            tuple(n[k] for k in _NAME_KEYS) + tuple(a[k] for k in _ADDRESS_KEYS) + (country,)
        )
    return out


class PreparedTable:
    """Columnar record-level views keyed by entity_id."""

    columns = _NAME_KEYS + _ADDRESS_KEYS + ("country",)

    def __init__(self, entity_ids: list[str], rows: list[tuple]):
        self.entity_ids = entity_ids
        self.index = {eid: i for i, eid in enumerate(entity_ids)}
        for position, column in enumerate(self.columns):
            setattr(self, column, [row[position] for row in rows])

    def __len__(self) -> int:
        return len(self.entity_ids)


def prepare_records(df: pl.DataFrame, *, workers: int = 1, chunk_size: int = 50_000) -> PreparedTable:
    """Prepare unique records from a frame with the shared source columns."""

    df = df.unique(subset="entity_id", keep="first", maintain_order=True)
    ids = df["entity_id"].to_list()
    rows = list(
        zip(
            df["business_name"].fill_null("").to_list(),
            df["business_address"].fill_null("").to_list(),
            df["country"].fill_null("").cast(pl.Utf8).str.strip_chars().to_list(),
        )
    )
    if workers <= 1 or len(rows) < 2 * chunk_size:
        prepared = _prepare_chunk(rows)
    else:
        chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
        prepared = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(_prepare_chunk, chunks):
                prepared.extend(part)
    return PreparedTable(ids, prepared)


# ---------------------------------------------------------------------------
# Pair helpers
# ---------------------------------------------------------------------------

def _cpdist(a: list[str], b: list[str], scorer, workers: int) -> np.ndarray:
    if not a:
        return np.zeros(0, dtype=np.float32)
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _soft_idf(tokens_a, tokens_b, set_b, idf, skel_a=None, skel_b=None, noise=None):
    """Directional soft-IDF coverage of A by B, plus max IDF of unmatched A tokens.

    A token counts as covered when an exact, fuzzy (ratio >= 80) or, when
    skeletons are given, phonetic-skeleton match exists in B. With ``noise``
    (token -> S2/S3-vs-S1 log ratio) a third value is returned: the largest
    noise ratio among unmatched A tokens (NaN when every token is matched).
    """

    total = 0.0
    covered = 0.0
    unmatched = 0.0
    noise_max = -math.inf
    skel_set_b = set(skel_b) if skel_b else ()
    for position, token in enumerate(tokens_a):
        weight = idf(token)
        total += weight
        if token in set_b:
            covered += weight
            continue
        best = 0.0
        if skel_a is not None and len(skel_a[position]) >= 2 and skel_a[position] in skel_set_b:
            best = 90.0
        for other in tokens_b:
            score = fuzz.ratio(token, other)
            if score > best:
                best = score
        if best >= _SOFT_TOKEN_THRESHOLD:
            covered += weight * best / 100.0
            continue
        if weight > unmatched:
            unmatched = weight
        if noise is not None:
            noise_max = max(noise_max, noise.get(token, 0.0))
    if noise is not None:
        noise_value = math.nan if noise_max == -math.inf else noise_max
        if total == 0.0:
            return math.nan, math.nan, noise_value
        return covered / total, unmatched, noise_value
    if total == 0.0:
        return math.nan, math.nan
    return covered / total, unmatched


def _idf_jaccard(set_a, set_b, idf) -> tuple[float, float]:
    union = set_a | set_b
    if not union:
        return math.nan, math.nan
    shared = set_a & set_b
    num = sum(idf(t) for t in shared)
    den = sum(idf(t) for t in union)
    shared_max = max((idf(t) for t in shared), default=0.0)
    return num / den, shared_max


def _number_features(na: frozenset, nb: frozenset, la: str, lb: str) -> tuple:
    count_a, count_b = len(na), len(nb)
    if not na or not nb:
        return count_a, count_b, math.nan, 0, 0, 0, 0, 0
    shared = na & nb
    jaccard = len(shared) / len(na | nb)
    suffix = 0
    if not shared:
        for x in na:
            for y in nb:
                if len(x) >= 2 and len(y) >= 2 and (x.endswith(y) or y.endswith(x)):
                    suffix = 1
                    break
            if suffix:
                break
    any_shared = int(bool(shared))
    conflict = int(not shared and not suffix)
    return (
        count_a, count_b, jaccard, min(len(shared), 127), any_shared, suffix, conflict,
        int(la == lb),
    )


# ---------------------------------------------------------------------------
# Main batch function
# ---------------------------------------------------------------------------

def calculate_pair_features(
    pairs: pl.DataFrame,
    source1: PreparedTable | pl.DataFrame,
    secondary: PreparedTable | pl.DataFrame,
    token_stats: TokenStats,
    *,
    name_noise: Mapping[str, float],
    workers: int = -1,
    prepare_workers: int = 1,
) -> pl.DataFrame:
    """Compute the frozen pair-feature schema for every row of ``pairs``.

    ``name_noise`` comes from ``name_stats.load_token_noise``. The output also
    carries two Int64 helper columns (``_core_hash_a/_b``) that
    ``name_stats.add_name_frequency_features`` consumes and drops.

    ``pairs`` needs ``source1_entity_id`` and ``candidate_entity_id``; a
    ``target_source`` column is derived from the ID prefix when absent. Extra
    columns (``label``, ``fold``, retrieval fields) are passed through unchanged.
    Output rows keep the input order.
    """

    if isinstance(source1, pl.DataFrame):
        source1 = prepare_records(source1, workers=prepare_workers)
    if isinstance(secondary, pl.DataFrame):
        secondary = prepare_records(secondary, workers=prepare_workers)
    if "target_source" not in pairs.columns:
        pairs = pairs.with_columns(
            pl.col("candidate_entity_id").str.slice(0, 2).alias("target_source")
        )

    s1_ids = pairs["source1_entity_id"].to_list()
    sec_ids = pairs["candidate_entity_id"].to_list()
    try:
        ia = [source1.index[x] for x in s1_ids]
        ib = [secondary.index[x] for x in sec_ids]
    except KeyError as error:
        raise KeyError(f"pair references an unprepared record: {error}") from None
    n = len(ia)

    def gather(table: PreparedTable, column: str, idx: list[int]) -> list:
        values = getattr(table, column)
        return [values[i] for i in idx]

    A = {c: gather(source1, c, ia) for c in PreparedTable.columns}
    B = {c: gather(secondary, c, ib) for c in PreparedTable.columns}
    out: dict[str, np.ndarray] = {}

    def sim(col_a: str, col_b: str, scorer) -> np.ndarray:
        return _cpdist(A[col_a], B[col_b], scorer, workers) / 100.0

    # ---- name: vectorised string scorers --------------------------------
    out["name_exact_latin"] = np.fromiter(
        (a == b and a != "" for a, b in zip(A["name_latin"], B["name_latin"])), np.int8, n
    )
    out["name_exact_core"] = np.fromiter(
        (a == b and a != "" for a, b in zip(A["name_core"], B["name_core"])), np.int8, n
    )
    out["name_ratio"] = sim("name_core", "name_core", fuzz.ratio)
    out["name_token_sort_ratio"] = sim("name_core", "name_core", fuzz.token_sort_ratio)
    out["name_token_set_ratio"] = sim("name_core", "name_core", fuzz.token_set_ratio)
    out["name_partial_ratio"] = sim("name_core", "name_core", fuzz.partial_ratio)
    out["name_wratio_latin"] = sim("name_latin", "name_latin", fuzz.WRatio)
    out["name_jaro_winkler"] = _cpdist(
        A["name_core_sorted"], B["name_core_sorted"], JaroWinkler.normalized_similarity, workers
    )
    out["name_levenshtein_sim"] = _cpdist(
        A["name_latin"], B["name_latin"], Levenshtein.normalized_similarity, workers
    )
    out["name_unicode_ratio"] = sim("name_unicode", "name_unicode", fuzz.token_sort_ratio)
    out["name_skeleton_ratio"] = sim("name_core_skeleton", "name_core_skeleton", fuzz.ratio)
    out["name_skeleton_token_set"] = sim(
        "name_core_skeleton", "name_core_skeleton", fuzz.token_set_ratio
    )
    out["name_nospace_ratio"] = sim("name_nospace", "name_nospace", fuzz.ratio)
    out["name_nospace_partial"] = sim("name_nospace", "name_nospace", fuzz.partial_ratio)

    # ---- address: vectorised string scorers -----------------------------
    missing_a = np.fromiter(A["address_missing"], np.int8, n)
    missing_b = np.fromiter(B["address_missing"], np.int8, n)
    any_missing = (missing_a | missing_b).astype(bool)
    out["addr_missing_a"] = missing_a
    out["addr_missing_b"] = missing_b
    out["addr_any_missing"] = any_missing.astype(np.int8)
    for feature, col, scorer in (
        ("addr_ratio", "address_latin", fuzz.ratio),
        ("addr_token_sort_ratio", "address_latin", fuzz.token_sort_ratio),
        ("addr_token_set_ratio", "address_latin", fuzz.token_set_ratio),
        ("addr_partial_ratio", "address_latin", fuzz.partial_ratio),
        ("addr_skeleton_token_set", "address_skeleton", fuzz.token_set_ratio),
    ):
        values = sim(col, col, scorer)
        values[any_missing] = np.nan
        out[feature] = values

    # ---- per-pair Python loop for set / IDF / numeric evidence ----------
    loop_names = (
        "name_alias_best_ratio", "name_token_jaccard", "name_token_containment",
        "name_idf_jaccard", "name_soft_idf_min", "name_soft_idf_max",
        "name_unmatched_max_idf", "name_shared_max_idf", "name_first_token_equal",
        "name_last_token_equal", "name_prefix3_equal", "name_acronym_match",
        "name_len_a", "name_len_b", "name_ntok_a", "name_ntok_b", "name_len_ratio",
        "name_legal_conflict", "name_legal_both_present",
        "addr_token_jaccard", "addr_token_containment", "addr_alpha_containment",
        "addr_idf_jaccard", "addr_soft_idf_min", "addr_soft_idf_max",
        "addr_unmatched_max_idf", "addr_ntok_a", "addr_ntok_b",
        "addr_num_count_a", "addr_num_count_b", "addr_num_jaccard", "addr_num_shared",
        "addr_num_any_shared", "addr_num_suffix_match", "addr_num_conflict",
        "addr_longest_num_equal", "name_a_unmatched_max_noise", "name_b_unmatched_max_noise",
        "addr_primary_equal", "addr_primary_cross", "addr_num_only_a", "addr_num_only_b",
    )
    buffers = {name: np.empty(n, dtype=np.float32) for name in loop_names}
    name_idf = token_stats.name_idf
    address_idf = token_stats.address_idf
    nan = math.nan
    for i in range(n):
        # names
        ta, tb = A["name_core_tokens"][i], B["name_core_tokens"][i]
        sa, sb = set(ta), set(tb)
        best_alias = 0.0
        for pa in A["name_alias_parts"][i]:
            for pb in B["name_alias_parts"][i]:
                score = fuzz.token_sort_ratio(pa, pb)
                if score > best_alias:
                    best_alias = score
        buffers["name_alias_best_ratio"][i] = best_alias / 100.0
        union = sa | sb
        inter = sa & sb
        buffers["name_token_jaccard"][i] = len(inter) / len(union) if union else nan
        smaller = min(len(sa), len(sb))
        buffers["name_token_containment"][i] = len(inter) / smaller if smaller else nan
        idf_j, shared_max = _idf_jaccard(sa, sb, name_idf)
        buffers["name_idf_jaccard"][i] = idf_j
        buffers["name_shared_max_idf"][i] = shared_max
        ka, kb = A["name_core_skeleton_tokens"][i], B["name_core_skeleton_tokens"][i]
        cov_ab, un_a, noise_a = _soft_idf(ta, tb, sb, name_idf, ka, kb, name_noise)
        cov_ba, un_b, noise_b = _soft_idf(tb, ta, sa, name_idf, kb, ka, name_noise)
        buffers["name_a_unmatched_max_noise"][i] = noise_a
        buffers["name_b_unmatched_max_noise"][i] = noise_b
        buffers["name_soft_idf_min"][i] = min(cov_ab, cov_ba)
        buffers["name_soft_idf_max"][i] = max(cov_ab, cov_ba)
        buffers["name_unmatched_max_idf"][i] = max(un_a, un_b)
        buffers["name_first_token_equal"][i] = bool(ta and tb and ta[0] == tb[0])
        buffers["name_last_token_equal"][i] = bool(ta and tb and ta[-1] == tb[-1])
        ca, cb = A["name_nospace"][i], B["name_nospace"][i]
        buffers["name_prefix3_equal"][i] = len(ca) >= 3 and ca[:3] == cb[:3]
        aa, ab = A["name_acronym"][i], B["name_acronym"][i]
        buffers["name_acronym_match"][i] = (
            (len(aa) >= 2 and aa == cb) or (len(ab) >= 2 and ab == ca)
            or (len(aa) >= 3 and aa == ab and len(ta) >= 3)
        )
        len_a, len_b = len(A["name_core"][i]), len(B["name_core"][i])
        buffers["name_len_a"][i] = min(len_a, 32767)
        buffers["name_len_b"][i] = min(len_b, 32767)
        buffers["name_ntok_a"][i] = min(len(ta), 127)
        buffers["name_ntok_b"][i] = min(len(tb), 127)
        buffers["name_len_ratio"][i] = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) else nan
        fa, fb = A["name_legal_families"][i], B["name_legal_families"][i]
        buffers["name_legal_both_present"][i] = bool(fa and fb)
        buffers["name_legal_conflict"][i] = bool(fa and fb and not (fa & fb))

        # addresses
        ada, adb = A["address_tokens"][i], B["address_tokens"][i]
        buffers["addr_ntok_a"][i] = min(len(ada), 32767)
        buffers["addr_ntok_b"][i] = min(len(adb), 32767)
        (
            buffers["addr_num_count_a"][i], buffers["addr_num_count_b"][i],
            buffers["addr_num_jaccard"][i], buffers["addr_num_shared"][i],
            buffers["addr_num_any_shared"][i], buffers["addr_num_suffix_match"][i],
            buffers["addr_num_conflict"][i], buffers["addr_longest_num_equal"][i],
        ) = _number_features(
            A["address_numbers"][i], B["address_numbers"][i],
            A["address_longest_number"][i], B["address_longest_number"][i],
        )
        num_a, num_b = A["address_numbers"][i], B["address_numbers"][i]
        buffers["addr_num_only_a"][i] = min(len(num_a - num_b), 127)
        buffers["addr_num_only_b"][i] = min(len(num_b - num_a), 127)
        prim_a, prim_b = A["address_primary_number"][i], B["address_primary_number"][i]
        if prim_a and prim_b:
            buffers["addr_primary_equal"][i] = prim_a == prim_b
            buffers["addr_primary_cross"][i] = (
                set(prim_a.split(".")) <= num_b and set(prim_b.split(".")) <= num_a
            )
        else:
            buffers["addr_primary_equal"][i] = nan
            buffers["addr_primary_cross"][i] = nan
        if any_missing[i]:
            for key in (
                "addr_token_jaccard", "addr_token_containment", "addr_alpha_containment",
                "addr_idf_jaccard", "addr_soft_idf_min", "addr_soft_idf_max",
                "addr_unmatched_max_idf",
            ):
                buffers[key][i] = nan
            buffers["addr_longest_num_equal"][i] = 0
            continue
        inter = ada & adb
        buffers["addr_token_jaccard"][i] = len(inter) / len(ada | adb)
        buffers["addr_token_containment"][i] = len(inter) / min(len(ada), len(adb))
        pa_, pb_ = A["address_alpha_tokens"][i], B["address_alpha_tokens"][i]
        smaller = min(len(pa_), len(pb_))
        buffers["addr_alpha_containment"][i] = len(pa_ & pb_) / smaller if smaller else nan
        buffers["addr_idf_jaccard"][i] = _idf_jaccard(ada, adb, address_idf)[0]
        cov_ab, un_a = _soft_idf(tuple(pa_), tuple(pb_), pb_, address_idf)
        cov_ba, un_b = _soft_idf(tuple(pb_), tuple(pa_), pa_, address_idf)
        buffers["addr_soft_idf_min"][i] = min(cov_ab, cov_ba)
        buffers["addr_soft_idf_max"][i] = max(cov_ab, cov_ba)
        buffers["addr_unmatched_max_idf"][i] = max(un_a, un_b) if not (
            math.isnan(un_a) or math.isnan(un_b)) else nan
    out.update(buffers)
    # When either side has no numbers the numeric comparison is "not applicable",
    # not a conflict; the count columns carry that information explicitly.

    # ---- script / country -----------------------------------------------
    nl_a = np.fromiter(A["name_non_latin"], np.int8, n)
    nl_b = np.fromiter(B["name_non_latin"], np.int8, n)
    out["name_non_latin_a"] = nl_a
    out["name_non_latin_b"] = nl_b
    out["name_script_mismatch"] = (nl_a != nl_b).astype(np.int8)
    out["same_country"] = np.fromiter(
        (a.casefold() == b.casefold() for a, b in zip(A["country"], B["country"])), np.int8, n
    )
    out["target_is_s3"] = np.fromiter((x.startswith("S3") for x in sec_ids), np.int8, n)

    # ---- combined evidence ----------------------------------------------
    name_ev = name_evidence_np(out)
    addr_ev = out["addr_token_set_ratio"]
    stacked = np.vstack([name_ev, addr_ev])
    with np.errstate(invalid="ignore"):
        out["combo_min"] = np.where(any_missing, np.nan, np.nanmin(stacked, axis=0))
        out["combo_max"] = np.where(any_missing, np.nan, np.nanmax(stacked, axis=0))
        out["combo_product"] = np.where(any_missing, np.nan, name_ev * addr_ev)
        out["combo_mean"] = np.where(any_missing, name_ev, (name_ev + addr_ev) / 2)
        out["combo_name_high_addr_low"] = ((name_ev >= 0.9) & (addr_ev < 0.5)).astype(np.int8)
        out["combo_addr_high_name_low"] = ((addr_ev >= 0.9) & (name_ev < 0.5)).astype(np.int8)

    features = pl.DataFrame(
        {name: pl.Series(name, out[name]).cast(dtype, strict=False)
         for name, dtype in PAIR_FEATURE_SCHEMA.items()}
    ).with_columns(
        pl.col(name).fill_nan(NA_SENTINEL_OVERRIDES.get(name, NA_SENTINEL))
        .fill_null(NA_SENTINEL_OVERRIDES.get(name, NA_SENTINEL))
        for name, dtype in PAIR_FEATURE_SCHEMA.items() if dtype.is_float()
    )
    hashes = pl.DataFrame({
        "_core_hash_a": [core_hash(c) for c in A["name_core"]],
        "_core_hash_b": [core_hash(c) for c in B["name_core"]],
    }, schema={"_core_hash_a": pl.Int64, "_core_hash_b": pl.Int64})
    return pl.concat([pairs, features, hashes], how="horizontal")


# Single "name evidence" definition shared by combo, context and hard-negative
# code. token_set-style scores are avoided here because they reach 1.0 whenever
# one side's (possibly tiny) token set is contained in the other.
NAME_EVIDENCE_COLUMNS = ("name_token_sort_ratio", "name_skeleton_ratio", "name_alias_best_ratio")


def name_evidence_np(values: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum.reduce([values[c] for c in NAME_EVIDENCE_COLUMNS])


def name_evidence_expr() -> pl.Expr:
    return pl.max_horizontal(*NAME_EVIDENCE_COLUMNS)


# ---------------------------------------------------------------------------
# Group-level context features
# ---------------------------------------------------------------------------

def add_context_features(features: pl.DataFrame) -> pl.DataFrame:
    """Rank/gap context inside each S1 group and across competing S1 records.

    Must run on the complete candidate table (all chunks concatenated) so that
    every S1 group and every candidate's competing-S1 set is whole.
    """

    name_score = name_evidence_expr()
    # Missing addresses give null rank/gap (NaN), never a fake low score.
    addr_score = pl.when(pl.col("addr_any_missing") == 1).then(None).otherwise(
        pl.col("addr_token_set_ratio"))
    s1 = "source1_entity_id"
    cand = "candidate_entity_id"
    df = features.with_columns(
        _name=name_score, _addr=addr_score, _combo=pl.col("combo_mean")
    )
    df = df.with_columns(
        ctx_n_candidates=pl.len().over(s1),
        ctx_n_candidates_source=pl.len().over(s1, "target_source"),
        ctx_name_rank=pl.col("_name").rank("min", descending=True).over(s1),
        ctx_name_gap=pl.col("_name").max().over(s1) - pl.col("_name"),
        ctx_addr_rank=pl.col("_addr").rank("min", descending=True).over(s1),
        ctx_addr_gap=pl.col("_addr").max().over(s1) - pl.col("_addr"),
        ctx_combo_rank=pl.col("_combo").rank("min", descending=True).over(s1),
        ctx_combo_gap=pl.col("_combo").max().over(s1) - pl.col("_combo"),
        ctx_rev_n_s1=pl.len().over(cand),
        ctx_rev_combo_rank=pl.col("_combo").rank("min", descending=True).over(cand),
        ctx_rev_combo_gap=pl.col("_combo").max().over(cand) - pl.col("_combo"),
    )
    # Gap between this candidate and the best *other* candidate in the S1 group
    # (positive when this candidate is the clear winner).
    df = df.with_columns(
        ctx_combo_second_gap=pl.when(pl.col("ctx_combo_rank") == 1)
        .then(pl.col("_combo") - pl.col("_combo").top_k(2).min().over(s1))
        .otherwise(pl.col("_combo") - pl.col("_combo").max().over(s1))
    )
    df = df.with_columns(
        [pl.col(c).cast(t) for c, t in CONTEXT_FEATURE_SCHEMA.items()]
    ).drop("_name", "_addr", "_combo")
    return df


def add_retrieval_features(features: pl.DataFrame, candidates_long: pl.DataFrame) -> pl.DataFrame:
    """Aggregate Person 2's long candidate table (one row per route) per pair."""

    key = ["source1_entity_id", "candidate_entity_id"]
    route_best = candidates_long.group_by(["source1_entity_id", "retrieval_route"]).agg(
        pl.col("retrieval_score").max().alias("_route_top")
    )
    agg = (
        candidates_long.join(route_best, on=["source1_entity_id", "retrieval_route"])
        .with_columns(_gap=pl.col("_route_top") - pl.col("retrieval_score"))
        .group_by(key)
        .agg(
            retr_min_rank=pl.col("retrieval_rank").min(),
            retr_n_routes=pl.col("retrieval_route").n_unique(),
            retr_max_score=pl.col("retrieval_score").max(),
            retr_score_gap=pl.col("_gap").min(),
        )
        .with_columns([pl.col(c).cast(t) for c, t in RETRIEVAL_FEATURE_SCHEMA.items()])
    )
    return features.join(agg, on=key, how="left", maintain_order="left")


def model_feature_columns(include_context: bool = True, include_retrieval: bool = False) -> list[str]:
    columns = list(PAIR_FEATURE_COLUMNS) + list(NAME_FREQUENCY_SCHEMA)
    if include_context:
        columns += CONTEXT_FEATURE_COLUMNS
    if include_retrieval:
        columns += RETRIEVAL_FEATURE_COLUMNS
    return columns


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------------
# Process-parallel driver (same function, same output as the serial path)
# ---------------------------------------------------------------------------

_WORKER_STATS: TokenStats | None = None
_WORKER_NOISE: dict[str, float] | None = None


def _init_worker(token_stats_path: str, name_stats_dir: str) -> None:
    global _WORKER_STATS, _WORKER_NOISE
    _WORKER_STATS = TokenStats.load(token_stats_path)
    _WORKER_NOISE = load_token_noise(name_stats_dir)


def _feature_chunk(payload: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    pairs, s1_raw, sec_raw = payload
    return calculate_pair_features(
        pairs, prepare_records(s1_raw), prepare_records(sec_raw), _WORKER_STATS,
        name_noise=_WORKER_NOISE, workers=1,
    )


def calculate_pair_features_parallel(
    pairs: pl.DataFrame,
    source1_raw: pl.DataFrame,
    secondary_raw: pl.DataFrame,
    token_stats_path: str | os.PathLike,
    name_stats_dir: str | os.PathLike,
    *,
    processes: int | None = None,
    chunk_pairs: int = 50_000,
) -> pl.DataFrame:
    """Run ``calculate_pair_features`` over S1-aligned chunks in worker processes.

    Each worker receives only the raw records its chunk references, prepares
    them and computes features, so nothing large is pickled. Chunks never split
    an S1 group and results are concatenated in input order, so the output is
    identical to one serial call. Name-frequency features are joined here, so
    the result is ready for ``add_context_features``.
    """

    processes = processes or default_workers()
    source1_raw = source1_raw.with_columns(pl.all().fill_null(""))
    secondary_raw = secondary_raw.with_columns(pl.all().fill_null(""))
    group_ids = pairs["source1_entity_id"]
    # Chunk boundaries only where the S1 id changes (pairs are expected grouped).
    boundaries = [0]
    change = (group_ids != group_ids.shift(1)).fill_null(True).to_numpy()
    starts = np.flatnonzero(change)
    next_cut = chunk_pairs
    for start in starts:
        if start >= next_cut:
            boundaries.append(int(start))
            next_cut = start + chunk_pairs
    boundaries.append(pairs.height)
    payloads = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        chunk = pairs.slice(lo, hi - lo)
        payloads.append((
            chunk,
            source1_raw.filter(pl.col("entity_id").is_in(chunk["source1_entity_id"].unique().implode())),
            secondary_raw.filter(pl.col("entity_id").is_in(chunk["candidate_entity_id"].unique().implode())),
        ))
    if processes <= 1 or len(payloads) <= 1:
        _init_worker(str(token_stats_path), str(name_stats_dir))
        features = pl.concat([_feature_chunk(p) for p in payloads])
    else:
        with ProcessPoolExecutor(processes, initializer=_init_worker,
                                 initargs=(str(token_stats_path), str(name_stats_dir))) as pool:
            features = pl.concat(list(pool.map(_feature_chunk, payloads)))
    return add_name_frequency_features(features, name_stats_dir)


# ---------------------------------------------------------------------------
# Feature dictionary (kept beside the schema so documentation cannot drift).
# "core" = Latin view with legal-form words, honorifics and DBA/FKA markers
# removed; "Latin" = accent-folded view with Indic scripts romanised;
# "skeleton" = script-neutral phonetic consonant key. Similarities are in [0, 1].
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "name_exact_latin": "Latin names identical",
    "name_exact_core": "Core names identical",
    "name_ratio": "Indel ratio of core names",
    "name_token_sort_ratio": "Ratio after sorting core tokens (word-order tolerant)",
    "name_token_set_ratio": "Token-set ratio of core names (subset tolerant)",
    "name_partial_ratio": "Best substring alignment of core names",
    "name_wratio_latin": "RapidFuzz WRatio on full Latin names",
    "name_jaro_winkler": "Jaro-Winkler on token-sorted core names",
    "name_levenshtein_sim": "Normalised Levenshtein similarity of Latin names",
    "name_unicode_ratio": "Token-sort ratio on Unicode-preserving names (local-script evidence)",
    "name_skeleton_ratio": "Ratio of phonetic skeletons (cross-script, vowel/voicing/OCR tolerant)",
    "name_skeleton_token_set": "Token-set ratio of phonetic skeletons",
    "name_nospace_ratio": "Ratio of core names with spaces removed (handles, URLs, joined words)",
    "name_nospace_partial": "Partial ratio of space-free core names",
    "name_alias_best_ratio": "Best token-sort ratio over DBA/FKA/AKA name segments",
    "name_token_jaccard": "Jaccard of core token sets",
    "name_token_containment": "Shared core tokens / smaller token set",
    "name_idf_jaccard": "IDF-weighted Jaccard of core tokens",
    "name_soft_idf_min": "Soft-IDF coverage (exact/fuzzy>=80/skeleton), weaker direction",
    "name_soft_idf_max": "Soft-IDF coverage, stronger direction",
    "name_unmatched_max_idf": "Highest IDF of a core token with no fuzzy/skeleton partner (distinctive mismatch)",
    "name_shared_max_idf": "Highest IDF among exactly shared core tokens (0 = none shared)",
    "name_first_token_equal": "First core tokens equal",
    "name_last_token_equal": "Last core tokens equal",
    "name_prefix3_equal": "First three characters of space-free core names equal",
    "name_acronym_match": "One name is the acronym of the other (or equal 3+ letter acronyms)",
    "name_len_a": "S1 core-name length (chars)",
    "name_len_b": "Candidate core-name length (chars)",
    "name_ntok_a": "S1 core-name token count",
    "name_ntok_b": "Candidate core-name token count",
    "name_len_ratio": "Shorter / longer core-name length",
    "name_legal_conflict": "Both names carry legal forms and the families are disjoint (e.g. LLC vs Inc)",
    "name_legal_both_present": "Both names carry a legal-form word",
    "name_non_latin_a": "S1 name contains non-Latin letters",
    "name_non_latin_b": "Candidate name contains non-Latin letters",
    "name_script_mismatch": "Exactly one name contains non-Latin letters",
    "addr_missing_a": "S1 address missing (blank or placeholder such as None/null/N/A)",
    "addr_missing_b": "Candidate address missing",
    "addr_any_missing": "Either address missing; all addr_* similarities are NaN when set",
    "addr_ratio": "Indel ratio of normalised Latin addresses",
    "addr_token_sort_ratio": "Token-sort ratio of addresses (component reordering tolerant)",
    "addr_token_set_ratio": "Token-set ratio of addresses (partial-address tolerant)",
    "addr_partial_ratio": "Best substring alignment of addresses",
    "addr_skeleton_token_set": "Token-set ratio of address phonetic skeletons (transliteration tolerant)",
    "addr_token_jaccard": "Jaccard of address tokens",
    "addr_token_containment": "Shared address tokens / smaller token set",
    "addr_alpha_containment": "Containment over alphabetic address tokens only (street/locality words)",
    "addr_idf_jaccard": "IDF-weighted Jaccard of address tokens",
    "addr_soft_idf_min": "Soft-IDF coverage of alphabetic address tokens, weaker direction",
    "addr_soft_idf_max": "Soft-IDF coverage of alphabetic address tokens, stronger direction",
    "addr_unmatched_max_idf": "Highest IDF of an alphabetic address token with no fuzzy partner",
    "addr_ntok_a": "S1 address token count",
    "addr_ntok_b": "Candidate address token count",
    "addr_num_count_a": "Numbers in S1 address (leading zeros and ordinals normalised)",
    "addr_num_count_b": "Numbers in candidate address",
    "addr_num_jaccard": "Jaccard of address number sets (NaN when either side has none)",
    "addr_num_shared": "Count of shared address numbers",
    "addr_num_any_shared": "At least one address number shared",
    "addr_num_suffix_match": "No exact shared number but one ends with another (5550 vs 550)",
    "addr_num_conflict": "Both have numbers, none shared and no suffix match",
    "addr_longest_num_equal": "Longest address numbers equal (often house number or postcode)",
    "addr_primary_equal": "Primary compound number (all numbers of the first numeric component, e.g. 8.3.898.30.3) equal; NaN if absent",
    "addr_primary_cross": "Each side's primary numbers all appear in the other address; NaN if absent",
    "addr_num_only_a": "Address numbers present only on the S1 side",
    "addr_num_only_b": "Address numbers present only on the candidate side",
    "name_a_unmatched_max_noise": "Max S2/S3-vs-S1 log frequency ratio among S1 core tokens with no fuzzy/skeleton partner (NaN if all matched)",
    "name_b_unmatched_max_noise": "Same for candidate tokens; high = unmatched word looks like injected noise (services, holdings, www)",
    "name_core_freq_s1_a": "log1p(# S1 records, train+test, with the S1 core name)",
    "name_core_freq_sec_a": "log1p(# S2/S3 records with the S1 core name)",
    "name_core_freq_s1_b": "log1p(# S1 records with the candidate core name)",
    "name_core_freq_sec_b": "log1p(# S2/S3 records with the candidate core name)",
    "combo_min": "min(name evidence, address token-set); NaN if an address is missing",
    "combo_max": "max(name evidence, address token-set); NaN if an address is missing",
    "combo_product": "name evidence x address token-set; NaN if an address is missing",
    "combo_mean": "Mean of name evidence and address token-set (name only when address missing)",
    "combo_name_high_addr_low": "Name evidence >= 0.9 and address token-set < 0.5",
    "combo_addr_high_name_low": "Address token-set >= 0.9 and name evidence < 0.5",
    "same_country": "Country strings equal (case-insensitive, open set)",
    "target_is_s3": "Candidate comes from Source 3",
    "ctx_n_candidates": "Candidates for this S1",
    "ctx_n_candidates_source": "Candidates for this S1 from the same target source",
    "ctx_name_rank": "Rank of name evidence within the S1 group (1 = best)",
    "ctx_name_gap": "Best name evidence in S1 group minus this pair's",
    "ctx_addr_rank": "Rank of address token-set within S1 group (null if address missing)",
    "ctx_addr_gap": "Best address token-set in S1 group minus this pair's",
    "ctx_combo_rank": "Rank of combo_mean within the S1 group",
    "ctx_combo_gap": "Best combo_mean in S1 group minus this pair's",
    "ctx_combo_second_gap": "Margin over the best other candidate (positive only for the top candidate)",
    "ctx_rev_n_s1": "Number of S1 records that retrieved this candidate",
    "ctx_rev_combo_rank": "Rank of this S1 among all S1 records competing for the candidate",
    "ctx_rev_combo_gap": "Best competing combo_mean for the candidate minus this pair's",
    "retr_min_rank": "Best retrieval rank over routes (Person 2)",
    "retr_n_routes": "Number of retrieval routes that proposed the pair",
    "retr_max_score": "Max retrieval score over routes",
    "retr_score_gap": "Smallest gap to the top score of the same route within the S1",
}


# ---------------------------------------------------------------------------
# Frozen model contract (Day 2). Context and retrieval features are excluded:
# ablation showed context is neutral (95% CI spans zero on both architectures)
# and both depend on whole-table/candidate-generation details that differ
# between training (positive guards) and inference.
# ---------------------------------------------------------------------------

FROZEN_MODEL_FEATURES: tuple[str, ...] = tuple(model_feature_columns(include_context=False))
FROZEN_FEATURE_DTYPES: dict[str, pl.DataType] = {
    **PAIR_FEATURE_SCHEMA, **NAME_FREQUENCY_SCHEMA,
}
FROZEN_FEATURE_DTYPES = {c: FROZEN_FEATURE_DTYPES[c] for c in FROZEN_MODEL_FEATURES}


def validate_feature_frame(features: pl.DataFrame) -> None:
    """Raise if frozen columns are missing, mistyped, NaN/null, infinite or unexpected."""

    missing = [c for c in FROZEN_MODEL_FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"feature frame is missing frozen columns: {missing}")
    wrong = {c: str(features.schema[c]) for c, t in FROZEN_FEATURE_DTYPES.items()
             if features.schema[c] != t}
    if wrong:
        raise ValueError(f"feature dtypes differ from the frozen contract: {wrong}")
    infinite = [c for c, t in FROZEN_FEATURE_DTYPES.items()
                if t.is_float() and int(features[c].is_infinite().sum())]
    if infinite:
        raise ValueError(f"infinite values in: {infinite}")
    nan_or_null = [c for c, t in FROZEN_FEATURE_DTYPES.items()
                   if features[c].null_count() or (t.is_float() and int(features[c].is_nan().sum()))]
    if nan_or_null:
        raise ValueError(f"NaN/null values in: {nan_or_null}")
    leftovers = [c for c in features.columns if c.startswith("_core_hash")]
    if leftovers:
        raise ValueError("name-frequency join not applied (hash helper columns present)")
