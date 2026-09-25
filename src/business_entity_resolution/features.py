"""Deterministic comparison features for the replaceable Day 1 baseline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from rapidfuzz import fuzz

from business_entity_resolution.records import PreparedRecord, write_tsv
from business_entity_resolution.schemas import CANDIDATE_PAIR_COLUMNS, require_exact_columns


FEATURE_COLUMNS = (
    "country_equal",
    "name_unicode_exact",
    "name_ascii_exact",
    "address_unicode_exact",
    "address_ascii_exact",
    "name_token_jaccard",
    "address_token_jaccard",
    "address_number_jaccard",
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "address_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "name_prefix_ratio",
    "address_prefix_ratio",
    "name_length_ratio",
    "address_length_ratio",
    "both_address_missing",
    "one_address_missing",
    "retrieval_score",
    "retrieval_rank_inverse",
    "route_count",
    "target_is_s3",
)

# Retrieval score/rank are retained in the handoff file, but the temporary positive
# guard used only for training does not exist at inference. The Day 1 model therefore
# excludes those two values to prevent a training-only route from becoming leakage.
MODEL_FEATURE_COLUMNS = tuple(
    column
    for column in FEATURE_COLUMNS
    if column not in {"retrieval_score", "retrieval_rank_inverse"}
)

FEATURE_FILE_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "target_source",
    "label",
    "fold",
    *FEATURE_COLUMNS,
)


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _prefix_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    limit = min(len(left), len(right))
    count = 0
    while count < limit and left[count] == right[count]:
        count += 1
    return count / max(len(left), len(right), 1)


def _length_ratio(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    return min(len(left), len(right)) / longest


def comparison_features(
    source1: PreparedRecord,
    candidate: PreparedRecord,
    *,
    retrieval_route: str,
    retrieval_rank: int,
    retrieval_score: float,
) -> dict[str, float | int]:
    left = source1.normalized
    right = candidate.normalized
    left_address_missing = left.address_missing
    right_address_missing = right.address_missing

    return {
        "country_equal": int(source1.country == candidate.country),
        "name_unicode_exact": int(bool(left.name_unicode) and left.name_unicode == right.name_unicode),
        "name_ascii_exact": int(bool(left.name_ascii) and left.name_ascii == right.name_ascii),
        "address_unicode_exact": int(
            bool(left.address_unicode) and left.address_unicode == right.address_unicode
        ),
        "address_ascii_exact": int(
            bool(left.address_ascii) and left.address_ascii == right.address_ascii
        ),
        "name_token_jaccard": _jaccard(left.name_tokens, right.name_tokens),
        "address_token_jaccard": _jaccard(left.address_tokens, right.address_tokens),
        "address_number_jaccard": _jaccard(left.address_numbers, right.address_numbers),
        "name_ratio": fuzz.ratio(left.name_unicode, right.name_unicode) / 100.0,
        "name_token_sort_ratio": (
            fuzz.token_sort_ratio(left.name_unicode, right.name_unicode) / 100.0
        ),
        "name_token_set_ratio": (
            fuzz.token_set_ratio(left.name_unicode, right.name_unicode) / 100.0
        ),
        "address_ratio": fuzz.ratio(left.address_unicode, right.address_unicode) / 100.0,
        "address_token_sort_ratio": (
            fuzz.token_sort_ratio(left.address_unicode, right.address_unicode) / 100.0
        ),
        "address_token_set_ratio": (
            fuzz.token_set_ratio(left.address_unicode, right.address_unicode) / 100.0
        ),
        "name_prefix_ratio": _prefix_ratio(left.name_unicode, right.name_unicode),
        "address_prefix_ratio": _prefix_ratio(
            left.address_unicode, right.address_unicode
        ),
        "name_length_ratio": _length_ratio(left.name_unicode, right.name_unicode),
        "address_length_ratio": _length_ratio(
            left.address_unicode, right.address_unicode
        ),
        "both_address_missing": int(left_address_missing and right_address_missing),
        "one_address_missing": int(left_address_missing != right_address_missing),
        "retrieval_score": retrieval_score,
        "retrieval_rank_inverse": 1.0 / max(retrieval_rank, 1),
        "route_count": len(set(retrieval_route.split("+"))),
        "target_is_s3": int(candidate.entity_id.startswith("S3-")),
    }


def write_feature_rows(
    candidate_path: Path,
    output_path: Path,
    *,
    source1_records: Mapping[str, PreparedRecord],
    secondary_records: Mapping[str, PreparedRecord],
    truth: Mapping[str, set[str]] | None,
    folds: Mapping[str, int],
) -> int:
    def rows():
        with candidate_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require_exact_columns(
                reader.fieldnames, CANDIDATE_PAIR_COLUMNS, label=str(candidate_path)
            )
            for candidate_row in reader:
                source1_id = candidate_row["source1_entity_id"]
                candidate_id = candidate_row["candidate_entity_id"]
                try:
                    source1 = source1_records[source1_id]
                    candidate = secondary_records[candidate_id]
                except KeyError as error:
                    raise ValueError(
                        f"candidate row references unknown ID {error.args[0]}"
                    ) from error
                feature_values = comparison_features(
                    source1,
                    candidate,
                    retrieval_route=candidate_row["retrieval_route"],
                    retrieval_rank=int(candidate_row["retrieval_rank"]),
                    retrieval_score=float(candidate_row["retrieval_score"]),
                )
                yield {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate_id,
                    "target_source": candidate_row["target_source"],
                    "label": int(candidate_id in (truth or {}).get(source1_id, set())),
                    "fold": folds[source1_id],
                    **{column: feature_values[column] for column in FEATURE_COLUMNS},
                }

    return write_tsv(output_path, FEATURE_FILE_COLUMNS, rows())
