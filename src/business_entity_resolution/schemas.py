"""Shared column contracts for all pipeline modules."""

from __future__ import annotations

from typing import Iterable, Sequence


SOURCE_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)

GROUND_TRUTH_COLUMNS = (
    "source1_entity_id",
    "matched_entity_ids",
)

CANDIDATE_PAIR_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "target_source",
    "retrieval_route",
    "retrieval_rank",
    "retrieval_score",
)

FEATURE_IDENTIFIER_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "target_source",
)

MODEL_SCORE_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "target_source",
    "match_probability",
)

MATCHING_RESULTS_COLUMNS = (
    "source1_entity_id",
    "matched_entity_ids",
)

CANDIDATE_RESULTS_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_ids",
)


def require_exact_columns(
    actual: Sequence[str] | None,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    """Raise a clear error when a TSV header differs from its contract."""

    if actual is None:
        raise ValueError(f"{label} has no header row")
    actual_tuple = tuple(actual)
    expected_tuple = tuple(expected)
    if actual_tuple != expected_tuple:
        raise ValueError(
            f"{label} has columns {actual_tuple}; expected exactly {expected_tuple}"
        )


def require_columns(
    actual: Iterable[str],
    required: Iterable[str],
    *,
    label: str,
) -> None:
    """Raise when one or more required columns are missing."""

    actual_set = set(actual)
    missing = [column for column in required if column not in actual_set]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
