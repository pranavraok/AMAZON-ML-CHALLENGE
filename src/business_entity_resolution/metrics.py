"""Evaluation metrics for candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from business_entity_resolution.blocking import CandidateEvidence
from business_entity_resolution.normalization import NormalizedRecord


@dataclass(frozen=True)
class RecallReport:
    """Candidate-generation recall report."""

    total_true_pairs: int
    retrieved_true_pairs: int
    missed_true_pairs: int

    pair_recall: float

    total_source1_entities: int
    entities_with_all_links_retrieved: int

    entity_coverage: float

    average_candidate_count: float
    median_candidate_count: float
    p95_candidate_count: float
    p99_candidate_count: float
    maximum_candidate_count: int


def _percentile(
    values: Sequence[int],
    percentile: float,
) -> float:
    """Calculate a percentile without requiring pandas."""

    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered))

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def evaluate_candidate_recall(
    candidate_results: Mapping[
        str,
        Sequence[CandidateEvidence],
    ],
    ground_truth: Mapping[str, set[str]],
) -> RecallReport:
    """Evaluate candidate recall against ground truth.

    Parameters
    ----------
    candidate_results:
        Mapping:
            S1 ID -> candidate evidence list

    ground_truth:
        Mapping:
            S1 ID -> set of true S2/S3 IDs

    Returns
    -------
    RecallReport
    """

    total_true_pairs = 0
    retrieved_true_pairs = 0
    entities_with_all_links_retrieved = 0

    candidate_counts: list[int] = []

    for source1_id, true_matches in ground_truth.items():

        candidates = candidate_results.get(
            source1_id,
            [],
        )

        candidate_ids = {
            candidate.candidate_entity_id
            for candidate in candidates
        }

        candidate_counts.append(
            len(candidate_ids)
        )

        total_true_pairs += len(true_matches)

        retrieved = true_matches & candidate_ids

        retrieved_true_pairs += len(retrieved)

        if true_matches.issubset(candidate_ids):
            entities_with_all_links_retrieved += 1

    missed_true_pairs = (
        total_true_pairs - retrieved_true_pairs
    )

    if total_true_pairs:
        pair_recall = (
            retrieved_true_pairs
            / total_true_pairs
        )
    else:
        pair_recall = 1.0

    total_entities = len(ground_truth)

    entity_coverage = (
        entities_with_all_links_retrieved
        / total_entities
        if total_entities
        else 1.0
    )

    average_candidate_count = (
        sum(candidate_counts)
        / len(candidate_counts)
        if candidate_counts
        else 0.0
    )

    median_candidate_count = (
        _percentile(candidate_counts, 0.50)
        if candidate_counts
        else 0.0
    )

    p95_candidate_count = (
        _percentile(candidate_counts, 0.95)
        if candidate_counts
        else 0.0
    )

    p99_candidate_count = (
        _percentile(candidate_counts, 0.99)
        if candidate_counts
        else 0.0
    )

    maximum_candidate_count = (
        max(candidate_counts)
        if candidate_counts
        else 0
    )

    return RecallReport(
        total_true_pairs=total_true_pairs,
        retrieved_true_pairs=retrieved_true_pairs,
        missed_true_pairs=missed_true_pairs,
        pair_recall=pair_recall,
        total_source1_entities=total_entities,
        entities_with_all_links_retrieved=(
            entities_with_all_links_retrieved
        ),
        entity_coverage=entity_coverage,
        average_candidate_count=average_candidate_count,
        median_candidate_count=median_candidate_count,
        p95_candidate_count=p95_candidate_count,
        p99_candidate_count=p99_candidate_count,
        maximum_candidate_count=maximum_candidate_count,
    )


def find_missed_pairs(
    candidate_results: Mapping[
        str,
        Sequence[CandidateEvidence],
    ],
    ground_truth: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    """Return true links missed by candidate generation."""

    missed: dict[str, set[str]] = {}

    for source1_id, true_matches in ground_truth.items():

        candidates = candidate_results.get(
            source1_id,
            []
        )

        candidate_ids = {
            candidate.candidate_entity_id
            for candidate in candidates
        }

        missing = true_matches - candidate_ids

        if missing:
            missed[source1_id] = missing

    return missed