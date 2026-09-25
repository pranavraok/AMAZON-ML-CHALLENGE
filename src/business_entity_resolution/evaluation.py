"""Exact entity-level macro F0.5 evaluation and Day 1 output helpers."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Iterable, Mapping

from business_entity_resolution.records import write_tsv
from business_entity_resolution.schemas import (
    CANDIDATE_PAIR_COLUMNS,
    CANDIDATE_RESULTS_COLUMNS,
    MATCHING_RESULTS_COLUMNS,
    MODEL_SCORE_COLUMNS,
    require_exact_columns,
)


def entity_fbeta(
    predicted: set[str],
    expected: set[str],
    *,
    beta: float = 0.5,
) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    true_positives = len(predicted & expected)
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(predicted)
    recall = true_positives / len(expected)
    beta_squared = beta * beta
    return (
        (1.0 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall)
    )


def macro_fbeta(
    predictions: Mapping[str, set[str]],
    truth: Mapping[str, set[str]],
    entity_ids: Iterable[str],
    *,
    beta: float = 0.5,
) -> float:
    ids = list(entity_ids)
    if not ids:
        raise ValueError("macro F-beta requires at least one Source 1 entity")
    return sum(
        entity_fbeta(predictions.get(entity_id, set()), truth.get(entity_id, set()), beta=beta)
        for entity_id in ids
    ) / len(ids)


def read_scores(path: Path) -> dict[str, list[tuple[str, float]]]:
    scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(reader.fieldnames, MODEL_SCORE_COLUMNS, label=str(path))
        for row in reader:
            scores[row["source1_entity_id"]].append(
                (row["candidate_entity_id"], float(row["match_probability"]))
            )
    return scores


def predictions_at_threshold(
    scores: Mapping[str, list[tuple[str, float]]],
    *,
    threshold: float,
) -> dict[str, set[str]]:
    return {
        source1_id: {
            candidate_id
            for candidate_id, probability in candidates
            if probability >= threshold
        }
        for source1_id, candidates in scores.items()
    }


def threshold_search(
    score_path: Path,
    *,
    truth: Mapping[str, set[str]],
    entity_ids: set[str],
    thresholds: Iterable[float] | None = None,
) -> dict[str, object]:
    scores = read_scores(score_path)
    threshold_values = list(thresholds or [index / 100 for index in range(5, 100, 5)])
    trials: list[dict[str, float]] = []
    for threshold in threshold_values:
        predictions = predictions_at_threshold(scores, threshold=threshold)
        value = macro_fbeta(predictions, truth, entity_ids, beta=0.5)
        trials.append({"threshold": float(threshold), "macro_f0_5": float(value)})

    best = max(trials, key=lambda row: (row["macro_f0_5"], row["threshold"]))
    best_predictions = predictions_at_threshold(
        scores, threshold=float(best["threshold"])
    )
    true_positives = false_positives = false_negatives = 0
    singleton_total = singleton_correct = 0
    for entity_id in entity_ids:
        predicted = best_predictions.get(entity_id, set())
        expected = truth.get(entity_id, set())
        true_positives += len(predicted & expected)
        false_positives += len(predicted - expected)
        false_negatives += len(expected - predicted)
        if not expected:
            singleton_total += 1
            singleton_correct += int(not predicted)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return {
        "best_threshold": best["threshold"],
        "macro_f0_5": best["macro_f0_5"],
        "micro_precision": precision,
        "micro_recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "validation_entities": len(entity_ids),
        "singleton_entities": singleton_total,
        "singleton_accuracy": (
            singleton_correct / singleton_total if singleton_total else 1.0
        ),
        "threshold_trials": trials,
    }


def write_validation_outputs(
    raw_candidate_path: Path,
    score_path: Path,
    matching_path: Path,
    candidate_output_path: Path,
    *,
    entity_ids: set[str],
    threshold: float,
) -> tuple[int, int]:
    candidate_ids: dict[str, set[str]] = defaultdict(set)
    with raw_candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(
            reader.fieldnames, CANDIDATE_PAIR_COLUMNS, label=str(raw_candidate_path)
        )
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id in entity_ids:
                candidate_ids[source1_id].add(row["candidate_entity_id"])

    scores = read_scores(score_path)
    predictions = predictions_at_threshold(scores, threshold=threshold)
    ordered_ids = sorted(entity_ids)
    matching_rows = (
        {
            "source1_entity_id": entity_id,
            "matched_entity_ids": ",".join(sorted(predictions.get(entity_id, set()))),
        }
        for entity_id in ordered_ids
    )
    candidate_rows = (
        {
            "source1_entity_id": entity_id,
            "candidate_entity_ids": ",".join(sorted(candidate_ids.get(entity_id, set()))),
        }
        for entity_id in ordered_ids
    )
    matching_count = write_tsv(matching_path, MATCHING_RESULTS_COLUMNS, matching_rows)
    candidate_count = write_tsv(
        candidate_output_path, CANDIDATE_RESULTS_COLUMNS, candidate_rows
    )
    return matching_count, candidate_count
