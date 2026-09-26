"""Error slices and robustness summaries for model handoff."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Mapping

from business_entity_resolution.evaluation import entity_fbeta
from business_entity_resolution.records import write_tsv
from business_entity_resolution.schemas import MODEL_SCORE_COLUMNS, require_exact_columns


ERROR_SLICE_COLUMNS = (
    "source1_entity_id",
    "country",
    "error_type",
    "candidate_entity_id",
    "target_source",
    "match_probability",
    "threshold",
)


def _read_detailed_scores(
    score_path: Path,
) -> dict[str, dict[str, tuple[str, float]]]:
    scores: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(reader.fieldnames, MODEL_SCORE_COLUMNS, label=str(score_path))
        for row in reader:
            scores[row["source1_entity_id"]][row["candidate_entity_id"]] = (
                row["target_source"],
                float(row["match_probability"]),
            )
    return scores


def write_error_slices(
    score_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    truth: Mapping[str, set[str]],
    countries: Mapping[str, str],
    entity_ids: set[str],
    threshold: float,
) -> dict[str, object]:
    """Write pair-level false-positive/false-negative cases for Person 4."""

    scores = _read_detailed_scores(score_path)
    rows: list[dict[str, object]] = []
    error_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for entity_id in sorted(entity_ids):
        entity_scores = scores.get(entity_id, {})
        expected = truth.get(entity_id, set())
        predicted = {
            candidate_id
            for candidate_id, (_, probability) in entity_scores.items()
            if probability >= threshold
        }
        country = countries.get(entity_id, "")
        for candidate_id in sorted(predicted - expected):
            target_source, probability = entity_scores[candidate_id]
            error_type = (
                "false_positive_singleton" if not expected else "false_positive"
            )
            rows.append(
                {
                    "source1_entity_id": entity_id,
                    "country": country,
                    "error_type": error_type,
                    "candidate_entity_id": candidate_id,
                    "target_source": target_source,
                    "match_probability": format(probability, ".17g"),
                    "threshold": threshold,
                }
            )
            error_counts[error_type] += 1
            country_counts[country] += 1
            target_counts[target_source] += 1
        for candidate_id in sorted(expected - predicted):
            score = entity_scores.get(candidate_id)
            target_source = candidate_id.split("-", 1)[0]
            if score is None:
                error_type = "false_negative_candidate_miss"
                probability: float | str = ""
            else:
                error_type = "false_negative_below_threshold"
                target_source, probability_value = score
                probability = format(probability_value, ".17g")
            rows.append(
                {
                    "source1_entity_id": entity_id,
                    "country": country,
                    "error_type": error_type,
                    "candidate_entity_id": candidate_id,
                    "target_source": target_source,
                    "match_probability": probability,
                    "threshold": threshold,
                }
            )
            error_counts[error_type] += 1
            country_counts[country] += 1
            target_counts[target_source] += 1

    write_tsv(output_path, ERROR_SLICE_COLUMNS, rows)
    summary: dict[str, object] = {
        "score_file": str(score_path.resolve()),
        "error_file": str(output_path.resolve()),
        "threshold": threshold,
        "validation_entities": len(entity_ids),
        "error_rows": len(rows),
        "counts_by_error_type": dict(sorted(error_counts.items())),
        "counts_by_country": dict(
            sorted(country_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "counts_by_target_source": dict(sorted(target_counts.items())),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def country_robustness_metrics(
    score_path: Path,
    *,
    truth: Mapping[str, set[str]],
    countries: Mapping[str, str],
    entity_ids: set[str],
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    """Report held-out-fold quality by country without claiming a LOCO experiment."""

    scores = _read_detailed_scores(score_path)
    ids_by_country: dict[str, list[str]] = defaultdict(list)
    for entity_id in entity_ids:
        ids_by_country[countries.get(entity_id, "")].append(entity_id)

    result: dict[str, dict[str, float | int]] = {}
    for country, ids in sorted(ids_by_country.items()):
        macro_total = 0.0
        true_positives = false_positives = false_negatives = 0
        for entity_id in ids:
            expected = truth.get(entity_id, set())
            predicted = {
                candidate_id
                for candidate_id, (_, probability) in scores.get(entity_id, {}).items()
                if probability >= threshold
            }
            macro_total += entity_fbeta(predicted, expected, beta=0.5)
            true_positives += len(predicted & expected)
            false_positives += len(predicted - expected)
            false_negatives += len(expected - predicted)
        result[country] = {
            "entities": len(ids),
            "macro_f0_5": macro_total / len(ids),
            "micro_precision": true_positives
            / max(true_positives + false_positives, 1),
            "micro_recall": true_positives
            / max(true_positives + false_negatives, 1),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
    return result
