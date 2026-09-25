"""Deterministic Day 1 candidate generation with replaceable shared contracts."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from rapidfuzz import fuzz

from business_entity_resolution.records import PreparedRecord, stable_index, write_tsv
from business_entity_resolution.schemas import (
    CANDIDATE_PAIR_COLUMNS,
    require_exact_columns,
)


_ROUTE_BASE_SCORES = {
    "exact_name_unicode": 1.00,
    "exact_name_ascii": 0.98,
    "exact_address_unicode": 0.97,
    "exact_address_ascii": 0.95,
    "name_prefix": 0.74,
    "rare_name_token": 0.62,
    "rare_address_number": 0.48,
    "country_fallback": 0.01,
}

LOGGER = logging.getLogger(__name__)


def _prefix(value: str, length: int = 6) -> str:
    return "".join(character for character in value if character.isalnum())[:length]


def _add(index: dict[tuple[str, str], list[str]], key: tuple[str, str], value: str) -> None:
    if key[1]:
        index[key].append(value)


@dataclass
class SecondaryIndex:
    label: str
    records: Mapping[str, PreparedRecord]
    exact_name_unicode: dict[tuple[str, str], list[str]]
    exact_name_ascii: dict[tuple[str, str], list[str]]
    exact_address_unicode: dict[tuple[str, str], list[str]]
    exact_address_ascii: dict[tuple[str, str], list[str]]
    name_prefix: dict[tuple[str, str], list[str]]
    rare_name_token: dict[tuple[str, str], list[str]]
    rare_address_number: dict[tuple[str, str], list[str]]
    by_country: dict[str, list[str]]

    @classmethod
    def build(
        cls,
        label: str,
        records: Mapping[str, PreparedRecord],
        *,
        max_token_bucket: int,
    ) -> "SecondaryIndex":
        token_counts: Counter[tuple[str, str]] = Counter()
        number_counts: Counter[tuple[str, str]] = Counter()
        for record in records.values():
            country = record.country
            token_counts.update((country, token) for token in set(record.normalized.name_tokens))
            number_counts.update(
                (country, number) for number in set(record.normalized.address_numbers)
            )

        exact_name_unicode: dict[tuple[str, str], list[str]] = defaultdict(list)
        exact_name_ascii: dict[tuple[str, str], list[str]] = defaultdict(list)
        exact_address_unicode: dict[tuple[str, str], list[str]] = defaultdict(list)
        exact_address_ascii: dict[tuple[str, str], list[str]] = defaultdict(list)
        name_prefix: dict[tuple[str, str], list[str]] = defaultdict(list)
        rare_name_token: dict[tuple[str, str], list[str]] = defaultdict(list)
        rare_address_number: dict[tuple[str, str], list[str]] = defaultdict(list)
        by_country: dict[str, list[str]] = defaultdict(list)

        for entity_id in sorted(records):
            record = records[entity_id]
            normalized = record.normalized
            country = record.country
            by_country[country].append(entity_id)
            _add(exact_name_unicode, (country, normalized.name_unicode), entity_id)
            _add(exact_name_ascii, (country, normalized.name_ascii), entity_id)
            _add(exact_address_unicode, (country, normalized.address_unicode), entity_id)
            _add(exact_address_ascii, (country, normalized.address_ascii), entity_id)
            prefix = _prefix(normalized.name_ascii or normalized.name_unicode)
            if len(prefix) >= 4:
                _add(name_prefix, (country, prefix), entity_id)
            for token in set(normalized.name_tokens):
                key = (country, token)
                if len(token) >= 4 and token_counts[key] <= max_token_bucket:
                    _add(rare_name_token, key, entity_id)
            for number in set(normalized.address_numbers):
                key = (country, number)
                if len(number) >= 3 and number_counts[key] <= max_token_bucket:
                    _add(rare_address_number, key, entity_id)

        return cls(
            label=label,
            records=records,
            exact_name_unicode=dict(exact_name_unicode),
            exact_name_ascii=dict(exact_name_ascii),
            exact_address_unicode=dict(exact_address_unicode),
            exact_address_ascii=dict(exact_address_ascii),
            name_prefix=dict(name_prefix),
            rare_name_token=dict(rare_name_token),
            rare_address_number=dict(rare_address_number),
            by_country=dict(by_country),
        )

    def _route_hits(self, query: PreparedRecord) -> dict[str, set[str]]:
        normalized = query.normalized
        country = query.country
        routes: dict[str, set[str]] = defaultdict(set)

        lookups: tuple[tuple[str, dict[tuple[str, str], list[str]], Iterable[str]], ...] = (
            (
                "exact_name_unicode",
                self.exact_name_unicode,
                (normalized.name_unicode,),
            ),
            ("exact_name_ascii", self.exact_name_ascii, (normalized.name_ascii,)),
            (
                "exact_address_unicode",
                self.exact_address_unicode,
                (normalized.address_unicode,),
            ),
            (
                "exact_address_ascii",
                self.exact_address_ascii,
                (normalized.address_ascii,),
            ),
            (
                "name_prefix",
                self.name_prefix,
                (_prefix(normalized.name_ascii or normalized.name_unicode),),
            ),
            ("rare_name_token", self.rare_name_token, set(normalized.name_tokens)),
            (
                "rare_address_number",
                self.rare_address_number,
                set(normalized.address_numbers),
            ),
        )
        for route, index, values in lookups:
            for value in values:
                if not value:
                    continue
                bucket = index.get((country, value), ())
                # Generic exact values such as "store" can otherwise create a
                # quadratic Day 1 job. Broad buckets are deferred to later TF-IDF
                # retrieval work instead of silently exhausting memory here.
                if len(bucket) <= 1_000:
                    routes[route].update(bucket)
        return routes

    def candidates_for(
        self,
        query: PreparedRecord,
        *,
        max_candidates: int,
        fallback_candidates: int,
        seed: int,
    ) -> list[dict[str, object]]:
        route_hits = self._route_hits(query)
        candidate_routes: dict[str, set[str]] = defaultdict(set)
        for route, entity_ids in route_hits.items():
            for entity_id in entity_ids:
                candidate_routes[entity_id].add(route)

        country_ids = self.by_country.get(query.country, [])
        if country_ids and fallback_candidates:
            start = stable_index(query.entity_id, seed=seed, size=len(country_ids))
            for offset in range(min(fallback_candidates, len(country_ids))):
                entity_id = country_ids[(start + offset) % len(country_ids)]
                candidate_routes[entity_id].add("country_fallback")

        ranked: list[tuple[float, str, str]] = []
        q = query.normalized
        for candidate_id, routes in candidate_routes.items():
            candidate = self.records[candidate_id].normalized
            name_ratio = max(
                fuzz.ratio(q.name_unicode, candidate.name_unicode),
                fuzz.ratio(q.name_ascii, candidate.name_ascii),
            ) / 100.0
            if q.address_missing or candidate.address_missing:
                address_ratio = 0.0
            else:
                address_ratio = max(
                    fuzz.ratio(q.address_unicode, candidate.address_unicode),
                    fuzz.ratio(q.address_ascii, candidate.address_ascii),
                ) / 100.0
            route_score = max(_ROUTE_BASE_SCORES[route] for route in routes)
            score = max(route_score, 0.68 * name_ratio + 0.32 * address_ratio)
            ranked.append((score, candidate_id, "+".join(sorted(routes))))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        rows: list[dict[str, object]] = []
        for rank, (score, candidate_id, routes) in enumerate(
            ranked[:max_candidates], start=1
        ):
            rows.append(
                {
                    "source1_entity_id": query.entity_id,
                    "candidate_entity_id": candidate_id,
                    "target_source": self.label,
                    "retrieval_route": routes,
                    "retrieval_rank": rank,
                    "retrieval_score": f"{score:.8f}",
                }
            )
        return rows


def iter_candidate_rows(
    source1_records: Mapping[str, PreparedRecord],
    source2_records: Mapping[str, PreparedRecord],
    source3_records: Mapping[str, PreparedRecord],
    *,
    max_candidates_per_source: int,
    fallback_candidates_per_source: int,
    max_token_bucket: int,
    seed: int,
) -> Iterator[dict[str, object]]:
    indexes = (
        SecondaryIndex.build(
            "S2", source2_records, max_token_bucket=max_token_bucket
        ),
        SecondaryIndex.build(
            "S3", source3_records, max_token_bucket=max_token_bucket
        ),
    )
    for row_number, source1_id in enumerate(sorted(source1_records), start=1):
        query = source1_records[source1_id]
        for index_offset, index in enumerate(indexes, start=2):
            yield from index.candidates_for(
                query,
                max_candidates=max_candidates_per_source,
                fallback_candidates=fallback_candidates_per_source,
                seed=seed + index_offset,
            )
        if row_number % 5_000 == 0:
            LOGGER.info(
                "candidate_generation_progress | source1_processed=%d total=%d",
                row_number,
                len(source1_records),
            )


def write_candidate_rows(
    path: Path,
    source1_records: Mapping[str, PreparedRecord],
    source2_records: Mapping[str, PreparedRecord],
    source3_records: Mapping[str, PreparedRecord],
    *,
    max_candidates_per_source: int,
    fallback_candidates_per_source: int,
    max_token_bucket: int,
    seed: int,
) -> int:
    return write_tsv(
        path,
        CANDIDATE_PAIR_COLUMNS,
        iter_candidate_rows(
            source1_records,
            source2_records,
            source3_records,
            max_candidates_per_source=max_candidates_per_source,
            fallback_candidates_per_source=fallback_candidates_per_source,
            max_token_bucket=max_token_bucket,
            seed=seed,
        ),
    )


def add_training_positive_guards(
    raw_candidate_path: Path,
    output_path: Path,
    *,
    truth: Mapping[str, set[str]],
    folds: Mapping[str, int],
    validation_fold: int,
    secondary_ids: set[str],
) -> tuple[int, int]:
    """Copy heuristic candidates and add missing positives only to training folds."""

    seen: set[tuple[str, str]] = set()
    guard_counter = [0]

    def rows():
        with raw_candidate_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require_exact_columns(
                reader.fieldnames, CANDIDATE_PAIR_COLUMNS, label=str(raw_candidate_path)
            )
            for row in reader:
                key = (row["source1_entity_id"], row["candidate_entity_id"])
                if key in seen:
                    continue
                seen.add(key)
                yield row

        for source1_id in sorted(truth):
            if folds[source1_id] == validation_fold:
                continue
            for candidate_id in sorted(truth[source1_id]):
                key = (source1_id, candidate_id)
                if key in seen:
                    continue
                if candidate_id not in secondary_ids:
                    raise ValueError(
                        f"truth references missing secondary ID {candidate_id}"
                    )
                yield {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate_id,
                    "target_source": candidate_id.split("-", 1)[0],
                    "retrieval_route": "train_positive_guard",
                    "retrieval_rank": 0,
                    "retrieval_score": "0.0",
                }
                seen.add(key)
                guard_counter[0] += 1

    row_count = write_tsv(output_path, CANDIDATE_PAIR_COLUMNS, rows())
    return row_count, guard_counter[0]


def candidate_recall(
    candidate_path: Path,
    *,
    truth: Mapping[str, set[str]],
    entity_ids: set[str],
) -> dict[str, float | int]:
    predicted: dict[str, set[str]] = defaultdict(set)
    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(reader.fieldnames, CANDIDATE_PAIR_COLUMNS, label=str(candidate_path))
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id in entity_ids:
                predicted[source1_id].add(row["candidate_entity_id"])

    true_links = 0
    covered_links = 0
    entities_with_truth = 0
    entities_fully_covered = 0
    for source1_id in entity_ids:
        expected = truth.get(source1_id, set())
        if not expected:
            continue
        entities_with_truth += 1
        true_links += len(expected)
        covered = expected & predicted.get(source1_id, set())
        covered_links += len(covered)
        if covered == expected:
            entities_fully_covered += 1
    return {
        "true_links": true_links,
        "covered_links": covered_links,
        "link_recall": covered_links / true_links if true_links else 1.0,
        "entities_with_truth": entities_with_truth,
        "entities_fully_covered": entities_fully_covered,
        "entity_full_recall": (
            entities_fully_covered / entities_with_truth if entities_with_truth else 1.0
        ),
    }
