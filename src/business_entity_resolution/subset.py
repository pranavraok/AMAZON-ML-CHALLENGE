"""Deterministic, streaming development-subset creation."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping

from business_entity_resolution.config import AppConfig
from business_entity_resolution.schemas import (
    GROUND_TRUTH_COLUMNS,
    SOURCE_COLUMNS,
    require_exact_columns,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceSubsetStats:
    source: str
    positive_rows: int
    negative_rows: int
    total_rows: int
    countries: dict[str, int]


@dataclass(frozen=True)
class DevelopmentSubsetSummary:
    seed: int
    requested_source1_count: int
    selected_source1_count: int
    negative_multiplier_per_source: float
    source1_countries: dict[str, int]
    ground_truth_positive_pairs: int
    source2: SourceSubsetStats
    source3: SourceSubsetStats


def _stable_score(seed: int, value: str) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _keep_smallest(
    heap: list[tuple[int, str, dict[str, str]]],
    *,
    limit: int,
    score: int,
    entity_id: str,
    row: Mapping[str, str],
) -> None:
    if limit <= 0:
        return
    item = (-score, entity_id, dict(row))
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    worst_score = -heap[0][0]
    if score < worst_score:
        heapq.heapreplace(heap, item)


def _priority_sample_source1(
    source1_path: Path,
    *,
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    heap: list[tuple[int, str, dict[str, str]]] = []
    with source1_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(reader.fieldnames, SOURCE_COLUMNS, label=str(source1_path))
        for row_number, row in enumerate(reader, start=1):
            entity_id = row["entity_id"]
            _keep_smallest(
                heap,
                limit=count,
                score=_stable_score(seed, entity_id),
                entity_id=entity_id,
                row=row,
            )
            if row_number % 1_000_000 == 0:
                LOGGER.info("Scanned %s Source 1 records", f"{row_number:,}")
    return sorted((item[2] for item in heap), key=lambda row: row["entity_id"])


def _write_tsv(
    path: Path,
    columns: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _selected_ground_truth(
    ground_truth_path: Path,
    selected_s1_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, set[str]], int]:
    selected_rows: list[dict[str, str]] = []
    positive_ids: dict[str, set[str]] = {"S2": set(), "S3": set()}
    positive_pairs = 0
    with ground_truth_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(
            reader.fieldnames,
            GROUND_TRUTH_COLUMNS,
            label=str(ground_truth_path),
        )
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id not in selected_s1_ids:
                continue
            selected_rows.append(row)
            for match_id in (
                item for item in row["matched_entity_ids"].split(",") if item
            ):
                prefix = match_id.split("-", 1)[0]
                if prefix not in positive_ids:
                    raise ValueError(
                        f"ground truth for {source1_id} contains invalid ID {match_id}"
                    )
                positive_ids[prefix].add(match_id)
                positive_pairs += 1

    selected_rows.sort(key=lambda row: row["source1_entity_id"])
    found_ids = {row["source1_entity_id"] for row in selected_rows}
    missing = selected_s1_ids - found_ids
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"ground truth is missing {len(missing)} selected S1 IDs; examples: {examples}"
        )
    return selected_rows, positive_ids, positive_pairs


def _allocate_negative_quotas(
    source1_countries: Counter[str],
    multiplier: float,
) -> dict[str, int]:
    return {
        country: int(round(count * multiplier))
        for country, count in source1_countries.items()
    }


def _write_secondary_subset(
    source_path: Path,
    output_path: Path,
    *,
    source_label: str,
    positive_ids: set[str],
    negative_quotas: Mapping[str, int],
    seed: int,
) -> SourceSubsetStats:
    negative_heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = defaultdict(list)
    positive_rows = 0
    countries: Counter[str] = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8", newline="") as source_handle, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle, delimiter="\t")
        require_exact_columns(reader.fieldnames, SOURCE_COLUMNS, label=str(source_path))
        writer = csv.DictWriter(
            output_handle,
            fieldnames=list(SOURCE_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for row_number, row in enumerate(reader, start=1):
            entity_id = row["entity_id"]
            country = row["country"]
            if entity_id in positive_ids:
                writer.writerow(row)
                positive_rows += 1
                countries[country] += 1
            elif country in negative_quotas:
                _keep_smallest(
                    negative_heaps[country],
                    limit=negative_quotas[country],
                    score=_stable_score(seed, entity_id),
                    entity_id=entity_id,
                    row=row,
                )
            if row_number % 1_000_000 == 0:
                LOGGER.info(
                    "Scanned %s %s records", f"{row_number:,}", source_label
                )

        sampled_negatives: list[dict[str, str]] = []
        for country, heap in negative_heaps.items():
            rows = [item[2] for item in heap]
            rows.sort(key=lambda row: row["entity_id"])
            sampled_negatives.extend(rows)
            countries[country] += len(rows)
        sampled_negatives.sort(key=lambda row: (row["country"], row["entity_id"]))
        writer.writerows(sampled_negatives)

    negative_rows = sum(len(heap) for heap in negative_heaps.values())
    found_positive_ids = set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["entity_id"] in positive_ids:
                found_positive_ids.add(row["entity_id"])
    missing_positives = positive_ids - found_positive_ids
    if missing_positives:
        examples = sorted(missing_positives)[:5]
        raise ValueError(
            f"{source_label} is missing {len(missing_positives)} labelled records; "
            f"examples: {examples}"
        )

    return SourceSubsetStats(
        source=source_label,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        total_rows=positive_rows + negative_rows,
        countries=dict(sorted(countries.items())),
    )


def create_development_subset(
    config: AppConfig,
    *,
    output_dir: Path | None = None,
    source1_count: int | None = None,
    negative_multiplier: float | None = None,
    overwrite: bool = False,
) -> DevelopmentSubsetSummary:
    """Create a deterministic, schema-compatible training subset.

    All true secondary matches for selected S1 records are retained. Additional
    secondary distractors are sampled within the selected S1 countries.
    """

    missing_inputs = [
        path
        for path in (
            config.paths.train_source1,
            config.paths.train_source2,
            config.paths.train_source3,
            config.paths.train_ground_truth,
        )
        if not path.is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing training input files: " + ", ".join(str(path) for path in missing_inputs)
        )

    requested_count = source1_count or config.development_subset.source1_count
    multiplier = (
        config.development_subset.negative_multiplier_per_source
        if negative_multiplier is None
        else negative_multiplier
    )
    if requested_count <= 0:
        raise ValueError("source1_count must be positive")
    if multiplier < 0:
        raise ValueError("negative_multiplier cannot be negative")

    final_output = (output_dir or config.paths.development_dir).resolve()
    if final_output.exists() and not overwrite:
        raise FileExistsError(
            f"Development subset already exists: {final_output}. Use --overwrite to replace it."
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_output.name}-", dir=final_output.parent)
    )
    train_output = staging / "train"

    try:
        selected_s1 = _priority_sample_source1(
            config.paths.train_source1,
            count=requested_count,
            seed=config.project.seed,
        )
        if not selected_s1:
            raise ValueError("No Source 1 records were selected")
        selected_ids = {row["entity_id"] for row in selected_s1}
        source1_countries = Counter(row["country"] for row in selected_s1)
        ground_truth_rows, positive_ids, positive_pairs = _selected_ground_truth(
            config.paths.train_ground_truth,
            selected_ids,
        )

        _write_tsv(train_output / "train_source1.tsv", SOURCE_COLUMNS, selected_s1)
        _write_tsv(
            train_output / "train_ground_truth.tsv",
            GROUND_TRUTH_COLUMNS,
            ground_truth_rows,
        )

        quotas = _allocate_negative_quotas(source1_countries, multiplier)
        source2_stats = _write_secondary_subset(
            config.paths.train_source2,
            train_output / "train_source2.tsv",
            source_label="S2",
            positive_ids=positive_ids["S2"],
            negative_quotas=quotas,
            seed=config.project.seed + 2,
        )
        source3_stats = _write_secondary_subset(
            config.paths.train_source3,
            train_output / "train_source3.tsv",
            source_label="S3",
            positive_ids=positive_ids["S3"],
            negative_quotas=quotas,
            seed=config.project.seed + 3,
        )

        summary = DevelopmentSubsetSummary(
            seed=config.project.seed,
            requested_source1_count=requested_count,
            selected_source1_count=len(selected_s1),
            negative_multiplier_per_source=multiplier,
            source1_countries=dict(sorted(source1_countries.items())),
            ground_truth_positive_pairs=positive_pairs,
            source2=source2_stats,
            source3=source3_stats,
        )
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(summary), handle, indent=2, sort_keys=True)
            handle.write("\n")

        if final_output.exists():
            shutil.rmtree(final_output)
        staging.replace(final_output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
