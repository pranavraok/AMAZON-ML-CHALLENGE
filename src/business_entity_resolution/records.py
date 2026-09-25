"""Small, explicit TSV helpers shared by the Day 1 baseline pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from business_entity_resolution.normalization import NormalizedRecord, normalize_record
from business_entity_resolution.schemas import (
    GROUND_TRUTH_COLUMNS,
    SOURCE_COLUMNS,
    require_exact_columns,
)


@dataclass(frozen=True)
class PreparedRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str
    normalized: NormalizedRecord


def read_source_records(path: Path) -> dict[str, PreparedRecord]:
    records: dict[str, PreparedRecord] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(reader.fieldnames, SOURCE_COLUMNS, label=str(path))
        for row in reader:
            normalized = normalize_record(row)
            entity_id = normalized.entity_id
            if entity_id in records:
                raise ValueError(f"{path} contains duplicate entity ID {entity_id}")
            records[entity_id] = PreparedRecord(
                entity_id=entity_id,
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"],
                normalized=normalized,
            )
    return records


def read_ground_truth(path: Path) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(reader.fieldnames, GROUND_TRUTH_COLUMNS, label=str(path))
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id in truth:
                raise ValueError(f"{path} contains duplicate Source 1 ID {source1_id}")
            truth[source1_id] = {
                item for item in row["matched_entity_ids"].split(",") if item
            }
    return truth


def write_tsv(
    path: Path,
    columns: Sequence[str],
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
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return list(reader.fieldnames), list(reader)


def stable_fold(entity_id: str, *, seed: int, n_folds: int) -> int:
    payload = f"{seed}:{entity_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value % n_folds


def stable_index(entity_id: str, *, seed: int, size: int) -> int:
    if size <= 0:
        raise ValueError("size must be positive")
    payload = f"{seed}:{entity_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value % size
