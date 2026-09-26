"""Phase 3 chunked generator: equivalence, resume and determinism.

The chunked path is only acceptable if it produces the same candidates as
the in-memory Day-2 generator. These tests assert exact equality of the
candidate id sequence per record, not just set equality, because the
downstream file must also be deterministic.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from business_entity_resolution.candidate_generation import (
    CandidateGenerator,
)
from business_entity_resolution.chunked_generation import (
    ChunkedTargetIndex,
    Manifest,
    count_s1_rows,
    finalize,
    iter_s1_chunks,
    load_partition,
    partition_targets_by_country,
    run_chunked,
)
from business_entity_resolution.normalization import normalize_record
from business_entity_resolution.validate_candidates import (
    check_no_label_leakage,
    check_schema,
    check_within_row_duplicates,
    validate,
)

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "business_entity_resolution"

TARGET_FILES = (
    ("test_source2.tsv", "S2"),
    ("test_source3.tsv", "S3"),
)


def _write_sources(directory: Path, s1_rows, s2_rows, s3_rows) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    for name, rows in (
        ("test_source1.tsv", s1_rows),
        ("test_source2.tsv", s2_rows),
        ("test_source3.tsv", s3_rows),
    ):
        pd.DataFrame(rows).to_csv(
            directory / name, sep="\t", index=False
        )


def _row(entity_id, name, address, country):
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
    }


SMALL_S2 = [
    _row("S2-1", "Acme Bakery", "12 Main Street", "United States"),
    _row("S2-2", "Acme Bakker", "12 Main St", "United States"),
    _row("S2-3", "Borek Shop", "9 Oak Road", "United States"),
    _row("S2-4", "Zilch", "", "United States"),
    _row("S2-5", "Boulangerie du Coin", "3 rue Lafayette", "France"),
    _row("S2-6", "Boulangerie du Coin", "3 Rue Lafayette", "France"),
]

SMALL_S3 = [
    _row("S3-1", "Northwind Trading", "88 Harbour Road", "United Kingdom"),
    _row("S3-2", "Northwind Trading Co", "88 Harbour Rd", "United Kingdom"),
    _row("S3-3", "Nilagari Tea", "5 Park Street", "India"),
]

SMALL_S1 = [
    _row("S1-1", "Acme Baker", "12 Main Street", "United States"),
    _row("S1-2", "Boulangerie du Coin", "3 rue Lafayette", "France"),
    _row("S1-3", "Northwind Trading", "88 Harbour Road", "United Kingdom"),
    _row("S1-4", "Nilagari Tea", "5 Park Street", "India"),
    _row("S1-5", "Totally Unknown", "999 Nowhere", "United States"),
]


@pytest.fixture()
def sources(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    _write_sources(
        directory, SMALL_S1, SMALL_S2, SMALL_S3
    )
    return directory


# ==============================================================
# EQUIVALENCE WITH THE IN-MEMORY GENERATOR
# ==============================================================


def test_chunked_matches_in_memory_generator(sources: Path, tmp_path: Path):
    """Every record must get the identical candidate id sequence."""

    s1_path = sources / "test_source1.tsv"

    source1_records = [
        normalize_record(row)
        for row in pd.read_csv(
            s1_path, sep="\t", dtype=str, keep_default_na=False
        ).to_dict(orient="records")
    ]

    # Every Source 1 record must appear, including those that match
    # nothing: the deliverable allows an empty candidate list.
    expected: dict[str, list[str]] = {
        record.entity_id: [] for record in source1_records
    }

    for name, _label in TARGET_FILES:
        targets = [
            normalize_record(row)
            for row in pd.read_csv(
                sources / name, sep="\t", dtype=str,
                keep_default_na=False,
            ).to_dict(orient="records")
        ]
        generator = CandidateGenerator(
            targets, blocking_backend="compact"
        )
        for source1 in source1_records:
            for candidate in generator.generate_for_one(source1):
                bucket = expected[source1.entity_id]
                if (
                    candidate.candidate_entity_id not in bucket
                ):
                    bucket.append(candidate.candidate_entity_id)

    work_root = tmp_path / "work"
    manifests = []
    work_dirs = []

    for name, label in TARGET_FILES:
        partition = work_root / f"partition_{label}"
        partition_targets_by_country(
            sources / name, partition
        )
        work_dir = work_root / label
        manifests.append(
            run_chunked(
                s1_path,
                load_partition(partition),
                label,
                work_dir,
                chunk_size=2,
            )
        )
        work_dirs.append(work_dir)

    output = tmp_path / "candidate_pairs.tsv"
    finalize(s1_path, manifests, work_dirs, output)

    observed = {
        row["source1_entity_id"]: [
            value
            for value in str(row["candidate_entity_ids"]).split(",")
            if value and value != "nan"
        ]
        for row in pd.read_csv(
            output, sep="\t", dtype=str, keep_default_na=False,
        ).to_dict(orient="records")
    }

    assert observed == expected, (
        f"chunked output differs from the in-memory generator\n"
        f"expected {expected}\nobserved {observed}"
    )


# ==============================================================
# RESUME
# ==============================================================


def test_resume_skips_completed_chunks(sources: Path, tmp_path: Path):
    s1_path = sources / "test_source1.tsv"
    partition = tmp_path / "partition_S2"
    partition_targets_by_country(
        sources / "test_source2.tsv", partition
    )
    work_dir = tmp_path / "work_S2"
    countries = load_partition(partition)

    first = run_chunked(
        s1_path, countries, "S2", work_dir, chunk_size=2
    )
    assert len(first.chunks) == 3

    # Fingerprint the chunks that must not be recomputed.
    untouched = {}
    for index in (0, 2):
        path = work_dir / first.chunks[index].output
        untouched[index] = (path.read_bytes(), path.stat().st_mtime_ns)

    # Simulate an interruption: drop the manifest entry and the output for
    # the middle chunk, keeping the first and last.
    middle = 1
    record = first.chunks[middle]
    del first.chunks[middle]
    first.save()
    (work_dir / record.output).unlink()

    resumed = run_chunked(
        s1_path, countries, "S2", work_dir, chunk_size=2
    )

    assert len(resumed.chunks) == 3
    assert resumed.is_complete(middle, work_dir)

    # Unchanged bytes and mtime prove the surviving chunks were skipped
    # rather than regenerated.
    for index, (payload, mtime) in untouched.items():
        path = work_dir / resumed.chunks[index].output
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtime


def test_manifest_requires_output_to_exist(tmp_path: Path):
    manifest = Manifest(
        path=tmp_path / "manifest.json",
        target_source="S2",
        chunk_size=10,
    )
    manifest.total_s1_rows = 30
    manifest.save()

    reloaded = Manifest.load(tmp_path / "manifest.json", "S2")
    assert reloaded.total_s1_rows == 30
    assert reloaded.completed_indices(tmp_path) == []


# ==============================================================
# DETERMINISM
# ==============================================================


def test_two_runs_produce_identical_files(
    sources: Path, tmp_path: Path
):
    s1_path = sources / "test_source1.tsv"
    partition = tmp_path / "partition_S2"
    partition_targets_by_country(
        sources / "test_source2.tsv", partition
    )
    countries = load_partition(partition)

    outputs = []
    for run in ("a", "b"):
        work_dir = tmp_path / f"work_{run}"
        manifest = run_chunked(
            s1_path, countries, "S2", work_dir, chunk_size=3
        )
        output = tmp_path / f"candidates_{run}.tsv"
        finalize(s1_path, [manifest], [work_dir], output)
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]


# ==============================================================
# PARTITIONING
# ==============================================================


def test_partition_round_trips_country_names(sources: Path, tmp_path: Path):
    """Country lookup uses the real value, not the file name."""

    partition = tmp_path / "partition_S2"
    partition_targets_by_country(
        sources / "test_source2.tsv", partition
    )
    countries = load_partition(partition)

    assert set(countries) == {"United States", "France"}


def test_count_s1_rows(sources: Path):
    assert count_s1_rows(
        sources / "test_source1.tsv"
    ) == len(SMALL_S1)


def test_iter_s1_chunks_partitions_exactly(sources: Path):
    rows = list(
        iter_s1_chunks(sources / "test_source1.tsv", 2)
    )
    assert [first for first, _ in rows] == [0, 2, 4]
    assert sum(len(block) for _, block in rows) == len(SMALL_S1)


# ==============================================================
# VALIDATION
# ==============================================================


def test_validation_passes_on_generated_output(
    sources: Path, tmp_path: Path
):
    s1_path = sources / "test_source1.tsv"
    work_root = tmp_path / "work"
    manifests = []
    work_dirs = []

    for name, label in TARGET_FILES:
        partition = work_root / f"partition_{label}"
        partition_targets_by_country(sources / name, partition)
        work_dir = work_root / label
        manifests.append(
            run_chunked(
                s1_path, load_partition(partition), label,
                work_dir, chunk_size=2,
            )
        )
        work_dirs.append(work_dir)

    output = tmp_path / "candidate_pairs.tsv"
    finalize(s1_path, manifests, work_dirs, output)

    results = validate(
        output,
        s1_path,
        sources / "test_source2.tsv",
        sources / "test_source3.tsv",
        PACKAGE,
        expected_rows=len(SMALL_S1),
    )

    for result in results:
        assert result.passed, result.render()

    assert check_schema(output).passed
    assert check_within_row_duplicates(output).passed


def test_validation_catches_foreign_candidate_id(
    sources: Path, tmp_path: Path
):
    bad = tmp_path / "bad.tsv"
    bad.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-2\tS9-999\n",
        encoding="utf-8",
    )

    results = validate(
        bad,
        sources / "test_source1.tsv",
        sources / "test_source2.tsv",
        sources / "test_source3.tsv",
        PACKAGE,
    )

    validity = [
        result for result in results
        if result.name.startswith("C.")
    ]
    assert validity and not validity[0].passed


def test_no_label_leakage_passes():
    result = check_no_label_leakage(PACKAGE)
    assert result.passed, result.detail


def test_generation_modules_do_not_import_metrics():
    """The final path must not touch the evaluation module."""

    for name in (
        "chunked_generation.py",
        "run_chunked_generation.py",
        "compact_index.py",
    ):
        text = (PACKAGE / name).read_text(encoding="utf-8")
        assert "metrics" not in text, f"{name} references metrics"
