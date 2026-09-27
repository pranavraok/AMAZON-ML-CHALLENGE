"""Restartable, chunked candidate generation for the full test sources.

The full test set is roughly 1.73M Source 1 records against 4.89M Source 2
and 5.08M Source 3 records. Holding all of that as Python objects, or
scoring character similarity against a single global matrix, does not fit
in memory.

This module keeps the Day-2 retrieval semantics exactly and changes only
how the work is scheduled:

  - target sources are partitioned to disk by country, then indexed one
    country at a time, so the normalized records are released as soon as
    they are indexed. Every blocking family and both TF-IDF retrievers are
    already country-partitioned, so this changes no candidate.
  - Source 1 is streamed in chunks and never fully materialised.
  - each finished chunk is written atomically and recorded in a manifest,
    so an interrupted run resumes by skipping completed chunks.
  - S2 and S3 are processed independently and merged only at the end.

The output of a chunk is the same candidate list the in-memory
``CandidateGenerator`` produces for those records, in the same
deterministic order.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from .candidate_generation import CandidateGenerationConfig
from .compact_index import CompactBlockingIndex
from .normalization import normalize_record
from .retrieval import CharacterTfidfRetriever

CHUNK_HEADER = "source1_entity_id\tcandidate_entity_ids"
FINAL_HEADER = CHUNK_HEADER

DEFAULT_CHUNK_SIZE = 50_000

_REQUIRED_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)


# ==============================================================
# MEMORY
# ==============================================================


class PeakMemoryMonitor:
    """Sample process RSS on a background thread.

    ``tracemalloc`` only sees Python allocations, so it under-reports the
    numpy arrays and scipy sparse matrices that dominate this workload.
    RSS is the number that actually matters for fitting in memory, so it
    is sampled for real instead of estimated.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.peak_bytes = 0
        self._stop = None
        self._thread = None

    def __enter__(self) -> "PeakMemoryMonitor":
        import threading

        import psutil

        process = psutil.Process()

        self.peak_bytes = process.memory_info().rss
        self._stop = threading.Event()

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    rss = process.memory_info().rss
                except Exception:
                    return
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

        return self

    def __exit__(self, *exception) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / 1024 ** 3


# ==============================================================
# MANIFEST
# ==============================================================


@dataclass
class ChunkRecord:
    index: int
    first_row: int
    row_count: int
    output: str
    candidate_pairs: int
    s1_with_candidates: int
    seconds: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "first_row": self.first_row,
            "row_count": self.row_count,
            "output": self.output,
            "candidate_pairs": self.candidate_pairs,
            "s1_with_candidates": self.s1_with_candidates,
            "seconds": round(self.seconds, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkRecord":
        return cls(
            index=int(data["index"]),
            first_row=int(data["first_row"]),
            row_count=int(data["row_count"]),
            output=str(data["output"]),
            candidate_pairs=int(data["candidate_pairs"]),
            s1_with_candidates=int(data["s1_with_candidates"]),
            seconds=float(data["seconds"]),
        )


@dataclass
class Manifest:
    """Completion record for one target source."""

    path: Path
    target_source: str
    chunk_size: int
    total_s1_rows: int = 0
    chunks: dict[int, ChunkRecord] = field(default_factory=dict)

    # ----------------------------------------------------------
    # PERSISTENCE
    # ----------------------------------------------------------

    def save(self) -> None:
        """Write the manifest atomically.

        A crash mid-write must not leave a manifest that claims a chunk is
        complete when its output is missing, so the manifest is only ever
        replaced by a complete file.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "target_source": self.target_source,
            "chunk_size": self.chunk_size,
            "total_s1_rows": self.total_s1_rows,
            "chunks": [
                self.chunks[index].to_dict()
                for index in sorted(self.chunks)
            ],
        }

        temporary = self.path.with_suffix(".json.tmp")

        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        os.replace(temporary, self.path)

    @classmethod
    def load(cls, path: Path, target_source: str) -> "Manifest":
        manifest = cls(
            path=path,
            target_source=target_source,
            chunk_size=DEFAULT_CHUNK_SIZE,
        )

        if not path.exists():
            return manifest

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        manifest.chunk_size = int(payload["chunk_size"])
        manifest.total_s1_rows = int(payload["total_s1_rows"])

        for entry in payload["chunks"]:
            record = ChunkRecord.from_dict(entry)
            manifest.chunks[record.index] = record

        return manifest

    # ----------------------------------------------------------
    # COMPLETION
    # ----------------------------------------------------------

    def is_complete(self, index: int, work_dir: Path) -> bool:
        """A chunk counts as done only if its output is still on disk."""

        record = self.chunks.get(index)

        if record is None:
            return False

        return (work_dir / record.output).exists()

    def completed_indices(self, work_dir: Path) -> list[int]:
        return sorted(
            index
            for index in self.chunks
            if self.is_complete(index, work_dir)
        )

    def record_chunk(self, record: ChunkRecord) -> None:
        self.chunks[record.index] = record


# ==============================================================
# STREAMING READERS
# ==============================================================


def count_s1_rows(path: Path) -> int:
    """Row count excluding the header, without loading the file."""

    total = 0

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += block.count(b"\n")

    # A trailing newline produces a phantom final line.
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size:
            handle.seek(size - 1)
            if handle.read(1) != b"\n":
                total += 1

    return max(0, total - 1)


def iter_s1_chunks(
    path: Path,
    chunk_size: int,
) -> Iterator[tuple[int, list]]:
    """Yield ``(first_row, records)`` for successive Source 1 chunks.

    Only one chunk of normalized records exists at a time. Chunk
    boundaries are fixed by row position, not by content, so a resumed run
    recomputes exactly the same chunks.
    """

    first_row = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )

    for block in reader:

        records = [
            normalize_record(row)
            for row in block.to_dict(orient="records")
        ]

        yield first_row, records

        first_row += len(records)


def _s1_order(path: Path) -> list[str]:
    """Source 1 entity ids in file order, for the final row layout.

    This is ids only, not records, so it is cheap even for 1.73M rows.
    """

    order: list[str] = []

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id"],
        chunksize=200_000,
    )

    for block in reader:
        order.extend(block["entity_id"].tolist())

    return order


def partition_targets_by_country(
    source_path: Path,
    destination: Path,
) -> dict[str, Path]:
    """Split a target source into one file per country.

    Every blocking family and both TF-IDF retrievers key on country, so a
    country can be indexed and released on its own without changing any
    candidate. This partition is what lets the normalized records be
    streamed instead of held.
    """

    destination.mkdir(parents=True, exist_ok=True)

    handles: dict[str, object] = {}
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}

    try:
        reader = pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=200_000,
        )

        for block in reader:

            missing = set(_REQUIRED_COLUMNS) - set(block.columns)
            if missing:
                raise ValueError(
                    f"{source_path} is missing required columns: "
                    f"{sorted(missing)}"
                )

            for country, group in block.groupby("country", sort=False):
                country = str(country).strip() or "__unknown__"

                if country not in handles:
                    target = destination / f"{_safe(country)}.tsv"
                    paths[country] = target
                    handle = open(
                        target, "w", encoding="utf-8", newline=""
                    )
                    handle.write("\t".join(block.columns) + "\n")
                    handles[country] = handle
                    counts[country] = 0

                group.to_csv(
                    handles[country],
                    sep="\t",
                    index=False,
                    header=False,
                    lineterminator="\n",
                )
                counts[country] += len(group)

    finally:
        for handle in handles.values():
            handle.close()

    for country, path in paths.items():
        print(
            f"  partition {country:<28} "
            f"{counts[country]:>10,} rows"
        )

    # The file name is a lossy transform of the country, so the original
    # spelling is recorded. Country lookup at query time uses the real
    # value, and a reused partition must resolve to the same key.
    with open(destination / "partition_map.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                country: path.name
                for country, path in paths.items()
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    return paths


def load_partition(destination: Path) -> dict[str, Path]:
    """Read a partition written by :func:`partition_targets_by_country`."""

    map_path = destination / "partition_map.json"

    if not map_path.exists():
        raise FileNotFoundError(
            f"{map_path} is missing; rerun without --skip-partition"
        )

    with open(map_path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    resolved = {
        country: destination / filename
        for country, filename in mapping.items()
    }

    missing = [
        path for path in resolved.values() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"partition files are missing, first {missing[0]}"
        )

    return resolved


def _safe(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )[:64] or "unnamed"


def iter_country_records(path: Path) -> Iterator:
    """Stream one country's records, one block at a time."""

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=50_000,
    )

    for block in reader:
        for row in block.to_dict(orient="records"):
            yield normalize_record(row)


# ==============================================================
# TARGET INDEX
# ==============================================================


class ChunkedTargetIndex:
    """Day-2 candidate generation over country-partitioned targets.

    The blocking index is built by streaming every country's records
    through a single incremental build, so no more than one country's
    records are alive at once. The TF-IDF retrievers are already
    country-partitioned and are populated the same way.
    """

    def __init__(
        self,
        country_paths: dict[str, Path],
        target_source: str,
        config: CandidateGenerationConfig | None = None,
        *,
        index_dir: Path | None = None,
        reuse_index: bool = False,
        cache_tfidf: bool = False,
        verify_no_hash_collisions: bool = False,
    ) -> None:

        if not country_paths:
            raise ValueError(
                "ChunkedTargetIndex requires at least one country"
            )

        self.config = config or CandidateGenerationConfig()
        self.target_source = target_source

        ordered = sorted(country_paths.items())

        # ---- blocking index, streamed over all countries ----
        #
        # A full test target source costs roughly 25 minutes of CPU to index,
        # so the finished index is cached on disk and memory-mapped back on
        # every later run. Nothing about the stored bytes changes a
        # candidate; they are the same hashes, indptr and postings.

        metadata = (
            index_dir / CompactBlockingIndex.INDEX_METADATA
            if index_dir is not None
            else None
        )
        cached = (
            reuse_index
            and metadata is not None
            and metadata.is_file()
        )

        if cached:
            print(f"  reusing cached blocking index {index_dir}...")
            started = time.perf_counter()
            self.blocking_index = CompactBlockingIndex.load(
                index_dir, mmap_mode="r"
            )
            print(
                f"  mapped blocking index: "
                f"{len(self.blocking_index.entity_ids):,} targets in "
                f"{time.perf_counter() - started:.1f}s"
            )
        else:
            print(f"  building blocking index for {target_source}...")

            started = time.perf_counter()

            self.blocking_index = CompactBlockingIndex.from_stream(
                self._all_records(ordered),
                rare_token_max_postings=self.config.rare_token_max_postings,
                numeric_token_max_postings=(
                    self.config.numeric_token_max_postings
                ),
                translit_token_max_postings=(
                    self.config.translit_token_max_postings
                ),
                address_token_max_postings=(
                    self.config.address_token_max_postings
                ),
                address_pair_max_postings=(
                    self.config.address_pair_max_postings
                ),
                verify_no_hash_collisions=verify_no_hash_collisions,
            )

            print(
                f"  blocking index: "
                f"{len(self.blocking_index.entity_ids):,} targets in "
                f"{time.perf_counter() - started:.1f}s"
            )

            if index_dir is not None:
                started = time.perf_counter()
                self.blocking_index.save(index_dir)
                print(
                    f"  cached blocking index to {index_dir} in "
                    f"{time.perf_counter() - started:.1f}s"
                )

        # ---- TF-IDF retrievers, one country at a time ----

        self.name_retrievers: dict[str, CharacterTfidfRetriever] = {}
        self.address_retrievers: dict[str, CharacterTfidfRetriever] = {}

        for country, path in ordered:

            records = list(iter_country_records(path))

            if not records:
                continue

            for field, retrievers, top_k, min_score in (
                (
                    "name",
                    self.name_retrievers,
                    self.config.name_top_k,
                    self.config.name_min_score,
                ),
                (
                    "address",
                    self.address_retrievers,
                    self.config.address_top_k,
                    self.config.address_min_score,
                ),
            ):
                retrievers[country] = self._retriever(
                    field,
                    records,
                    top_k,
                    min_score,
                    country,
                    path,
                    index_dir if cache_tfidf else None,
                )

            # The retrievers keep entity ids and transposed matrices only.
            del records

    def _retriever(
        self,
        field: str,
        records: list,
        top_k: int,
        min_score: float,
        country: str,
        path: Path,
        cache_dir: Path | None,
    ) -> CharacterTfidfRetriever:
        """Build one TF-IDF retriever, reusing a cached one when possible.

        The cache key covers the fitted parameters and the size of the
        country partition, so a config change or different input data
        rebuilds instead of silently reusing a stale matrix.
        """

        if cache_dir is None:
            return CharacterTfidfRetriever(
                records,
                field=field,
                ngram_range=self.config.ngram_range,
                top_k=top_k,
                min_score=min_score,
                target_source=self.target_source,
            )

        import pickle

        key = "|".join(
            str(part)
            for part in (
                self.target_source,
                country,
                field,
                self.config.ngram_range,
                top_k,
                min_score,
                path.stat().st_size,
            )
        )
        fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        cache_file = cache_dir / f"tfidf_{fingerprint}.pkl"

        if cache_file.is_file():
            try:
                with open(cache_file, "rb") as handle:
                    retriever = pickle.load(handle)
                print(
                    f"  reusing cached {field} tfidf for "
                    f"{country} ({cache_file.name})"
                )
                return retriever
            except Exception as error:  # noqa: BLE001
                print(
                    f"  ignoring unusable {field} tfidf cache "
                    f"{cache_file.name}: {error}"
                )

        started = time.perf_counter()
        retriever = CharacterTfidfRetriever(
            records,
            field=field,
            ngram_range=self.config.ngram_range,
            top_k=top_k,
            min_score=min_score,
            target_source=self.target_source,
        )
        print(
            f"  {field} tfidf for {country}: {len(records):,} records in "
            f"{time.perf_counter() - started:.1f}s"
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".pkl.tmp")
        try:
            with open(temporary, "wb") as handle:
                pickle.dump(retriever, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, cache_file)
        except Exception as error:  # noqa: BLE001
            print(f"  could not cache {field} tfidf: {error}")
            temporary.unlink(missing_ok=True)

        return retriever

    @staticmethod
    def _all_records(ordered: Sequence[tuple[str, Path]]) -> Iterator:
        for country, path in ordered:
            for record in iter_country_records(path):
                yield record

    # ----------------------------------------------------------
    # QUERY
    # ----------------------------------------------------------

    def generate_for_one(self, source1) -> list[str]:
        """Candidate entity ids for one Source 1 record.

        Mirrors ``CandidateGenerator.generate_for_one`` ordering: the
        evidence sort is keyed on route count, retrieval score, retrieval
        rank and finally entity id, so the order is deterministic and
        independent of the order records were streamed in.
        """

        from .blocking import CandidateEvidence

        evidence: dict[str, CandidateEvidence] = {}

        for evidence_item in self.blocking_index.generate(source1):
            evidence[evidence_item.candidate_entity_id] = evidence_item

        country = _country_of(source1)
        target_source = self.target_source

        for field, retrievers in (
            ("name", self.name_retrievers),
            ("address", self.address_retrievers),
        ):
            retriever = retrievers.get(country)
            if retriever is None:
                continue

            for result in retriever.retrieve(source1):

                candidate_id = result.candidate_entity_id

                candidate = evidence.get(candidate_id)

                if candidate is None:
                    candidate = CandidateEvidence(
                        source1_entity_id=source1.entity_id,
                        candidate_entity_id=candidate_id,
                        target_source=target_source,
                    )
                    evidence[candidate_id] = candidate

                candidate.add(
                    result.route,
                    score=result.score,
                    rank=result.rank,
                )

        candidates = list(evidence.values())

        candidates.sort(
            key=lambda candidate: (
                -len(candidate.routes),
                -candidate.retrieval_score,
                candidate.retrieval_rank,
                candidate.candidate_entity_id,
            )
        )

        return [
            candidate.candidate_entity_id for candidate in candidates
        ]


def _country_of(record) -> str:
    country = getattr(record, "country_code", None)
    if country:
        return country
    raw = getattr(record, "country", "") or ""
    return str(raw).strip() or "__unknown__"


# ==============================================================
# CHUNK EXECUTION
# ==============================================================


def run_chunked(
    s1_path: Path,
    country_paths: dict[str, Path],
    target_source: str,
    work_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    config: CandidateGenerationConfig | None = None,
    limit: int | None = None,
    progress_every: int = 1,
    index_dir: Path | None = None,
    reuse_index: bool = False,
    cache_tfidf: bool = False,
) -> Manifest:
    """Generate and persist every chunk for one target source."""

    work_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = work_dir / "manifest.json"
    manifest = Manifest.load(manifest_path, target_source)
    manifest.chunk_size = chunk_size

    total_rows = count_s1_rows(s1_path)

    if limit is not None:
        total_rows = min(total_rows, limit)

    manifest.total_s1_rows = total_rows
    manifest.save()

    done = manifest.completed_indices(work_dir)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    if done:
        print(
            f"  resuming: {len(done)}/{total_chunks} chunks already done"
        )

    if len(done) < total_chunks:
        print(f"  building target index for {target_source}...")
        index = ChunkedTargetIndex(
            country_paths,
            target_source,
            config=config,
            index_dir=index_dir,
            reuse_index=reuse_index,
            cache_tfidf=cache_tfidf,
        )
    else:
        index = None

    for chunk_index, (first_row, records) in enumerate(
        iter_s1_chunks(s1_path, chunk_size)
    ):

        if first_row >= total_rows:
            break

        if len(records) > total_rows - first_row:
            records = records[: total_rows - first_row]

        if manifest.is_complete(chunk_index, work_dir):
            if progress_every and chunk_index % progress_every == 0:
                print(
                    f"    chunk {chunk_index:>5} skipped "
                    f"(already complete)"
                )
            continue

        if index is None:
            index = ChunkedTargetIndex(
                country_paths,
                target_source,
                config=config,
                index_dir=index_dir,
                reuse_index=reuse_index,
                cache_tfidf=cache_tfidf,
            )

        relative = Path(f"chunk_{chunk_index:06d}.tsv")
        destination = work_dir / relative
        temporary = destination.with_suffix(".tsv.tmp")

        started = time.perf_counter()

        pair_count = 0
        with_candidates = 0

        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            handle.write(CHUNK_HEADER + "\n")

            for record in records:
                candidates = index.generate_for_one(record)
                pair_count += len(candidates)
                if candidates:
                    with_candidates += 1
                handle.write(
                    record.entity_id
                    + "\t"
                    + ",".join(candidates)
                    + "\n"
                )

        # The chunk is only renamed into place once it is fully written,
        # so a partial file can never be mistaken for a finished one.
        os.replace(temporary, destination)

        elapsed = time.perf_counter() - started

        manifest.record_chunk(
            ChunkRecord(
                index=chunk_index,
                first_row=first_row,
                row_count=len(records),
                output=str(relative).replace("\\", "/"),
                candidate_pairs=pair_count,
                s1_with_candidates=with_candidates,
                seconds=elapsed,
            )
        )
        manifest.save()

        print(
            f"    chunk {chunk_index:>5}  "
            f"rows {len(records):>7,}  "
            f"pairs {pair_count:>10,}  "
            f"{elapsed:>6.1f}s"
        )

    return manifest


# ==============================================================
# FINALIZE
# ==============================================================


def finalize(
    s1_path: Path,
    manifests: Sequence[Manifest],
    work_dirs: Sequence[Path],
    destination: Path,
    limit: int | None = None,
) -> dict:
    """Merge per-target chunk files into the single final deliverable.

    Source 1 order is preserved. A Source 1 record with no candidates gets
    an empty field. Candidate ids are unioned across S2 and S3, deduped,
    and kept in first-seen order.

    ``limit`` restricts the deliverable to the first N Source 1 records,
    which is what a bounded dry run produces. The full run passes None.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    # candidate id -> ordered unique list, per Source 1 record
    candidates: dict[str, list[str]] = {}

    # Source 1 record -> which target sources produced a row for it. S2 and
    # S3 each cover every record, so overlap between sources is expected;
    # overlap within one source would mean a chunk was written twice.
    covered_by: dict[str, set[str]] = {}

    for manifest, work_dir in zip(manifests, work_dirs):

        for chunk_index in sorted(manifest.chunks):

            record = manifest.chunks[chunk_index]
            path = work_dir / record.output

            if not path.exists():
                raise FileNotFoundError(
                    f"missing chunk output {path}; rerun to regenerate"
                )

            with open(path, "r", encoding="utf-8") as handle:
                header = handle.readline().rstrip("\n")
                if header != CHUNK_HEADER:
                    raise ValueError(
                        f"{path} has unexpected header {header!r}"
                    )

                for line in handle:
                    line = line.rstrip("\n")
                    if not line:
                        continue

                    entity_id, _, payload = line.partition("\t")

                    sources = covered_by.setdefault(entity_id, set())
                    if manifest.target_source in sources:
                        raise ValueError(
                            f"{entity_id} appears in more than one "
                            f"{manifest.target_source} chunk"
                        )
                    sources.add(manifest.target_source)

                    for candidate_id in payload.split(","):
                        if not candidate_id:
                            continue
                        bucket = candidates.get(entity_id)
                        if bucket is None:
                            candidates[entity_id] = [candidate_id]
                        elif candidate_id not in bucket:
                            bucket.append(candidate_id)

    expected_order = _s1_order(s1_path)

    if limit is not None:
        expected_order = expected_order[:limit]

    expected_set = set(expected_order)

    missing = [
        entity_id
        for entity_id in expected_order
        if entity_id not in covered_by
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} Source 1 records have no chunk output, "
            f"first {missing[:5]}"
        )

    unexpected = set(covered_by) - expected_set
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} chunk rows are not Source 1 records, "
            f"first {sorted(unexpected)[:5]}"
        )

    total_pairs = 0
    empty_rows = 0

    with open(destination, "w", encoding="utf-8", newline="") as handle:
        handle.write(FINAL_HEADER + "\n")

        for entity_id in expected_order:
            found = candidates.get(entity_id) or []
            total_pairs += len(found)
            if not found:
                empty_rows += 1
            handle.write(entity_id + "\t" + ",".join(found) + "\n")

    return {
        "path": str(destination),
        "rows": len(expected_order),
        "candidate_pairs": total_pairs,
        "empty_rows": empty_rows,
    }
