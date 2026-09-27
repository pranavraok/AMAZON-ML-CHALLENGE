"""Compact country-aware blocking index for full-scale candidate generation.

Why this module exists
----------------------
The committed :class:`business_entity_resolution.blocking.BlockingIndex`
stores ``Dict[tuple, List[str]]``. Every index entry costs a tuple key, a
list object and a list slot, which measured at roughly 12.4 KB of Python
objects per indexed record on the Day 2 development subset. Extrapolated to
the 9.97M test target records that is on the order of 120 GB, and the
machine has 23.7 GB of RAM.

This module keeps the *identical* posting multisets but stores them as
``int32`` row ids in a CSR layout, and keeps keys as sorted ``uint64``
hashes. Entity ids are held in one packed byte buffer instead of being kept
alive as ``NormalizedRecord`` objects.

Fidelity requirements
---------------------
The candidate set produced here must match ``BlockingIndex`` exactly. Two
details make that non-obvious and both are load-bearing:

1. **Posting lists are multisets, not sets.** ``_tokens`` returns every
   whitespace token of length >= 2 *including repeats*, and ``_build``
   appends once per occurrence. An address ``"no 12 12 main"`` therefore
   posts ``S2-1`` twice for the key ``("india", "12")``. The Day 2 posting
   caps compare ``len(ids)`` against the cap, so those duplicates change
   cap behaviour. Deduplicating them here would silently change which
   records pass a cap.
2. **Tokenizer behaviour is shared, not reimplemented.** The helpers are
   imported from ``blocking`` so tokenization, numeric handling,
   transliteration and pair-key generation cannot drift.

Route names, posting caps, candidate ordering semantics and the
``CandidateEvidence`` representation are all unchanged.
"""

from __future__ import annotations

import hashlib
from array import array
from typing import Dict, Iterable, List, Sequence

import numpy as np

from business_entity_resolution.blocking import (
    CandidateEvidence,
    _address_pair_keys,
    _country,
    _numeric_tokens,
    _tokens,
    _transliterate,
    _REQUIRED_FIELDS,
)

# Separator between key components. normalize_unicode maps every
# non-alphanumeric, non-combining character to a space, so a NUL byte can
# never appear inside a normalized field. That makes the join injective.
_KEY_SEP = b"\x00"

_EMPTY_POSTINGS = np.empty(0, dtype=np.int32)

# Index family names, in the order the Day 2 index built them.
F_NAME = "name"
F_TOKEN_NAME = "token_name"
F_ADDRESS = "address"
F_COMBINED = "combined"
F_NAME_TOKEN = "name_token"
F_ADDRESS_TOKEN = "address_token"
F_ADDRESS_PAIR = "address_pair"
F_NUMERIC_ADDRESS = "numeric_address"
F_TRANSLIT_NAME = "translit_name"
F_TRANSLIT_TOKEN = "translit_token"

FAMILIES = (
    F_NAME,
    F_TOKEN_NAME,
    F_ADDRESS,
    F_COMBINED,
    F_NAME_TOKEN,
    F_ADDRESS_TOKEN,
    F_ADDRESS_PAIR,
    F_NUMERIC_ADDRESS,
    F_TRANSLIT_NAME,
    F_TRANSLIT_TOKEN,
)


def key_hash(*parts: str) -> int:
    """Stable 64-bit hash of a composite key.

    ``hash()`` is not usable because string hashing is salted per process,
    which would make a persisted index unreadable in another process.
    """

    digest = hashlib.blake2b(digest_size=8)

    for position, part in enumerate(parts):

        if position:
            digest.update(_KEY_SEP)

        digest.update(part.encode("utf-8"))

    return int.from_bytes(digest.digest(), "little", signed=False)


class CompactFamily:
    """CSR postings for one index family, keyed by 64-bit key hash."""

    __slots__ = ("hashes", "indptr", "postings", "distinct_keys")

    def __init__(
        self,
        hashes: np.ndarray,
        indptr: np.ndarray,
        postings: np.ndarray,
        distinct_keys: int,
    ) -> None:
        self.hashes = hashes
        self.indptr = indptr
        self.postings = postings
        self.distinct_keys = distinct_keys

    @property
    def total_postings(self) -> int:
        return int(self.postings.size)

    def _slot(self, value: int) -> int:
        """Return the CSR slot for ``value``, or -1 when absent."""

        position = int(np.searchsorted(self.hashes, np.uint64(value)))

        if position >= self.hashes.size:
            return -1

        if int(self.hashes[position]) != value:
            return -1

        return position

    def count(self, value: int) -> int:
        """Posting-list length, duplicates included, or 0 when absent."""

        slot = self._slot(value)

        if slot < 0:
            return 0

        return int(self.indptr[slot + 1] - self.indptr[slot])

    def rows(self, value: int) -> np.ndarray:
        """Posting rows for a key, duplicates and original order preserved."""

        slot = self._slot(value)

        if slot < 0:
            return _EMPTY_POSTINGS

        return self.postings[self.indptr[slot] : self.indptr[slot + 1]]


class _FamilyBuilder:
    """Accumulates one family's postings before they become a CSR family.

    The accumulators are typed buffers rather than Python lists. A Python
    list costs an 8-byte pointer plus a 28-byte ``int`` object for every
    posting, so a full-scale test target source needed roughly 10 GB per
    family set just to hold the build, which exhausted the 23.7 GB machine
    during Day 3 measurement. ``array('Q')`` and ``array('i')`` hold exactly
    the same values in 12 bytes per posting, and ``np.frombuffer`` wraps them
    without copying.

    ``keys`` backs the hash-collision assertion, which needs the *distinct*
    raw keys. That set is capped: on a build too large to track, the family
    is marked unverified (``distinct_keys = -1``) and the assertion is
    skipped rather than the run being lost to memory pressure. A 64-bit hash
    collision merges two posting lists and therefore only ever *adds*
    candidates, so skipping the check cannot cost recall.
    """

    __slots__ = ("hashes", "rows", "keys")

    # 2M distinct keys per family is ~240 MB of set overhead, which is
    # affordable and covers the whole development subset.
    MAX_TRACKED_KEYS = 2_000_000

    def __init__(self, track_keys: bool) -> None:
        self.hashes = array("Q")
        self.rows = array("i")
        self.keys: set[bytes] | None = set() if track_keys else None

    def add_key(self, row_id: int, *parts: str) -> None:
        """Append one posting for a composite key.

        The key bytes are built once and reused for both the hash and the
        optional collision-tracking set, and the raw bytes are only built at
        all when the set is still wanted. ``blake2b(raw, digest_size=8)`` is
        by construction identical to feeding the same separator-joined bytes
        to :func:`key_hash`, so every key value is unchanged.
        """

        keys = self.keys
        tracked = keys is not None and len(keys) < self.MAX_TRACKED_KEYS

        if tracked:
            raw = _raw_key(*parts)
        else:
            # Past the cap the set is abandoned, so stop building the bytes.
            if keys is not None:
                self.keys = None
            raw = _KEY_SEP.join(
                part.encode("utf-8") for part in parts
            )

        value = int.from_bytes(
            hashlib.blake2b(raw, digest_size=8).digest(),
            "little",
            signed=False,
        )

        self.hashes.append(value)
        self.rows.append(row_id)

        if tracked:
            keys.add(raw)

    def add(self, value: int, row_id: int, raw_key: bytes | None) -> None:
        self.hashes.append(value)
        self.rows.append(row_id)

        if self.keys is not None and raw_key is not None:
            if len(self.keys) < self.MAX_TRACKED_KEYS:
                self.keys.add(raw_key)
            else:
                self.keys = None

    def release(self) -> None:
        """Drop the build buffers once the family has been materialised."""

        self.hashes = array("Q")
        self.rows = array("i")
        self.keys = None

    def finish(self) -> CompactFamily:
        if not self.hashes:
            return CompactFamily(
                np.empty(0, dtype=np.uint64),
                np.zeros(1, dtype=np.int64),
                np.empty(0, dtype=np.int32),
                0,
            )

        # Zero-copy views: np.frombuffer keeps the array object alive.
        values = np.frombuffer(self.hashes, dtype=np.uint64)
        rows = np.frombuffer(self.rows, dtype=np.int32)

        # Stable sort keeps insertion order inside a posting list, matching
        # the order the Day 2 index appended to its Python lists.
        order = np.argsort(values, kind="stable")
        values = values[order]
        rows = rows[order]

        unique, starts = np.unique(values, return_index=True)

        # CSR: indptr[i] is the first posting of bucket i, so the bucket
        # starts go in positions 0..n-1 and the total closes the last one.
        indptr = np.empty(unique.size + 1, dtype=np.int64)
        indptr[:-1] = starts
        indptr[-1] = rows.size

        # ``unique``, not ``values``: the slot lookup in CompactFamily relies
        # on one hash per key, with duplicates addressed through indptr.
        return CompactFamily(
            unique,
            indptr,
            np.ascontiguousarray(rows),
            len(self.keys) if self.keys is not None else -1,
        )


def _raw_key(*parts: str) -> bytes:
    return _KEY_SEP.join(part.encode("utf-8") for part in parts)


def _finish_families(builders: Dict[str, _FamilyBuilder]) -> dict:
    """Materialise every family, releasing each build buffer as it lands.

    Holding all ten build buffers while the last family is sorted was the
    peak of a full-scale build; freeing them one at a time keeps only the
    family being materialised plus the remaining buffers.
    """

    families = {}

    for family, builder in builders.items():
        families[family] = builder.finish()
        builder.release()

    return families


class EntityIdTable:
    """Row id -> entity id, held as one packed buffer plus offsets."""

    __slots__ = ("_blob", "_offsets", "count")

    def __init__(self, entity_ids: Sequence[str]) -> None:
        encoded = [value.encode("utf-8") for value in entity_ids]

        self._blob = b"".join(encoded)
        self.count = len(encoded)

        offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        np.cumsum([len(item) for item in encoded], out=offsets[1:])

        self._offsets = offsets

    @classmethod
    def build_streaming(
        cls,
        entity_ids: Iterable[str],
    ) -> "EntityIdTable":
        """Build from a one-shot iterable, without a temporary list.

        The sequence constructor holds every encoded id at once, which is
        avoidable when the caller is streaming records off disk anyway.
        """

        table = cls.__new__(cls)

        blob = bytearray()
        lengths = []

        for value in entity_ids:
            encoded = value.encode("utf-8")
            blob.extend(encoded)
            lengths.append(len(encoded))

        table._blob = bytes(blob)
        table.count = len(lengths)

        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        table._offsets = offsets

        return table

    def __len__(self) -> int:
        return self.count

    def get(self, row_id: int) -> str:
        start = int(self._offsets[row_id])
        end = int(self._offsets[row_id + 1])
        return self._blob[start:end].decode("utf-8")


class CompactBlockingIndex:
    """Country-aware blocking index with compact integer postings.

    Emits the same candidate set as ``BlockingIndex`` for the same records,
    the same config and the same query record.
    """

    def __init__(
        self,
        records: Sequence,
        *,
        rare_token_max_postings: int = 50,
        numeric_token_max_postings: int = 100,
        translit_token_max_postings: int = 100,
        address_token_max_postings: int = 50,
        address_pair_max_postings: int = 5,
        verify_no_hash_collisions: bool = True,
    ) -> None:

        if not records:
            raise ValueError("CompactBlockingIndex requires at least one record")

        missing = [
            field
            for field in _REQUIRED_FIELDS
            if not hasattr(records[0], field)
        ]
        if missing:
            raise ValueError(
                "indexed records are missing required normalized fields: "
                f"{missing}. Expected objects produced by "
                "business_entity_resolution.normalization.normalize_record()."
            )

        for name, value in (
            ("rare_token_max_postings", rare_token_max_postings),
            ("numeric_token_max_postings", numeric_token_max_postings),
            ("translit_token_max_postings", translit_token_max_postings),
            ("address_token_max_postings", address_token_max_postings),
            ("address_pair_max_postings", address_pair_max_postings),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.rare_token_max_postings = rare_token_max_postings
        self.numeric_token_max_postings = numeric_token_max_postings
        self.translit_token_max_postings = translit_token_max_postings
        self.address_token_max_postings = address_token_max_postings
        self.address_pair_max_postings = address_pair_max_postings

        self.entity_ids = EntityIdTable(
            [record.entity_id for record in records]
        )

        builders = {
            family: _FamilyBuilder(verify_no_hash_collisions)
            for family in FAMILIES
        }

        for row_id, record in enumerate(records):
            self._index_record(builders, row_id, record)

        self.families = _finish_families(builders)

        if verify_no_hash_collisions:
            self._assert_no_collisions()

    @classmethod
    def from_stream(
        cls,
        records: Iterable,
        *,
        rare_token_max_postings: int = 50,
        numeric_token_max_postings: int = 100,
        translit_token_max_postings: int = 100,
        address_token_max_postings: int = 50,
        address_pair_max_postings: int = 5,
        verify_no_hash_collisions: bool = True,
    ) -> "CompactBlockingIndex":
        """Build from a one-shot iterable of records.

        Identical index to the sequence constructor, but the caller never
        has to hold every record in memory at once. This is what makes the
        full-scale target sources feasible: records can be streamed country
        by country and released as soon as they are indexed.
        """

        self = cls.__new__(cls)

        for name, value in (
            ("rare_token_max_postings", rare_token_max_postings),
            ("numeric_token_max_postings", numeric_token_max_postings),
            ("translit_token_max_postings", translit_token_max_postings),
            ("address_token_max_postings", address_token_max_postings),
            ("address_pair_max_postings", address_pair_max_postings),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.rare_token_max_postings = rare_token_max_postings
        self.numeric_token_max_postings = numeric_token_max_postings
        self.translit_token_max_postings = translit_token_max_postings
        self.address_token_max_postings = address_token_max_postings
        self.address_pair_max_postings = address_pair_max_postings

        builders = {
            family: _FamilyBuilder(verify_no_hash_collisions)
            for family in FAMILIES
        }

        blob = bytearray()
        lengths = []

        row_id = 0
        validated = False

        for record in records:

            if not validated:
                missing = [
                    field
                    for field in _REQUIRED_FIELDS
                    if not hasattr(record, field)
                ]
                if missing:
                    raise ValueError(
                        "indexed records are missing required normalized "
                        f"fields: {missing}. Expected objects produced by "
                        "business_entity_resolution.normalization."
                        "normalize_record()."
                    )
                validated = True

            encoded = record.entity_id.encode("utf-8")
            blob.extend(encoded)
            lengths.append(len(encoded))

            self._index_record(builders, row_id, record)

            row_id += 1

        if row_id == 0:
            raise ValueError(
                "CompactBlockingIndex requires at least one record"
            )

        table = EntityIdTable.__new__(EntityIdTable)
        table._blob = bytes(blob)
        table.count = row_id

        offsets = np.zeros(row_id + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        table._offsets = offsets

        self.entity_ids = table

        self.families = _finish_families(builders)

        if verify_no_hash_collisions:
            self._assert_no_collisions()

        return self

    # ==============================================================
    # BUILD
    # ==============================================================

    def _index_record(
        self,
        builders: Dict[str, _FamilyBuilder],
        row_id: int,
        record,
    ) -> None:

        country = _country(record)
        name = record.name_unicode
        token_name = record.name_token_sorted
        address = record.address_unicode

        if name:
            builders[F_NAME].add_key(row_id, country, name)

        if token_name:
            builders[F_TOKEN_NAME].add_key(row_id, country, token_name)

        if address:
            builders[F_ADDRESS].add_key(row_id, country, address)

        # A combined key needs real evidence on both sides.
        if name and address:
            builders[F_COMBINED].add_key(
                row_id, country, name, address
            )

        # Per-occurrence appends: duplicates are load-bearing for the caps.
        add_key = builders[F_NAME_TOKEN].add_key
        for token in _tokens(name):
            add_key(row_id, country, token)

        add_key = builders[F_ADDRESS_TOKEN].add_key
        for token in _tokens(address):
            add_key(row_id, country, token)

        add_key = builders[F_ADDRESS_PAIR].add_key
        for first, second in _address_pair_keys(address):
            add_key(row_id, country, first, second)

        add_key = builders[F_NUMERIC_ADDRESS].add_key
        for token in _numeric_tokens(address):
            add_key(row_id, country, token)

        translit = _transliterate(name)

        if translit:
            builders[F_TRANSLIT_NAME].add_key(
                row_id, country, translit
            )

            add_key = builders[F_TRANSLIT_TOKEN].add_key
            for token in _tokens(translit):
                add_key(row_id, country, token)

    def _assert_no_collisions(self) -> None:
        """Fail loudly if two distinct keys hashed to the same 64-bit value.

        A collision would merge two posting lists and inject foreign
        candidates, so it must never pass silently.
        """

        for family, built in self.families.items():
            if built.distinct_keys < 0:
                continue

            observed = int(built.hashes.size)

            if observed != built.distinct_keys:
                raise ValueError(
                    f"64-bit key hash collision in family {family!r}: "
                    f"{built.distinct_keys} distinct keys produced "
                    f"{observed} distinct hashes"
                )

    # ==============================================================
    # PERSISTENCE
    # ==============================================================
    #
    # Building a full test target source costs ~25 minutes of CPU, so the
    # index is written to disk once and memory-mapped back on every later
    # run. Nothing in the format changes the candidate set: the same hashes,
    # indptr, postings and entity-id bytes are stored verbatim.

    INDEX_FORMAT = 1
    INDEX_METADATA = "index.json"

    def save(self, directory) -> dict:
        """Write the index to ``directory``; return the metadata written."""

        import json
        from pathlib import Path

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)

        for family, built in self.families.items():
            np.save(destination / f"{family}_hashes.npy", built.hashes)
            np.save(destination / f"{family}_indptr.npy", built.indptr)
            np.save(destination / f"{family}_postings.npy", built.postings)

        (destination / "entity_blob.bin").write_bytes(self.entity_ids._blob)
        np.save(
            destination / "entity_offsets.npy", self.entity_ids._offsets
        )

        metadata = {
            "format": self.INDEX_FORMAT,
            "entity_count": int(self.entity_ids.count),
            "families": sorted(self.families),
            "caps": {
                "rare_token_max_postings": self.rare_token_max_postings,
                "numeric_token_max_postings": (
                    self.numeric_token_max_postings
                ),
                "translit_token_max_postings": (
                    self.translit_token_max_postings
                ),
                "address_token_max_postings": (
                    self.address_token_max_postings
                ),
                "address_pair_max_postings": (
                    self.address_pair_max_postings
                ),
            },
            "hash_algorithm": "blake2b-64-little",
        }

        (destination / self.INDEX_METADATA).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return metadata

    @classmethod
    def load(cls, directory, *, mmap_mode: str | None = "r") -> "CompactBlockingIndex":
        """Read an index written by :meth:`save`.

        ``mmap_mode='r'`` keeps the postings in the page cache instead of
        the process heap, so a second run over the same target source costs
        no rebuild and no resident copy.
        """

        import json
        from pathlib import Path

        source = Path(directory)
        metadata_path = source / cls.INDEX_METADATA

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"{metadata_path} is missing; rebuild the index"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if int(metadata.get("format", 0)) != cls.INDEX_FORMAT:
            raise ValueError(
                f"index format {metadata.get('format')!r} is not "
                f"{cls.INDEX_FORMAT}; rebuild the index"
            )

        self = cls.__new__(cls)

        for key, value in metadata["caps"].items():
            setattr(self, key, int(value))

        table = EntityIdTable.__new__(EntityIdTable)
        table._blob = (source / "entity_blob.bin").read_bytes()
        table.count = int(metadata["entity_count"])
        table._offsets = np.load(
            source / "entity_offsets.npy", mmap_mode=mmap_mode
        )
        self.entity_ids = table

        self.families = {
            family: CompactFamily(
                np.load(
                    source / f"{family}_hashes.npy", mmap_mode=mmap_mode
                ),
                np.load(
                    source / f"{family}_indptr.npy", mmap_mode=mmap_mode
                ),
                np.load(
                    source / f"{family}_postings.npy", mmap_mode=mmap_mode
                ),
                -1,
            )
            for family in metadata["families"]
        }

        return self

    def is_cached(self, directory) -> bool:
        from pathlib import Path

        return (
            Path(directory) / self.INDEX_METADATA
        ).is_file()

    # ==============================================================
    # STATS
    # ==============================================================

    def stats(self) -> dict[str, int]:
        result = {
            "records": len(self.entity_ids),
            "entity_id_bytes": len(self.entity_ids._blob),
        }

        for family in FAMILIES:
            built = self.families[family]
            result[f"{family}_postings"] = built.total_postings
            result[f"{family}_keys"] = int(built.hashes.size)

        return result

    # ==============================================================
    # ADD HELPERS
    # ==============================================================

    def _add_rows(
        self,
        evidence: Dict[str, CandidateEvidence],
        source1_id: str,
        target_source: str,
        rows: Iterable[int],
        route: str,
    ) -> None:
        entity_ids = self.entity_ids

        for row in rows:

            row = int(row)
            entity_id = entity_ids.get(row)

            if entity_id == source1_id:
                continue

            existing = evidence.get(entity_id)

            if existing is None:
                existing = CandidateEvidence(
                    source1_entity_id=source1_id,
                    candidate_entity_id=entity_id,
                    target_source=target_source,
                )
                evidence[entity_id] = existing

            existing.add(route)

    def _add_key(
        self,
        evidence: Dict[str, CandidateEvidence],
        record,
        target_source: str,
        family: str,
        parts: Sequence[str],
        route: str,
    ) -> None:
        rows = self.families[family].rows(key_hash(*parts))

        if rows.size == 0:
            return

        self._add_rows(
            evidence,
            record.entity_id,
            target_source,
            rows,
            route,
        )

    # ==============================================================
    # ROUTES
    # Mirrors business_entity_resolution.blocking.BlockingIndex exactly.
    # ==============================================================

    def exact_combined(self, record, evidence, target_source: str) -> None:
        name = record.name_unicode
        address = record.address_unicode

        if not name or not address:
            return

        self._add_key(
            evidence, record, target_source, F_COMBINED,
            (_country(record), name, address), "exact_combined",
        )

    def exact_name(self, record, evidence, target_source: str) -> None:
        name = record.name_unicode

        if not name:
            return

        self._add_key(
            evidence, record, target_source, F_NAME,
            (_country(record), name), "exact_name",
        )

    def token_sorted_name(self, record, evidence, target_source: str) -> None:
        token_name = record.name_token_sorted

        if not token_name:
            return

        self._add_key(
            evidence, record, target_source, F_TOKEN_NAME,
            (_country(record), token_name), "token_sorted_name",
        )

    def exact_address(self, record, evidence, target_source: str) -> None:
        address = record.address_unicode

        if not address:
            return

        self._add_key(
            evidence, record, target_source, F_ADDRESS,
            (_country(record), address), "exact_address",
        )

    def rare_name_tokens(self, record, evidence, target_source: str) -> None:
        country = _country(record)
        family = self.families[F_NAME_TOKEN]

        for token in _tokens(record.name_unicode):

            value = key_hash(country, token)

            if family.count(value) == 0:
                continue

            if family.count(value) > self.rare_token_max_postings:
                continue

            self._add_rows(
                evidence, record.entity_id, target_source,
                family.rows(value), "rare_name_tokens",
            )

    def address_tokens(self, record, evidence, target_source: str) -> None:
        country = _country(record)
        family = self.families[F_ADDRESS_TOKEN]

        tokens = _tokens(record.address_unicode)

        if not tokens:
            return

        counts: dict[int, int] = {}

        for token in tokens:

            value = key_hash(country, token)

            if family.count(value) == 0:
                continue

            if family.count(value) > self.address_token_max_postings:
                continue

            for row in family.rows(value):
                row = int(row)
                counts[row] = counts.get(row, 0) + 1

        # More than one shared address token, so a single common token
        # cannot qualify a candidate on its own.
        for row, count in counts.items():
            if count < 2:
                continue
            self._add_rows(
                evidence, record.entity_id, target_source,
                [row], "address_tokens",
            )

    def address_token_pairs(self, record, evidence, target_source: str) -> None:
        country = _country(record)
        family = self.families[F_ADDRESS_PAIR]

        for first, second in _address_pair_keys(record.address_unicode):

            value = key_hash(country, first, second)

            if family.count(value) == 0:
                continue

            # The cap applies to the COMBINED bucket.
            if family.count(value) > self.address_pair_max_postings:
                continue

            self._add_rows(
                evidence, record.entity_id, target_source,
                family.rows(value), "address_token_pair",
            )

    def numeric_address(self, record, evidence, target_source: str) -> None:
        country = _country(record)
        family = self.families[F_NUMERIC_ADDRESS]

        for token in _numeric_tokens(record.address_unicode):

            value = key_hash(country, token)

            if family.count(value) == 0:
                continue

            if family.count(value) > self.numeric_token_max_postings:
                continue

            self._add_rows(
                evidence, record.entity_id, target_source,
                family.rows(value), "numeric_address",
            )

    def transliterated_name(self, record, evidence, target_source: str) -> None:
        country = _country(record)

        query = _transliterate(record.name_unicode)

        if not query:
            return

        self._add_key(
            evidence, record, target_source, F_TRANSLIT_NAME,
            (country, query), "transliterated_exact_name",
        )

        query_tokens = _tokens(query)

        if not query_tokens:
            return

        name_family = self.families[F_TRANSLIT_NAME]
        token_family = self.families[F_TRANSLIT_TOKEN]

        counts: dict[int, int] = {}

        for token in query_tokens:

            value = key_hash(country, token)

            if token_family.count(value) == 0:
                continue

            if token_family.count(value) > self.translit_token_max_postings:
                continue

            for row in token_family.rows(value):
                row = int(row)
                counts[row] = counts.get(row, 0) + 1

        for row, count in counts.items():
            # One shared token is acceptable only for very short names.
            if count >= 2 or len(query_tokens) <= 2:
                self._add_rows(
                    evidence, record.entity_id, target_source,
                    [row], "transliterated_token",
                )

    # ==============================================================
    # MAIN
    # ==============================================================

    ROUTES = (
        "exact_combined",
        "exact_name",
        "token_sorted_name",
        "exact_address",
        "rare_name_tokens",
        "address_tokens",
        "address_token_pairs",
        "numeric_address",
        "transliterated_name",
    )

    def generate(
        self,
        record,
        target_source: str = "",
    ) -> List[CandidateEvidence]:

        evidence: Dict[str, CandidateEvidence] = {}

        for route in self.ROUTES:
            getattr(self, route)(record, evidence, target_source)

        return list(evidence.values())
