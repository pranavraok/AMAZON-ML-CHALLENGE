"""Exact equivalence between CompactBlockingIndex and the Day-2 BlockingIndex.

The Day-2 index is the reference. These tests assert that the compact
rewrite returns the *identical* candidate pair set, not merely a similar
recall number, because a recall match can hide compensating errors.
"""

from __future__ import annotations

import pandas as pd
import pytest

from business_entity_resolution.blocking import BlockingIndex
from business_entity_resolution.candidate_generation import (
    CandidateGenerator,
)
from business_entity_resolution.compact_index import CompactBlockingIndex
from business_entity_resolution.normalization import normalize_record


def _record(entity_id, name, address, country="India"):
    return normalize_record(
        {
            "entity_id": entity_id,
            "business_name": name,
            "business_address": address,
            "country": country,
        }
    )


def _index_pair(records, **caps):
    caps = {
        "rare_token_max_postings": 50,
        "numeric_token_max_postings": 100,
        "translit_token_max_postings": 100,
        "address_token_max_postings": 50,
        "address_pair_max_postings": 5,
    } | caps
    return (
        BlockingIndex(records, **caps),
        CompactBlockingIndex(records, **caps),
    )


def _pairs(index, record, target_source="S2"):
    return {e.candidate_entity_id for e in index.generate(record, target_source)}


def _routes(index, record, target_source="S2"):
    out = {}
    for evidence in index.generate(record, target_source):
        out[evidence.candidate_entity_id] = tuple(evidence.routes)
    return out


# ==============================================================
# Duplicate postings are load-bearing
# ==============================================================


def test_duplicate_tokens_produce_duplicate_postings():
    """A repeated token must post the same row more than once.

    The Day 2 caps compare len(postings) against the cap, so collapsing
    duplicates would change which records pass a cap.
    """

    target = _record("S2-1", "new york new", "no 12 12 main")
    reference = BlockingIndex([target])

    assert reference.address_token_index[("india", "12")] == ["S2-1", "S2-1"]
    assert reference.name_token_index[("india", "new")] == ["S2-1", "S2-1"]
    assert reference.numeric_address_index[("india", "12")] == ["S2-1", "S2-1"]


def test_compact_index_preserves_duplicate_posting_counts():
    target = _record("S2-1", "new york new", "no 12 12 main")
    compact = CompactBlockingIndex([target])

    from business_entity_resolution.compact_index import key_hash

    assert compact.families["address_token"].count(key_hash("india", "12")) == 2
    assert compact.families["name_token"].count(key_hash("india", "new")) == 2
    assert (
        compact.families["numeric_address"].count(key_hash("india", "12")) == 2
    )


def test_duplicate_aware_cap_behaviour_matches():
    """Two postings sit under a cap of 1 in the reference but not a cap of 2."""

    targets = [
        _record("S2-1", "acme", "no 12 12 main"),
        _record("S2-2", "acme", "no 12 12 main"),
    ]
    query = _record("S1-9", "acme", "no 12 12 main")

    for cap in (1, 2, 3):
        reference, compact = _index_pair(
            targets, numeric_token_max_postings=cap
        )
        assert _pairs(reference, query) == _pairs(compact, query), (
            f"numeric cap {cap} diverged"
        )


# ==============================================================
# Route-level equivalence on hand-built cases
# ==============================================================


ROUTE_CASES = [
    ("exact_name", "al infotech private limited", "somewhere", "S2-1"),
    (
        "token_sorted_name",
        "private limited al infotech",
        "somewhere",
        "S2-1",
    ),
    ("exact_address", "al infotech", "12 main street", "S2-1"),
    (
        "exact_combined",
        "al infotech private limited",
        "12 main street",
        "S2-1",
    ),
    ("rare_name_tokens", "infotech", "nowhere at all", "S2-1"),
    ("address_tokens", "zzz", "12 main street", "S2-1"),
    ("address_token_pair", "zzz", "12 main street", "S2-1"),
    ("numeric_address", "zzz", "no 14 a", "S2-1"),
    ("transliterated_exact_name", "ಅಲ್ ಇನ್ಫೋಟೆಕ್", "nowhere", "S2-1"),
]


@pytest.mark.parametrize(
    "route,target_name,target_address,candidate_id", ROUTE_CASES
)
def test_each_route_fires_identically(route, target_name, target_address, candidate_id):
    targets = [_record(candidate_id, target_name, target_address)]
    reference, compact = _index_pair(targets)

    query = _record("S1-9", target_name, target_address)

    assert route in _routes(reference, query)["S2-1"]
    assert _routes(reference, query) == _routes(compact, query)


def test_country_isolation_is_preserved():
    targets = [
        _record("S2-1", "al infotech", "12 main", "India"),
        _record("S2-2", "al infotech", "12 main", "France"),
    ]
    reference, compact = _index_pair(targets)

    query = _record("S1-9", "al infotech", "12 main", "France")

    assert _pairs(reference, query) == {"S2-2"}
    assert _pairs(reference, query) == _pairs(compact, query)


def test_blank_address_never_matches():
    """A blank address must not satisfy any address-bearing route.

    The candidate can still be found on name evidence alone; what must never
    happen is an address route firing on missing evidence.
    """

    targets = [
        _record("S2-1", "al infotech", ""),
        _record("S2-2", "al infotech", "12 main"),
    ]
    reference, compact = _index_pair(targets)

    address_routes = {
        "exact_address",
        "exact_combined",
        "address_tokens",
        "address_token_pair",
        "numeric_address",
    }

    blank_query = _record("S1-9", "al infotech", "")
    assert _routes(reference, blank_query) == _routes(compact, blank_query)
    for routes in _routes(reference, blank_query).values():
        assert not (address_routes & set(routes))

    populated_query = _record("S1-8", "al infotech", "12 main")
    assert _routes(reference, populated_query) == _routes(compact, populated_query)


def test_target_source_is_propagated():
    targets = [_record("S2-1", "al infotech", "12 main")]
    _, compact = _index_pair(targets)

    query = _record("S1-9", "al infotech", "12 main")

    for evidence in compact.generate(query, "S2"):
        assert evidence.target_source == "S2"
        assert evidence.source1_entity_id == "S1-9"


def test_caps_are_respected_identically():
    targets = [
        _record(f"S2-{i}", f"company{i}", "12 main street") for i in range(1, 7)
    ]
    query = _record("S1-9", "company", "12 main street")

    for cap in range(1, 7):
        reference, compact = _index_pair(targets, address_pair_max_postings=cap)
        assert _pairs(reference, query) == _pairs(compact, query), (
            f"address pair cap {cap} diverged"
        )


def test_no_self_match():
    targets = [_record("S1-9", "al infotech", "12 main")]
    reference, compact = _index_pair(targets)

    query = _record("S1-9", "al infotech", "12 main")

    assert "S1-9" not in _pairs(reference, query)
    assert _pairs(reference, query) == _pairs(compact, query)


# ==============================================================
# Real development data
# ==============================================================


DEV_DIR = "data/dev/train"


def _load(name, limit=None):
    df = pd.read_csv(
        f"{DEV_DIR}/{name}", sep="\t", dtype=str, keep_default_na=False
    )
    if limit:
        df = df.head(limit)
    return [normalize_record(row) for row in df.to_dict(orient="records")]


@pytest.mark.skipif(
    not pd.io.common.file_exists(f"{DEV_DIR}/train_source2.tsv"),
    reason="development subset not present",
)
def test_matches_reference_index_on_real_development_records():
    """Route-level and pair-level equality on real normalized dev records."""

    s2 = _load("train_source2.tsv", limit=4000)
    s1 = _load("train_source1.tsv", limit=200)

    reference = BlockingIndex(s2)
    compact = CompactBlockingIndex(s2)

    compared = 0

    for record in s1:

        expected = {
            e.candidate_entity_id: tuple(e.routes)
            for e in reference.generate(record, "S2")
        }
        actual = {
            e.candidate_entity_id: tuple(e.routes)
            for e in compact.generate(record, "S2")
        }

        assert actual == expected, (
            f"divergence for {record.entity_id}: "
            f"only-compact={sorted(set(actual) - set(expected))[:5]} "
            f"only-reference={sorted(set(expected) - set(actual))[:5]}"
        )
        compared += len(expected)

    assert compared > 0


def test_compact_index_is_smaller_than_reference():
    """The whole point: compact postings must cost less memory."""

    s2 = _load("train_source2.tsv", limit=3000)

    reference = BlockingIndex(s2)
    compact = CompactBlockingIndex(s2)

    def deep_size(obj, seen):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            for key, value in obj.items():
                size += deep_size(key, seen) + deep_size(value, seen)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                size += deep_size(item, seen)
        return size

    import sys

    reference_indexes = [
        reference.name_index,
        reference.token_name_index,
        reference.address_index,
        reference.combined_index,
        reference.name_token_index,
        reference.address_token_index,
        reference.address_pair_index,
        reference.numeric_address_index,
        reference.translit_name_index,
        reference.translit_token_index,
    ]

    seen: set[int] = set()
    reference_bytes = sum(deep_size(i, seen) for i in reference_indexes)

    compact_bytes = sum(
        family.postings.nbytes
        + family.hashes.nbytes
        + family.indptr.nbytes
        for family in compact.families.values()
    )
    compact_bytes += len(compact.entity_ids._blob)
    compact_bytes += compact.entity_ids._offsets.nbytes

    assert compact_bytes < reference_bytes, (
        f"compact {compact_bytes / 1e6:.1f} MB is not smaller than "
        f"reference {reference_bytes / 1e6:.1f} MB"
    )


# =============================================================
# STEP 3 WIRING
# =============================================================


def test_generator_backend_produces_identical_candidates():
    """The whole generator, not just the index, must match."""

    records = [
        _record("S2-1", "Acme Bakery", "12 Main Street", "United States"),
        _record("S2-2", "Acme Bakker", "12 Main St", "United States"),
        _record("S2-3", "Borek Shop", "9 Oak Road", "United States"),
        _record("S2-4", "Zilch", "", "United States"),
    ]

    query = normalize_record(
        {
            "entity_id": "S1-1",
            "business_name": "Acme Baker",
            "business_address": "12 Main Street",
            "country": "United States",
        }
    )

    outputs = {}

    for backend in ("legacy", "compact"):

        generator = CandidateGenerator(
            records, blocking_backend=backend
        )

        assert generator.blocking_backend == backend

        outputs[backend] = [
            (
                evidence.candidate_entity_id,
                tuple(evidence.routes),
                evidence.retrieval_score,
                evidence.retrieval_rank,
            )
            for evidence in generator.generate_for_one(query)
        ]

    assert outputs["legacy"] == outputs["compact"]


def test_generator_rejects_unknown_backend():
    records = [_record("S2-1", "Acme", "1 Main St", "United States")]

    with pytest.raises(ValueError, match="legacy"):
        CandidateGenerator(records, blocking_backend="nonsense")


def test_generator_defaults_to_legacy():
    """Day 2 callers that pass nothing must keep the old index."""

    records = [_record("S2-1", "Acme", "1 Main St", "United States")]

    generator = CandidateGenerator(records)

    assert generator.blocking_backend == "legacy"
    assert isinstance(generator.blocking_index, BlockingIndex)


def test_retriever_does_not_retain_records():
    """Retrievers must hold entity ids, not record objects."""

    from business_entity_resolution.retrieval import (
        CharacterTfidfRetriever,
    )

    records = [
        _record("S2-1", "Acme Bakery", "12 Main Street", "United States"),
        _record("S2-2", "Borek Shop", "9 Oak Road", "United States"),
    ]

    retriever = CharacterTfidfRetriever(records, field="name")

    assert retriever.records == []

    for value in vars(retriever).values():
        if isinstance(value, list):
            for item in value:
                assert isinstance(item, str), (
                    f"retriever retained a {type(item).__name__}, "
                    "not an entity id"
                )

    results = retriever.retrieve(
        normalize_record(
            {
                "entity_id": "S1-1",
                "business_name": "Acme Bakery",
                "business_address": "12 Main Street",
                "country": "United States",
            }
        )
    )

    # Retrieval must still work once the records are gone. The second hit
    # is legitimate: char-trigram overlap with "Borek Shop" clears the
    # 0.30 name threshold.
    ids = [r.candidate_entity_id for r in results]

    assert ids[0] == "S2-1"
    assert "S2-1" in ids
    assert all(isinstance(entity_id, str) for entity_id in ids)
    assert all(
        r.candidate_entity_id != "S1-1" for r in results
    )


def test_retriever_stores_only_transposed_matrix():
    """The forward matrix doubled peak memory for no benefit."""

    from business_entity_resolution.retrieval import (
        CharacterTfidfRetriever,
    )

    records = [
        _record("S2-1", "Acme Bakery", "12 Main Street", "United States"),
        _record("S2-2", "Borek Shop", "9 Oak Road", "United States"),
    ]

    retriever = CharacterTfidfRetriever(records, field="name")

    assert not hasattr(retriever, "_country_records")
    assert not hasattr(retriever, "_country_matrices")
    assert retriever._country_transposed_matrices
