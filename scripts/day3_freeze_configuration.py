"""Day 3 / Task 1: record the frozen candidate-generation configuration.

The configuration is introspected from the shipped code rather than
transcribed by hand, so this report cannot drift from what the pipeline
actually does. Nothing here changes behaviour: it only reads.

Usage:
    python scripts/day3_freeze_configuration.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from business_entity_resolution import blocking  # noqa: E402
from business_entity_resolution import candidate_generation  # noqa: E402
from business_entity_resolution import chunked_generation  # noqa: E402
from business_entity_resolution import compact_index  # noqa: E402
from business_entity_resolution import config as app_config  # noqa: E402
from business_entity_resolution import retrieval  # noqa: E402

OUTPUT = REPO / "docs" / "day3_final_configuration.json"

# Blocking route method -> the index family it reads. Kept here so the
# report names the physical index each route depends on, which is what a
# reduction experiment has to keep intact.
ROUTE_FAMILY = {
    "exact_combined": compact_index.F_COMBINED,
    "exact_name": compact_index.F_NAME,
    "token_sorted_name": compact_index.F_TOKEN_NAME,
    "exact_address": compact_index.F_ADDRESS,
    "rare_name_tokens": compact_index.F_NAME_TOKEN,
    "address_tokens": compact_index.F_ADDRESS_TOKEN,
    "address_token_pairs": compact_index.F_ADDRESS_PAIR,
    "numeric_address": compact_index.F_NUMERIC_ADDRESS,
    "transliterated_name": (
        compact_index.F_TRANSLIT_NAME,
        compact_index.F_TRANSLIT_TOKEN,
    ),
}

# Emission route names produced by the TF-IDF retrievers. The retriever
# builds the name as f"{field}_tfidf".
TFIDF_ROUTES = tuple(
    f"{field}_tfidf" for field in ("name", "address")
)


def _file_digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _retrieval_routes() -> list[dict[str, Any]]:
    routes = []

    for route in blocking.BlockingIndex.ROUTES:
        families = ROUTE_FAMILY[route]
        if isinstance(families, str):
            families = (families,)
        routes.append(
            {
                "route": route,
                "kind": "blocking",
                "implemented_by": "business_entity_resolution.compact_index"
                ".CompactBlockingIndex",
                "blocking_families": list(families),
                "emitted_route_names": sorted(
                    {
                        name
                        for family in families
                        for name in _route_names_using(family)
                    }
                ),
            }
        )

    for route in TFIDF_ROUTES:
        field = route[: -len("_tfidf")]
        routes.append(
            {
                "route": route,
                "kind": "tfidf",
                "implemented_by": "business_entity_resolution.retrieval"
                ".CharacterTfidfRetriever",
                "field": field,
                "emitted_route_names": [route],
            }
        )

    return routes


def _route_names_using(family: str) -> set[str]:
    """Route names emitted by the methods that read one index family."""

    names: set[str] = set()

    for route, families in ROUTE_FAMILY.items():
        if isinstance(families, str):
            families = (families,)
        if family in families:
            method = getattr(
                compact_index.CompactBlockingIndex, route
            )
            # transliterated_name emits two names; read them from the body.
            source = inspect.getsource(method)
            for literal in (
                '"transliterated_exact_name"',
                '"transliterated_token"',
                '"rare_name_tokens"',
                '"address_tokens"',
                '"address_token_pair"',
                '"numeric_address"',
                '"exact_name"',
                '"exact_address"',
                '"exact_combined"',
                '"token_sorted_name"',
            ):
                if literal.strip('"') in source:
                    names.add(literal.strip('"'))
            break

    return names


def build_report() -> dict[str, Any]:
    frozen = candidate_generation.CandidateGenerationConfig()
    settings = app_config.load_config(REPO / "configs" / "base.json")

    parameters = dataclasses.asdict(frozen)
    parameters["ngram_range"] = list(frozen.ngram_range)

    package = REPO / "src" / "business_entity_resolution"

    source_files = [
        _file_digest(package / name)
        for name in (
            "blocking.py",
            "candidate_generation.py",
            "chunked_generation.py",
            "compact_index.py",
            "normalization.py",
            "retrieval.py",
            "run_chunked_generation.py",
        )
    ]

    return {
        "report": "day3_final_configuration",
        "person": "Person 2 - candidate generation",
        "day": 3,
        "status": "FROZEN",
        "generated_by": "scripts/day3_freeze_configuration.py",
        "reproduce": (
            "python scripts/day3_freeze_configuration.py"
        ),
        "parameter_source_of_truth": (
            "business_entity_resolution.candidate_generation"
            ".CandidateGenerationConfig dataclass defaults"
        ),
        "parameters": parameters,
        "retrieval_routes": _retrieval_routes(),
        "retrieval_route_count": len(
            blocking.BlockingIndex.ROUTES
        )
        + len(TFIDF_ROUTES),
        "posting_caps": {
            "rare_name_token_max_postings": frozen.rare_token_max_postings,
            "transliterated_token_max_postings": (
                frozen.translit_token_max_postings
            ),
            "numeric_address_token_max_postings": (
                frozen.numeric_token_max_postings
            ),
            "address_token_max_postings": (
                frozen.address_token_max_postings
            ),
            "address_token_pair_max_postings": (
                frozen.address_pair_max_postings
            ),
            "semantics": {
                "rare_name_tokens": (
                    "token posting length > cap -> bucket rejected "
                    "entirely (count includes per-occurrence duplicates)"
                ),
                "transliterated_token": (
                    "token posting length > cap -> bucket rejected"
                ),
                "numeric_address": (
                    "numeric token posting length > cap -> bucket "
                    "rejected; leading zeros stripped before keying"
                ),
                "address_tokens": (
                    "token posting length > cap -> bucket rejected; a "
                    "candidate then needs >= 2 shared address tokens"
                ),
                "address_token_pair": (
                    "cap applies to the COMBINED two-token bucket, not to "
                    "each token separately"
                ),
            },
        },
        "country_handling": {
            "policy": "open set; no hard-coded country list",
            "blocking_partition_key": (
                "blocking._country(record): "
                "str(country).strip().lower()"
            ),
            "tfidf_partition_key": (
                "retrieval.CharacterTfidfRetriever groups by "
                "NormalizedRecord.country == str(country).strip() "
                "(case preserved)"
            ),
            "chunked_partition_key": (
                "chunked_generation.partition_targets_by_country writes "
                "one file per str(country).strip() or '__unknown__'; "
                "chunked_generation._country_of(source1) resolves the "
                "S1 side to the same key"
            ),
            "partition_map_file": "partition_map.json",
            "partition_filename_transform": (
                "chunked_generation._safe: keep [A-Za-z0-9-_], replace "
                "everything else with '_', truncate to 64 chars"
            ),
            "cross_country_candidates_possible": False,
            "blank_country_sentinel": "__unknown__",
            "test_countries_observed": ["France", "India", "US"],
        },
        "missing_evidence_policy": {
            "address_missing": (
                "a blank normalized address is MISSING evidence, never an "
                "exact match: exact_address, exact_combined, "
                "address_tokens, address_token_pairs, numeric_address and "
                "address_tfidf all require a non-empty address"
            ),
            "name_missing": (
                "a blank normalized name disables exact_name, "
                "token_sorted_name, rare_name_tokens and the "
                "transliteration routes"
            ),
        },
        "chunking": {
            "default_chunk_size": chunked_generation.DEFAULT_CHUNK_SIZE,
            "unit": "Source 1 records per chunk file",
            "chunk_boundary_rule": (
                "row position in the source file, not content, so a "
                "resumed run recomputes identical chunks"
            ),
            "s1_reader_chunksize": chunked_generation.DEFAULT_CHUNK_SIZE,
            "target_partition_reader_chunksize": 50_000,
            "chunk_header": chunked_generation.CHUNK_HEADER,
            "final_header": chunked_generation.FINAL_HEADER,
            "atomic_write": (
                "chunk written to <name>.tsv.tmp then os.replace; "
                "manifest written to manifest.json.tmp then os.replace"
            ),
        },
        "compact_index": {
            "class": "business_entity_resolution.compact_index"
            ".CompactBlockingIndex",
            "used_by": [
                "CandidateGenerator(blocking_backend='compact')",
                "chunked_generation.ChunkedTargetIndex"
                "(CompactBlockingIndex.from_stream)",
            ],
            "layout": "CSR postings of int32 row ids, sorted uint64 keys",
            "families": list(compact_index.FAMILIES),
            "key_hash": {
                "function": "business_entity_resolution.compact_index"
                ".key_hash",
                "algorithm": "blake2b",
                "digest_size_bytes": 8,
                "byte_order": "little",
                "signed": False,
                "component_separator": "NUL byte (0x00)",
                "separator_is_injective_because": (
                    "normalization.normalize_unicode maps every "
                    "non-alphanumeric, non-combining character to a "
                    "space, so NUL cannot occur inside a normalized field"
                ),
                "why_not_builtin_hash": (
                    "string hashing is salted per process, so a persisted "
                    "index would be unreadable in another process"
                ),
            },
            "verify_no_hash_collisions": True,
            "posting_multiset_semantics": (
                "postings are multisets, not sets: one append per token "
                "occurrence, so repeated tokens change cap behaviour and "
                "are preserved deliberately"
            ),
            "shared_tokenizers": [
                "blocking._tokens",
                "blocking._numeric_tokens",
                "blocking._transliterate",
                "blocking._address_pair_keys",
                "blocking._country",
            ],
        },
        "deduplication_rules": {
            "scope": "one S1 record's candidate list",
            "within_source": (
                "evidence is keyed by candidate_entity_id in a dict, so "
                "a candidate found by several routes appears once with "
                "its routes merged"
            ),
            "across_sources": (
                "S2 and S3 are generated independently; finalize() "
                "unions them and keeps first-seen order"
            ),
            "self_match_excluded": True,
            "self_match_mechanism": (
                "blocking._add_ids / compact_index._add_rows skip a "
                "candidate whose entity_id equals the S1 id; "
                "retrieval.retrieve skips the same"
            ),
            "order": (
                "sort key (-route_count, -retrieval_score, "
                "retrieval_rank, candidate_entity_id); the final "
                "entity_id tie-break makes the order total and "
                "independent of stream order"
            ),
            "finalize_dedup": (
                "chunked_generation.finalize keeps first-seen order and "
                "rejects an entity_id appearing in two chunks of the "
                "same target source"
            ),
        },
        "transliteration": {
            "library": "Unidecode==1.4.0",
            "role": "additional route layered on top of the Unicode view",
            "never_replaces": "the Unicode-preserving name index",
            "degradation": (
                "if Unidecode is unavailable the routes become duplicates "
                "of the Unicode routes instead of failing"
            ),
        },
        "entry_points": {
            "full_test_chunked": (
                "python -m business_entity_resolution.run_chunked_generation "
                "--s1 <test_source1.tsv> --target-s2 <test_source2.tsv> "
                "--target-s3 <test_source3.tsv> --work-dir <dir> "
                "--output <candidate_pairs.tsv>"
            ),
            "development_evaluation": (
                "python -m business_entity_resolution.dev_candidate_generation "
                "--limit <n> --blocking-backend compact"
            ),
            "final_output_validation": (
                "python -m business_entity_resolution.validate_candidates "
                "--candidates <file> --s1 <...> --s2 <...> --s3 <...>"
            ),
        },
        "project_settings": settings.as_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "imported_from": str(REPO / "src"),
        },
        "frozen_source_files": source_files,
        "fingerprint": hashlib.sha256(
            json.dumps(
                {
                    "parameters": parameters,
                    "routes": [route["route"] for route in
                               _retrieval_routes()],
                    "sources": source_files,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT.relative_to(REPO)}")
    print(f"fingerprint {report['fingerprint']}")
    print(
        f"routes {report['retrieval_route_count']} "
        f"({len(blocking.BlockingIndex.ROUTES)} blocking + "
        f"{len(TFIDF_ROUTES)} tfidf)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
