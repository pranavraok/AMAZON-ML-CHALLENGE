"""Candidate generation pipeline for Source 1 -> Source 2/3.

Combines cheap country-aware blocking routes with character n-gram TF-IDF
retrieval.

The final candidate set produced here is the set that is passed to the
matching model, so ``candidate_pairs.tsv`` must contain every final match.

Configuration defaults reflect the Day-2 development measurements on the
50k S1 development subset. See ``docs/interfaces.md`` for the column
contract this output must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from business_entity_resolution.blocking import (
    BlockingIndex,
    CandidateEvidence,
)
from business_entity_resolution.compact_index import CompactBlockingIndex
from business_entity_resolution.normalization import NormalizedRecord
from business_entity_resolution.retrieval import (
    CharacterTfidfRetriever,
)

@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Configuration for candidate generation.

    The address settings are deliberately deeper than the name settings.
    In the development data the dominant cause of missed true pairs is a
    cross-script name: the S1 name is Latin script while the true S2/S3
    match is written in a local script, so name similarity is exactly 0.
    The address is the only surviving evidence for those pairs.
    """

    # Name retrieval. name_min_score stays high on purpose: the true targets
    # for misspelled Latin names sit at rank 125-2116, so extra top-K depth
    # buys almost nothing while adding hundreds of low-value candidates.
    name_top_k: int = 100
    name_min_score: float = 0.30

    # Address retrieval. True address similarity for missed pairs measured
    # 0.128-0.418 with ranks 22-1686, so both depth and a real score floor
    # are needed. 0.15 keeps almost all of that band and rejects the
    # zero-similarity filler that min_score=0.0 used to admit.
    address_top_k: int = 100
    address_min_score: float = 0.15

    # Blocking posting caps.
    rare_token_max_postings: int = 50
    numeric_token_max_postings: int = 100
    translit_token_max_postings: int = 100
    address_token_max_postings: int = 50

    # Cap on an address token-PAIR bucket. Pairs of co-occurring address
    # tokens recover cross-script pairs that no other route reaches, at a
    # measured cost of about 7 extra candidates per S1.
    address_pair_max_postings: int = 5

    # Character n-gram range shared by both TF-IDF retrievers.
    ngram_range: tuple[int, int] = (3, 5)


class CandidateGenerator:
    """Generate candidates for S1 against one target source (S2 or S3)."""

    def __init__(
        self,
        target_records: Sequence[NormalizedRecord],
        *,
        config: CandidateGenerationConfig | None = None,
        blocking_backend: str = "legacy",
    ) -> None:

        if not target_records:
            raise ValueError(
                "CandidateGenerator requires target records"
            )

        self.config = config or CandidateGenerationConfig()

        if blocking_backend not in {"legacy", "compact"}:
            raise ValueError(
                "blocking_backend must be 'legacy' or 'compact'; "
                f"got {blocking_backend!r}"
            )

        self.blocking_backend = blocking_backend

        self.target_source = target_records[0].entity_id.split(
            "-", 1
        )[0]

        if self.target_source not in {"S2", "S3"}:
            raise ValueError(
                "Target records must belong to S2 or S3"
            )

        prefixes = {
            record.entity_id.split("-", 1)[0]
            for record in target_records
        }
        if prefixes != {self.target_source}:
            raise ValueError(
                "All target records must belong to a single source; "
                f"found {sorted(prefixes)}"
            )

        # ---------------------------------------------------------
        # Blocking
        #
        # 'legacy' is the Day 2 dictionary index. 'compact' is the
        # hashed CSR index, which returns the identical candidate set
        # for roughly a tenth of the memory.
        # ---------------------------------------------------------

        index_class = (
            CompactBlockingIndex
            if blocking_backend == "compact"
            else BlockingIndex
        )

        self.blocking_index = index_class(
            target_records,
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
        )

        # ---------------------------------------------------------
        # TF-IDF retrieval
        #
        # The retrievers retain only entity ids and the transposed
        # matrices, so with the compact backend the caller may drop its
        # own reference to the normalized records after construction.
        # ---------------------------------------------------------

        self.name_retriever = CharacterTfidfRetriever(
            target_records,
            field="name",
            ngram_range=self.config.ngram_range,
            top_k=self.config.name_top_k,
            min_score=self.config.name_min_score,
        )

        self.address_retriever = CharacterTfidfRetriever(
            target_records,
            field="address",
            ngram_range=self.config.ngram_range,
            top_k=self.config.address_top_k,
            min_score=self.config.address_min_score,
        )

    # =============================================================
    # ONE S1 RECORD
    # =============================================================

    def generate_for_one(
        self,
        source1: NormalizedRecord,
    ) -> list[CandidateEvidence]:
        """Generate the complete candidate set for one S1 record."""

        evidence: dict[str, CandidateEvidence] = {}

        # ---------------------------------------------------------
        # 1. Blocking routes
        # ---------------------------------------------------------

        # The target source must be passed through so that every candidate
        # carries a real "S2"/"S3" value, as docs/interfaces.md requires.
        blocking_candidates = self.blocking_index.generate(
            source1,
            self.target_source,
        )

        for candidate in blocking_candidates:
            existing = evidence.get(candidate.candidate_entity_id)

            if existing is None:
                evidence[candidate.candidate_entity_id] = candidate
                continue

            for route in candidate.routes:
                existing.add(route)

        # ---------------------------------------------------------
        # 2. TF-IDF retrieval (name, then address)
        # ---------------------------------------------------------

        retrievers = (
            self.name_retriever,
            self.address_retriever,
        )

        for results in (
            retriever.retrieve(source1)
            for retriever in retrievers
        ):

            for result in results:

                candidate_id = result.candidate_entity_id

                candidate = evidence.get(candidate_id)

                if candidate is None:

                    candidate = CandidateEvidence(
                        source1_entity_id=source1.entity_id,
                        candidate_entity_id=candidate_id,
                        target_source=result.target_source,
                    )
                    evidence[candidate_id] = candidate

                candidate.add(
                    result.route,
                    score=result.score,
                    rank=result.rank,
                )

        # ---------------------------------------------------------
        # FINAL SORT
        # ---------------------------------------------------------

        candidates = list(evidence.values())

        candidates.sort(
            key=lambda candidate: (
                # Candidates found by multiple routes first.
                -len(candidate.routes),

                # Strongest retrieval score.
                -candidate.retrieval_score,

                # Better retrieval rank.
                candidate.retrieval_rank,

                # Deterministic tie-breaker.
                candidate.candidate_entity_id,
            )
        )

        return candidates

    # =============================================================
    # MANY S1 RECORDS
    # =============================================================

    def generate(
        self,
        source1_records: Sequence[NormalizedRecord],
    ) -> dict[str, list[CandidateEvidence]]:
        """Generate candidates for multiple S1 records."""

        results: dict[str, list[CandidateEvidence]] = {}

        for source1 in source1_records:

            results[source1.entity_id] = self.generate_for_one(
                source1
            )

        return results


def build_candidate_generator(
    target_records: Sequence[NormalizedRecord],
    *,
    config: CandidateGenerationConfig | None = None,
) -> CandidateGenerator:
    """Convenience constructor."""

    return CandidateGenerator(
        target_records,
        config=config,
    )
