"""Candidate generation pipeline for Source 1 -> Source 2/3.

Combines cheap blocking routes with character n-gram TF-IDF retrieval.

The final candidate set produced here is the set that should be passed
to the matching model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from business_entity_resolution.blocking import (
    BlockingIndex,
    CandidateEvidence,
)
from business_entity_resolution.normalization import NormalizedRecord
from business_entity_resolution.retrieval import (
    CharacterTfidfRetriever,
)


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Configuration for candidate generation."""

    name_top_k: int = 20
    address_top_k: int = 20

    name_min_score: float = 0.0
    address_min_score: float = 0.0

    rare_token_max_postings: int = 50
    numeric_token_max_postings: int = 100


class CandidateGenerator:
    """Generate candidates for S1 against one target source."""

    def __init__(
        self,
        target_records: Sequence[NormalizedRecord],
        *,
        config: CandidateGenerationConfig | None = None,
    ) -> None:

        if not target_records:
            raise ValueError(
                "CandidateGenerator requires target records"
            )

        self.config = config or CandidateGenerationConfig()

        self.target_source = target_records[0].entity_id.split(
            "-", 1
        )[0]

        if self.target_source not in {"S2", "S3"}:
            raise ValueError(
                "Target records must belong to S2 or S3"
            )

        # ---------------------------------------------------------
        # Blocking
        # ---------------------------------------------------------

        self.blocking_index = BlockingIndex(
            target_records,
            rare_token_max_postings=self.config.rare_token_max_postings,
            numeric_token_max_postings=self.config.numeric_token_max_postings,
        )

        # ---------------------------------------------------------
        # TF-IDF retrieval
        # ---------------------------------------------------------

        self.name_retriever = CharacterTfidfRetriever(
            target_records,
            field="name",
            ngram_range=(3, 5),
            top_k=self.config.name_top_k,
            min_score=self.config.name_min_score,
        )

        self.address_retriever = CharacterTfidfRetriever(
            target_records,
            field="address",
            ngram_range=(3, 5),
            top_k=self.config.address_top_k,
            min_score=self.config.address_min_score,
        )

    # =============================================================
    # MERGING
    # =============================================================

    @staticmethod
    def _merge_evidence(
        destination: dict[str, CandidateEvidence],
        source: dict[str, CandidateEvidence],
    ) -> None:
        """Merge candidate evidence from one retrieval route."""

        for candidate_id, candidate in source.items():

            if candidate_id not in destination:
                destination[candidate_id] = candidate
                continue

            existing = destination[candidate_id]

            for route in candidate.routes:
                if route not in existing.routes:
                    existing.routes.append(route)

            existing.scores.extend(candidate.scores)
            existing.ranks.extend(candidate.ranks)

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

        blocking_candidates = self.blocking_index.generate(
            source1
        )

        for candidate in blocking_candidates:
            evidence[candidate.candidate_entity_id] = candidate

        # ---------------------------------------------------------
        # 2. Name TF-IDF
        # ---------------------------------------------------------

        name_results = self.name_retriever.retrieve(source1)

        for result in name_results:

            candidate_id = result.candidate_entity_id

            if candidate_id not in evidence:

                evidence[candidate_id] = CandidateEvidence(
                    source1_entity_id=source1.entity_id,
                    candidate_entity_id=candidate_id,
                    target_source=result.target_source,
                )

            evidence[candidate_id].add(
                result.route,
                score=result.score,
                rank=result.rank,
            )

        # ---------------------------------------------------------
        # 3. Address TF-IDF
        # ---------------------------------------------------------

        address_results = self.address_retriever.retrieve(
            source1
        )

        for result in address_results:

            candidate_id = result.candidate_entity_id

            if candidate_id not in evidence:

                evidence[candidate_id] = CandidateEvidence(
                    source1_entity_id=source1.entity_id,
                    candidate_entity_id=candidate_id,
                    target_source=result.target_source,
                )

            evidence[candidate_id].add(
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