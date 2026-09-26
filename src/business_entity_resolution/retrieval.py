"""
Fast character n-gram TF-IDF retrieval for candidate generation.

Maintains country-aware TF-IDF indexes for:
- business names
- business addresses

The public API remains compatible with candidate_generation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from business_entity_resolution.normalization import NormalizedRecord


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieval result for an S1 -> S2/S3 candidate."""

    candidate_entity_id: str
    target_source: str
    route: str
    rank: int
    score: float


class CharacterTfidfRetriever:

    def __init__(
        self,
        records: Sequence[NormalizedRecord],
        *,
        field: str,
        ngram_range: tuple[int, int] = (3, 5),
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> None:

        if not records:
            raise ValueError(
                "TF-IDF retriever requires at least one record"
            )

        if field not in {"name", "address"}:
            raise ValueError(
                "field must be either 'name' or 'address'"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        if (
            ngram_range[0] <= 0
            or ngram_range[1] < ngram_range[0]
        ):
            raise ValueError(
                "Invalid ngram_range"
            )

        self.records = list(records)

        self.field = field
        self.ngram_range = ngram_range
        self.top_k = top_k
        self.min_score = min_score

        self.target_source = self._get_target_source()

        # Country-specific indexes.
        #
        # Only the TRANSPOSED matrix is retained. Queries always evaluate
        # ``query_vector @ target_matrix.T``, so keeping the forward matrix
        # as well doubled the largest allocation for no benefit. The forward
        # matrix is discarded as soon as the transpose exists.
        self._country_transposed_matrices: dict[
            str,
            csr_matrix
        ] = {}

        # Entity ids per country, in matrix-row order. The records
        # themselves are not retained: at full scale they are the single
        # largest allocation, and only the id is needed to break score ties.
        self._country_entity_ids: dict[
            str,
            list[str]
        ] = {}

        self._country_vectorizers: dict[
            str,
            TfidfVectorizer
        ] = {}

        self._build_index()

        # Release the target records. Nothing below this point needs them.
        self.records = []

    # ==============================================================
    # SOURCE
    # ==============================================================

    def _get_target_source(self) -> str:

        prefix = self.records[0].entity_id.split(
            "-",
            1
        )[0]

        if prefix not in {"S2", "S3"}:
            raise ValueError(
                "TF-IDF target records must belong to S2 or S3"
            )

        for record in self.records:

            current_prefix = record.entity_id.split(
                "-",
                1
            )[0]

            if current_prefix != prefix:
                raise ValueError(
                    "All records in one TF-IDF retriever "
                    "must use the same source"
                )

        return prefix

    # ==============================================================
    # TEXT
    # ==============================================================

    def _get_text(
        self,
        record: NormalizedRecord,
    ) -> str:

        if self.field == "name":
            return record.name_unicode

        return record.address_unicode

    # ==============================================================
    # BUILD INDEX
    # ==============================================================

    def _build_index(self) -> None:

        grouped: dict[
            str,
            list[NormalizedRecord]
        ] = {}

        for record in self.records:

            grouped.setdefault(
                record.country,
                []
            ).append(record)

        for country, country_records in grouped.items():

            # Blank addresses are missing evidence.
            if self.field == "address":

                country_records = [
                    record
                    for record in country_records
                    if not record.address_missing
                ]

            if not country_records:
                continue

            texts = [
                self._get_text(record)
                for record in country_records
            ]

            if not any(texts):
                continue

            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.ngram_range,
                lowercase=False,
                norm="l2",
                sublinear_tf=True,
                dtype=np.float32,
            )

            matrix = vectorizer.fit_transform(
                texts
            ).tocsr()

            self._country_vectorizers[
                country
            ] = vectorizer

            self._country_entity_ids[
                country
            ] = [
                record.entity_id for record in country_records
            ]

            # Store the transpose and drop the forward matrix. Rows of the
            # transposed matrix are the target records, in the same order.
            self._country_transposed_matrices[
                country
            ] = matrix.transpose().tocsr()

            del matrix
            del texts

    # ==============================================================
    # RETRIEVE
    # ==============================================================

    def retrieve(
        self,
        source1: NormalizedRecord,
    ) -> list[RetrievalResult]:

        # Blank address -> no address retrieval.
        if (
            self.field == "address"
            and source1.address_missing
        ):
            return []

        query_text = self._get_text(
            source1
        )

        if not query_text:
            return []

        country = source1.country

        if country not in self._country_vectorizers:
            return []

        entity_ids = self._country_entity_ids[
            country
        ]

        vectorizer = self._country_vectorizers[
            country
        ]

        transposed_matrix = (
            self._country_transposed_matrices[
                country
            ]
        )

        # ----------------------------------------------------------
        # Transform ONE query.
        # ----------------------------------------------------------

        query_vector = vectorizer.transform(
            [query_text]
        ).tocsr()

        # ----------------------------------------------------------
        # FAST COSINE SIMILARITY
        #
        # query_vector shape:
        #     1 x features
        #
        # transposed matrix shape:
        #     features x records
        #
        # result:
        #     1 x records
        # ----------------------------------------------------------

        scores_sparse = (
            query_vector @ transposed_matrix
        )

        # Convert only the single resulting row.
        scores = np.asarray(
            scores_sparse.toarray()
        ).ravel()

        if scores.size == 0:
            return []

        # ----------------------------------------------------------
        # Find candidates above threshold.
        # ----------------------------------------------------------

        valid_indices = np.flatnonzero(
            scores >= self.min_score
        )

        if valid_indices.size == 0:
            return []

        # ----------------------------------------------------------
        # Top K without sorting everything.
        # ----------------------------------------------------------

        if valid_indices.size > self.top_k:

            valid_scores = scores[
                valid_indices
            ]

            top_positions = np.argpartition(
                valid_scores,
                -self.top_k
            )[-self.top_k:]

            candidate_indices = (
                valid_indices[
                    top_positions
                ]
            )

        else:

            candidate_indices = valid_indices

        # ----------------------------------------------------------
        # Deterministic ordering.
        # ----------------------------------------------------------

        ranked_indices = sorted(
            candidate_indices.tolist(),
            key=lambda index: (
                -float(scores[index]),
                entity_ids[index],
            ),
        )

        results: list[
            RetrievalResult
        ] = []

        rank = 1

        for index in ranked_indices:

            candidate_entity_id = entity_ids[index]

            # Safety: no self-match.
            if (
                candidate_entity_id
                == source1.entity_id
            ):
                continue

            results.append(
                RetrievalResult(
                    candidate_entity_id=(
                        candidate_entity_id
                    ),
                    target_source=(
                        self.target_source
                    ),
                    route=f"{self.field}_tfidf",
                    rank=rank,
                    score=float(
                        scores[index]
                    ),
                )
            )

            rank += 1

        return results


# ==================================================================
# FACTORY FUNCTIONS
# ==================================================================

def build_name_retriever(
    records: Sequence[NormalizedRecord],
    *,
    top_k: int = 20,
    min_score: float = 0.0,
) -> CharacterTfidfRetriever:

    return CharacterTfidfRetriever(
        records,
        field="name",
        ngram_range=(3, 5),
        top_k=top_k,
        min_score=min_score,
    )


def build_address_retriever(
    records: Sequence[NormalizedRecord],
    *,
    top_k: int = 20,
    min_score: float = 0.0,
) -> CharacterTfidfRetriever:

    return CharacterTfidfRetriever(
        records,
        field="address",
        ngram_range=(3, 5),
        top_k=top_k,
        min_score=min_score,
    )