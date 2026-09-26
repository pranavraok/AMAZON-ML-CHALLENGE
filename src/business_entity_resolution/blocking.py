from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


class CandidateEvidence:
    def __init__(
        self,
        source1_entity_id: str,
        candidate_entity_id: str,
        target_source: str,
    ):
        self.source1_entity_id = source1_entity_id
        self.candidate_entity_id = candidate_entity_id
        self.target_source = target_source

        self.routes: List[str] = []
        self.scores: List[float] = []
        self.ranks: List[int] = []

    def add(
        self,
        route: str,
        score: float | None = None,
        rank: int | None = None,
    ) -> None:

        if route not in self.routes:
            self.routes.append(route)

        if score is not None:
            self.scores.append(float(score))

        if rank is not None:
            self.ranks.append(int(rank))

    @property
    def route_count(self) -> int:
        return len(self.routes)

    @property
    def retrieval_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def retrieval_rank(self) -> int:
        return min(self.ranks) if self.ranks else 10**9


def _tokens(text: str) -> List[str]:
    if not text:
        return []

    return [
        x for x in str(text).split()
        if len(x) >= 2
    ]


def _numeric_tokens(text: str) -> List[str]:
    if not text:
        return []

    result = []

    for token in str(text).split():

        digits = "".join(
            ch for ch in token
            if ch.isdigit()
        )

        if len(digits) >= 2:
            result.append(digits)

    return result


def _transliterate(text: str) -> str:

    if not text:
        return ""

    text = str(text).lower()

    if unidecode is not None:
        text = unidecode(text)

    return " ".join(text.split())


class BlockingIndex:

    def __init__(
        self,
        records: Sequence,
        *,
        rare_token_max_postings: int = 50,
        numeric_token_max_postings: int = 100,
        translit_token_max_postings: int = 100,
    ) -> None:

        self.records = list(records)

        self.rare_token_max_postings = rare_token_max_postings
        self.numeric_token_max_postings = numeric_token_max_postings
        self.translit_token_max_postings = translit_token_max_postings

        # Exact indexes
        self.name_index = defaultdict(list)
        self.token_name_index = defaultdict(list)
        self.address_index = defaultdict(list)
        self.combined_index = defaultdict(list)

        # Token indexes
        self.name_token_index = defaultdict(list)
        self.address_token_index = defaultdict(list)
        self.numeric_address_index = defaultdict(list)

        # Global strong keys
        self.global_name_index = defaultdict(list)
        self.global_token_name_index = defaultdict(list)
        self.global_combined_index = defaultdict(list)

        # Transliteration
        self.translit_name_index = defaultdict(list)
        self.translit_token_index = defaultdict(list)

        self._build()

    # ============================================================
    # BUILD
    # ============================================================

    def _build(self):

        for record in self.records:

            entity_id = record.entity_id

            country = (
                getattr(record, "country", "") or ""
            ).strip().lower()

            name = getattr(
                record,
                "name_compact",
                "",
            )

            token_name = getattr(
                record,
                "name_token_sorted",
                "",
            )

            address = getattr(
                record,
                "address_compact",
                "",
            )

            # ----------------------------------------------------
            # Exact name
            # ----------------------------------------------------

            if name:

                self.name_index[
                    (country, name)
                ].append(entity_id)

                self.global_name_index[
                    name
                ].append(entity_id)

            # ----------------------------------------------------
            # Token sorted name
            # ----------------------------------------------------

            if token_name:

                self.token_name_index[
                    (country, token_name)
                ].append(entity_id)

                self.global_token_name_index[
                    token_name
                ].append(entity_id)

            # ----------------------------------------------------
            # Exact address
            # ----------------------------------------------------

            if address:

                self.address_index[
                    (country, address)
                ].append(entity_id)

            # ----------------------------------------------------
            # Combined name + address
            # ----------------------------------------------------

            if name and address:

                self.combined_index[
                    (country, name, address)
                ].append(entity_id)

                self.global_combined_index[
                    (name, address)
                ].append(entity_id)

            # ----------------------------------------------------
            # Name tokens
            # ----------------------------------------------------

            name_unicode = getattr(
                record,
                "name_unicode",
                "",
            )

            for token in _tokens(name_unicode):

                self.name_token_index[
                    (country, token)
                ].append(entity_id)

            # ----------------------------------------------------
            # Address tokens
            # ----------------------------------------------------

            for token in _tokens(address):

                self.address_token_index[
                    (country, token)
                ].append(entity_id)

            # ----------------------------------------------------
            # Numeric address tokens
            # ----------------------------------------------------

            for token in _numeric_tokens(address):

                self.numeric_address_index[
                    (country, token)
                ].append(entity_id)

            # ----------------------------------------------------
            # Transliteration
            # ----------------------------------------------------

            translit = _transliterate(name_unicode)

            if translit:

                self.translit_name_index[
                    (country, translit)
                ].append(entity_id)

                for token in _tokens(translit):

                    self.translit_token_index[
                        (country, token)
                    ].append(entity_id)

    # ============================================================
    # ADD
    # ============================================================

    @staticmethod
    def _add_ids(
        evidence: Dict[str, CandidateEvidence],
        source1_id: str,
        target_source: str,
        ids: Iterable[str],
        route: str,
        *,
        score: float | None = None,
        rank: int | None = None,
    ):

        for entity_id in ids:

            if entity_id == source1_id:
                continue

            if entity_id not in evidence:

                evidence[entity_id] = CandidateEvidence(
                    source1_entity_id=source1_id,
                    candidate_entity_id=entity_id,
                    target_source=target_source,
                )

            evidence[entity_id].add(
                route,
                score=score,
                rank=rank,
            )

    # ============================================================
    # EXACT COMBINED
    # ============================================================

    def exact_combined(
        self,
        record,
        evidence,
        target_source,
    ):

        name = getattr(
            record,
            "name_compact",
            "",
        )

        address = getattr(
            record,
            "address_compact",
            "",
        )

        if not name or not address:
            return

        country = (
            getattr(record, "country", "") or ""
        ).strip().lower()

        ids = self.combined_index.get(
            (country, name, address),
            [],
        )

        self._add_ids(
            evidence,
            record.entity_id,
            target_source,
            ids,
            "exact_combined",
        )

        ids = self.global_combined_index.get(
            (name, address),
            [],
        )

        self._add_ids(
            evidence,
            record.entity_id,
            target_source,
            ids,
            "exact_combined_global",
        )

    # ============================================================
    # EXACT NAME
    # ============================================================

    def exact_name(
        self,
        record,
        evidence,
        target_source,
    ):

        name = getattr(
            record,
            "name_compact",
            "",
        )

        if not name:
            return

        country = (
            getattr(record, "country", "") or ""
        ).strip().lower()

        ids = self.name_index.get(
            (country, name),
            [],
        )

        self._add_ids(
            evidence,
            record.entity_id,
            target_source,
            ids,
            "exact_name",
        )

        ids = self.global_name_index.get(
            name,
            [],
        )

        self._add_ids(
            evidence,
            record.entity_id,
            target_source,
            ids,
            "exact_name_global",
        )

    # ============================================================
    # TOKEN SORTED NAME
    # ============================================================

    def token_sorted_name(
        self,
        record,
        evidence,
        target_source,
    ):

        name = getattr(
            record,
            "name_token_sorted",
            "",
        )

        if not name:
            return

        country = (
            getattr(record, "country", "") or ""
        ).strip().lower()

        ids = self.token_name_index.get(
            (country, name),
            [],
        )

        self._add_ids(
            evidence,
            record.entity_id,
            target_source,
            ids,
            "token_sorted_name",
        )

        ids = self.global_token_name_index.get(
            name,
            [],
        )

        self._add_ids(
            evidence,
            record.entity_id,
            target_source,
            ids,
            "token_sorted_name_global",
        )

    # ============================================================
    # EXACT ADDRESS
    # ============================================================

    def exact_address(
        self,
        record,
        evidence,
        target_source,
    ):

        address = getattr(
            record,
            "address_compact",
            "",
        )

        if not address:
            return

        country = (
            getattr(record, "country", "") or ""
        ).strip().lower()

        ids = self.address_index.get(
            (country, address),
            [],
        )

        self._add_ids(
            evidence,
            record.entity_id,
            target_source,
            ids,
            "exact_address",
        )

    # ============================================================
    # RARE NAME TOKENS
    # ============================================================

    def rare_name_tokens(
        self,
        record,
        evidence,
        target_source,
    ):

        country = (
            getattr(record, "country", "") or ""
        ).strip().lower()

        name = getattr(
            record,
            "name_unicode",
            "",
        )

        for token in _tokens(name):

            ids = self.name_token_index.get(
                (country, token),
                [],
            )

            if not ids:
                continue

            if len(ids) > self.rare_token_max_postings:
                continue

            self._add_ids(
                evidence,
                record.entity_id,
                target_source,
                ids,
                "rare_name_tokens",
            )

    # ============================================================
    # ADDRESS TOKENS
    # ============================================================

    def address_tokens(
        self,
        record,
        evidence,
        target_source,
    ):

        country = (
            getattr(record, "country", "") or ""
        ).strip().lower()

        address = getattr(
            record,
            "address_compact",
            "",
        )

        tokens = _tokens(address)

        if not tokens:
            return

        counts = defaultdict(int)

        for token in tokens:

            ids = self.address_token_index.get(
                (country, token),
                [],
            )

            if not ids:
                continue

            if len(ids) > self.rare_token_max_postings:
                continue

            for entity_id in ids:
                counts[entity_id] += 1

        for entity_id, count in counts.items():

            if count < 2:
                continue

            self._add_ids(
                evidence,
                record.entity_id,
                target_source,
                [entity_id],
                "address_tokens",
            )

    # ============================================================
    # NUMERIC ADDRESS
    # ============================================================

    def numeric_address(
        self,
        record,
        evidence,
        target_source,
    ):

        country = (
            getattr(record, "country", "") or ""
        ).strip().lower()

        address = getattr(
            record,
            "address_compact",
            "",
        )

        for token in _numeric_tokens(address):

            ids = self.numeric_address_index.get(
                (country, token),
                [],
            )

            if not ids:
                continue

            if len(ids) > self.numeric_token_max_postings:
                continue

            self._add_ids(
                evidence,
                record.entity_id,
                target_source,
                ids,
                "numeric_address",
            )

    # ============================================================
    # TRANSLITERATION - FAST ONLY
    # ============================================================

    def transliterated_name(
        self,
        record,
        evidence,
        target_source,
    ):

        country = (
            getattr(record, "country", "") or ""
        ).strip().lower()

        name = getattr(
            record,
            "name_unicode",
            "",
        )

        query = _transliterate(name)

        if not query:
            return

        # Exact transliterated name
        ids = self.translit_name_index.get(
            (country, query),
            [],
        )

        self._add_ids(
            evidence,
            record.entity_id,
            target_source,
            ids,
            "transliterated_exact_name",
        )

        # Transliteration token matching
        query_tokens = _tokens(query)

        counts = defaultdict(int)

        for token in query_tokens:

            ids = self.translit_token_index.get(
                (country, token),
                [],
            )

            if not ids:
                continue

            if len(ids) > self.translit_token_max_postings:
                continue

            for entity_id in ids:
                counts[entity_id] += 1

        for entity_id, count in counts.items():

            # One token is acceptable for very short names.
            if (
                count >= 2
                or len(query_tokens) <= 2
            ):

                self._add_ids(
                    evidence,
                    record.entity_id,
                    target_source,
                    [entity_id],
                    "transliterated_token",
                )

    # ============================================================
    # MAIN
    # ============================================================

    def generate(
        self,
        record,
        target_source: str = "",
    ):

        evidence = {}

        self.exact_combined(
            record,
            evidence,
            target_source,
        )

        self.exact_name(
            record,
            evidence,
            target_source,
        )

        self.token_sorted_name(
            record,
            evidence,
            target_source,
        )

        self.exact_address(
            record,
            evidence,
            target_source,
        )

        self.rare_name_tokens(
            record,
            evidence,
            target_source,
        )

        self.address_tokens(
            record,
            evidence,
            target_source,
        )

        self.numeric_address(
            record,
            evidence,
            target_source,
        )

        self.transliterated_name(
            record,
            evidence,
            target_source,
        )

        # candidate_generation.py expects CandidateEvidence objects
        return list(evidence.values())