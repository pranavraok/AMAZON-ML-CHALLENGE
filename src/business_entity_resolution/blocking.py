"""Country-aware blocking routes for Source 1 -> Source 2/3 candidate generation.

Design notes
------------
Every route is partitioned by country. Countries are treated as open-set
strings, so France and any other test country works without code changes.

Blank addresses are treated as MISSING evidence, never as an exact match.
An empty normalized address therefore cannot satisfy ``exact_address`` or
``exact_combined``.

Unicode evidence is preserved. Routes match on the Unicode-preserving
normalized name. Transliteration is applied only as an *additional* route
layered on top; it never replaces or discards the original Unicode view.

Posting lists are capped so that common tokens cannot generate unbounded
candidate buckets. The caps are configurable per route family.

The exact index field names are validated once at construction time. A
renamed or missing field raises immediately instead of silently disabling a
route.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Sequence

try:  # optional, enables the transliteration routes
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    unidecode = None


# Fields every indexed record must expose. These map to the shared
# normalization contract in business_entity_resolution.normalization.
_REQUIRED_FIELDS = (
    "entity_id",
    "country",
    "name_unicode",
    "name_token_sorted",
    "address_unicode",
)

# Upper bound on how many address tokens take part in pair-key generation.
# An address with k tokens yields k*(k-1)/2 pair keys, so this keeps index
# construction linear in a predictable way for pathologically long addresses.
_MAX_PAIR_TOKENS = 12


class CandidateEvidence:
    """Accumulated retrieval evidence for one S1 -> S2/S3 candidate pair."""

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

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CandidateEvidence({self.candidate_entity_id!r}, "
            f"target_source={self.target_source!r}, routes={self.routes!r})"
        )


def _country(record) -> str:
    return str(getattr(record, "country", "") or "").strip().lower()


def _tokens(text: str) -> List[str]:
    """Whitespace tokens of length >= 2. Single characters carry no signal."""
    if not text:
        return []
    return [x for x in str(text).split() if len(x) >= 2]


def _numeric_tokens(text: str) -> List[str]:
    """Numeric tokens of length >= 2, with leading zeros stripped.

    Stripping leading zeros lets '14' and '0014' agree, which is a real
    pattern in the data (S1 ``F. No.-14-A`` vs target ``F. No.-0014-a``).
    """
    if not text:
        return []

    result: list[str] = []

    for token in str(text).split():
        digits = "".join(ch for ch in token if ch.isdigit())

        if len(digits) < 2:
            continue

        trimmed = digits.lstrip("0")

        result.append(trimmed if trimmed else "0")

    return result


def _transliterate(text: str) -> str:
    """Romanize text for the optional transliteration routes.

    Returns the input unchanged when ``unidecode`` is unavailable, so the
    transliteration routes degrade into duplicates of the Unicode routes
    instead of crashing or silently corrupting evidence.
    """
    if not text:
        return ""

    text = str(text).lower()

    if unidecode is not None:
        text = unidecode(text)

    return " ".join(text.split())


def _address_pair_keys(text: str) -> List[tuple]:
    """Sorted address-token pair keys for co-occurrence blocking.

    Individual address tokens such as 'delhi' or 'new' are far too common to
    block on, so a per-token posting cap rejects every record that shares them.
    A *pair* of common tokens is far more selective: ('delhi', 'new') is a
    small bucket even though both tokens individually are not.
    """
    tokens = sorted(set(_tokens(text)))

    if len(tokens) < 2:
        return []

    if len(tokens) > _MAX_PAIR_TOKENS:
        tokens = tokens[:_MAX_PAIR_TOKENS]

    return list(combinations(tokens, 2))


class BlockingIndex:
    """Inverted indexes over target records for cheap country-aware blocking."""

    def __init__(
        self,
        records: Sequence,
        *,
        rare_token_max_postings: int = 50,
        numeric_token_max_postings: int = 100,
        translit_token_max_postings: int = 100,
        address_token_max_postings: int = 50,
        address_pair_max_postings: int = 5,
    ) -> None:

        self.records = list(records)

        if rare_token_max_postings <= 0:
            raise ValueError("rare_token_max_postings must be positive")
        if numeric_token_max_postings <= 0:
            raise ValueError("numeric_token_max_postings must be positive")
        if translit_token_max_postings <= 0:
            raise ValueError("translit_token_max_postings must be positive")
        if address_token_max_postings <= 0:
            raise ValueError("address_token_max_postings must be positive")
        if address_pair_max_postings <= 0:
            raise ValueError("address_pair_max_postings must be positive")

        self.rare_token_max_postings = rare_token_max_postings
        self.numeric_token_max_postings = numeric_token_max_postings
        self.translit_token_max_postings = translit_token_max_postings
        self.address_token_max_postings = address_token_max_postings
        self.address_pair_max_postings = address_pair_max_postings

        # Exact keys, country-partitioned.
        self.name_index: Dict[tuple, List[str]] = defaultdict(list)
        self.token_name_index: Dict[tuple, List[str]] = defaultdict(list)
        self.address_index: Dict[tuple, List[str]] = defaultdict(list)
        self.combined_index: Dict[tuple, List[str]] = defaultdict(list)

        # Token keys, country-partitioned.
        self.name_token_index: Dict[tuple, List[str]] = defaultdict(list)
        self.address_token_index: Dict[tuple, List[str]] = defaultdict(list)
        self.address_pair_index: Dict[tuple, List[str]] = defaultdict(list)
        self.numeric_address_index: Dict[tuple, List[str]] = defaultdict(list)

        # Transliteration keys, country-partitioned.
        self.translit_name_index: Dict[tuple, List[str]] = defaultdict(list)
        self.translit_token_index: Dict[tuple, List[str]] = defaultdict(list)

        self._build()

    # ================================================================
    # BUILD
    # ================================================================

    def _validate_records(self) -> None:
        """Fail loudly if the indexed records do not match the shared contract.

        The previous implementation read ``name_compact``/``address_compact``
        through ``getattr(..., "")``. Those attributes do not exist on
        ``NormalizedRecord``, so every route that used them silently returned
        nothing. Validating once here makes that class of bug impossible.
        """
        if not self.records:
            raise ValueError("BlockingIndex requires at least one record")

        missing = [
            field
            for field in _REQUIRED_FIELDS
            if not hasattr(self.records[0], field)
        ]
        if missing:
            raise ValueError(
                "indexed records are missing required normalized fields: "
                f"{missing}. Expected objects produced by "
                "business_entity_resolution.normalization.normalize_record()."
            )

    def _build(self) -> None:

        self._validate_records()

        for record in self.records:

            entity_id = record.entity_id
            country = _country(record)

            name = record.name_unicode
            token_name = record.name_token_sorted
            address = record.address_unicode

            # ----------------------------------------------------
            # Exact name / combined keys
            # ----------------------------------------------------

            if name:
                self.name_index[(country, name)].append(entity_id)

            if token_name:
                self.token_name_index[(country, token_name)].append(entity_id)

            if address:
                self.address_index[(country, address)].append(entity_id)

            # A combined key needs real evidence on both sides. A blank
            # address is missing evidence, not a shared value.
            if name and address:
                self.combined_index[
                    (country, name, address)
                ].append(entity_id)

            # ----------------------------------------------------
            # Name tokens
            # ----------------------------------------------------

            for token in _tokens(name):
                self.name_token_index[(country, token)].append(entity_id)

            # ----------------------------------------------------
            # Address tokens
            # ----------------------------------------------------

            for token in _tokens(address):
                self.address_token_index[(country, token)].append(entity_id)

            # ----------------------------------------------------
            # Address token-pair co-occurrence keys
            # ----------------------------------------------------

            for first, second in _address_pair_keys(address):
                self.address_pair_index[
                    (country, first, second)
                ].append(entity_id)

            # ----------------------------------------------------
            # Numeric address tokens
            # ----------------------------------------------------

            for token in _numeric_tokens(address):
                self.numeric_address_index[
                    (country, token)
                ].append(entity_id)

            # ----------------------------------------------------
            # Transliteration (additional evidence, never a replacement)
            # ----------------------------------------------------

            translit = _transliterate(name)

            if translit:
                self.translit_name_index[(country, translit)].append(
                    entity_id
                )

                for token in _tokens(translit):
                    self.translit_token_index[
                        (country, token)
                    ].append(entity_id)

    # ================================================================
    # ADD
    # ================================================================

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
    ) -> None:

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

    def _add_key(
        self,
        evidence: Dict[str, CandidateEvidence],
        record,
        target_source: str,
        index: Dict[tuple, List[str]],
        key: tuple,
        route: str,
    ) -> None:
        ids = index.get(key)
        if ids:
            self._add_ids(
                evidence,
                record.entity_id,
                target_source,
                ids,
                route,
            )

    # ================================================================
    # EXACT COMBINED
    # ================================================================

    def exact_combined(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        name = record.name_unicode
        address = record.address_unicode

        # Both sides must carry real evidence.
        if not name or not address:
            return

        self._add_key(
            evidence,
            record,
            target_source,
            self.combined_index,
            (_country(record), name, address),
            "exact_combined",
        )

    # ================================================================
    # EXACT NAME
    # ================================================================

    def exact_name(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        name = record.name_unicode

        if not name:
            return

        self._add_key(
            evidence,
            record,
            target_source,
            self.name_index,
            (_country(record), name),
            "exact_name",
        )

    # ================================================================
    # TOKEN SORTED NAME
    # ================================================================

    def token_sorted_name(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        token_name = record.name_token_sorted

        if not token_name:
            return

        self._add_key(
            evidence,
            record,
            target_source,
            self.token_name_index,
            (_country(record), token_name),
            "token_sorted_name",
        )

    # ================================================================
    # EXACT ADDRESS
    # ================================================================

    def exact_address(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        address = record.address_unicode

        # A blank address is missing evidence, not an exact match.
        if not address:
            return

        self._add_key(
            evidence,
            record,
            target_source,
            self.address_index,
            (_country(record), address),
            "exact_address",
        )

    # ================================================================
    # RARE NAME TOKENS
    # ================================================================

    def rare_name_tokens(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        country = _country(record)

        for token in _tokens(record.name_unicode):

            ids = self.name_token_index.get((country, token))

            if not ids:
                continue

            # Common tokens are not informative. Skip rather than flood.
            if len(ids) > self.rare_token_max_postings:
                continue

            self._add_ids(
                evidence,
                record.entity_id,
                target_source,
                ids,
                "rare_name_tokens",
            )

    # ================================================================
    # ADDRESS TOKENS
    # ================================================================

    def address_tokens(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        country = _country(record)

        tokens = _tokens(record.address_unicode)

        if not tokens:
            return

        counts: dict[str, int] = defaultdict(int)

        for token in tokens:

            ids = self.address_token_index.get((country, token))

            if not ids:
                continue

            if len(ids) > self.address_token_max_postings:
                continue

            for entity_id in ids:
                counts[entity_id] += 1

        # Require more than one shared address token so that a single very
        # common token cannot qualify a candidate on its own.
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

    # ================================================================
    # ADDRESS TOKEN PAIRS
    # ================================================================

    def address_token_pairs(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        country = _country(record)

        # A blank address yields no pair keys, so missing evidence can never
        # produce a match here.
        for first, second in _address_pair_keys(record.address_unicode):

            ids = self.address_pair_index.get((country, first, second))

            if not ids:
                continue

            # The cap applies to the COMBINED bucket. Co-occurring common
            # tokens stay usable because the pair is far more selective
            # than either token alone.
            if len(ids) > self.address_pair_max_postings:
                continue

            self._add_ids(
                evidence,
                record.entity_id,
                target_source,
                ids,
                "address_token_pair",
            )

    # ================================================================
    # NUMERIC ADDRESS
    # ================================================================

    def numeric_address(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        country = _country(record)

        for token in _numeric_tokens(record.address_unicode):

            ids = self.numeric_address_index.get((country, token))

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

    # ================================================================
    # TRANSLITERATION
    # ================================================================

    def transliterated_name(
        self,
        record,
        evidence: Dict[str, CandidateEvidence],
        target_source: str,
    ) -> None:

        country = _country(record)

        query = _transliterate(record.name_unicode)

        if not query:
            return

        self._add_key(
            evidence,
            record,
            target_source,
            self.translit_name_index,
            (country, query),
            "transliterated_exact_name",
        )

        query_tokens = _tokens(query)

        if not query_tokens:
            return

        counts: dict[str, int] = defaultdict(int)

        for token in query_tokens:

            ids = self.translit_token_index.get((country, token))

            if not ids:
                continue

            if len(ids) > self.translit_token_max_postings:
                continue

            for entity_id in ids:
                counts[entity_id] += 1

        for entity_id, count in counts.items():

            # One shared token is acceptable only for very short names.
            if count >= 2 or len(query_tokens) <= 2:

                self._add_ids(
                    evidence,
                    record.entity_id,
                    target_source,
                    [entity_id],
                    "transliterated_token",
                )

    # ================================================================
    # MAIN
    # ================================================================

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
