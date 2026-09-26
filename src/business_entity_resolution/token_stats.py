"""Unsupervised token document-frequency statistics for IDF-weighted features.

Statistics are fitted once from source records only (never from labels) and
saved as a JSON artifact. Training and inference load the same artifact, so the
IDF values are identical in both paths. Tokens not seen at fit time (for example
France-only tokens when fitted on train) receive the IDF of a document
frequency of one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable

TOKEN_STATS_VERSION = 1


@dataclass
class TokenStats:
    n_docs: int = 1
    name_df: dict[str, int] = field(default_factory=dict)
    address_df: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._name_cache: dict[str, float] = {}
        self._address_cache: dict[str, float] = {}
        self.max_idf = math.log((self.n_docs + 1) / 2) + 1.0

    def name_idf(self, token: str) -> float:
        value = self._name_cache.get(token)
        if value is None:
            df = self.name_df.get(token, 1)
            value = math.log((self.n_docs + 1) / (df + 1)) + 1.0
            self._name_cache[token] = value
        return value

    def address_idf(self, token: str) -> float:
        value = self._address_cache.get(token)
        if value is None:
            df = self.address_df.get(token, 1)
            value = math.log((self.n_docs + 1) / (df + 1)) + 1.0
            self._address_cache[token] = value
        return value

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": TOKEN_STATS_VERSION,
            "n_docs": self.n_docs,
            "name_df": dict(sorted(self.name_df.items())),
            "address_df": dict(sorted(self.address_df.items())),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | Path) -> "TokenStats":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != TOKEN_STATS_VERSION:
            raise ValueError(f"unsupported token-stats version in {path}")
        return cls(
            n_docs=int(payload["n_docs"]),
            name_df={k: int(v) for k, v in payload["name_df"].items()},
            address_df={k: int(v) for k, v in payload["address_df"].items()},
        )


def fit_token_stats(
    name_token_sets: Iterable[Iterable[str]],
    address_token_sets: Iterable[Iterable[str]],
    *,
    min_df: int = 2,
) -> TokenStats:
    """Fit document frequencies from per-record token collections."""

    name_counter: Counter[str] = Counter()
    address_counter: Counter[str] = Counter()
    n_docs = 0
    for tokens in name_token_sets:
        name_counter.update(set(tokens))
        n_docs += 1
    for tokens in address_token_sets:
        address_counter.update(set(tokens))
    return TokenStats(
        n_docs=max(n_docs, 1),
        name_df={t: c for t, c in name_counter.items() if c >= min_df},
        address_df={t: c for t, c in address_counter.items() if c >= min_df},
    )
