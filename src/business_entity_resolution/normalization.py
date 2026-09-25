"""Shared, country-agnostic normalization contract.

The normalizer deliberately keeps a Unicode representation and an accent-folded
ASCII representation. Downstream modules may compare either view, but must not
discard local-script evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Mapping


_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class NormalizedRecord:
    entity_id: str
    country: str
    name_unicode: str
    name_ascii: str
    name_token_sorted: str
    address_unicode: str
    address_ascii: str
    address_token_sorted: str
    name_tokens: tuple[str, ...]
    address_tokens: tuple[str, ...]
    address_numbers: tuple[str, ...]
    address_missing: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _replace_separators(text: str) -> str:
    characters: list[str] = []
    for character in text:
        # Unicode combining marks are part of many Indic-script graphemes.
        # Treating them as punctuation would split or corrupt local-script words.
        if character.isalnum() or unicodedata.category(character).startswith("M"):
            characters.append(character)
        else:
            characters.append(" ")
    return _SPACE_RE.sub(" ", "".join(characters)).strip()


def normalize_unicode(value: object) -> str:
    """Return a Unicode-preserving, punctuation-normalized comparison string."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = text.replace("&", " and ")
    return _replace_separators(text)


def fold_ascii(value: object) -> str:
    """Return an accent-folded ASCII view without replacing the Unicode view."""

    unicode_text = normalize_unicode(value)
    decomposed = unicodedata.normalize("NFKD", unicode_text)
    ascii_text = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character) and ord(character) < 128
    )
    return _SPACE_RE.sub(" ", ascii_text).strip()


def tokenize(normalized_text: str) -> tuple[str, ...]:
    return tuple(token for token in normalized_text.split(" ") if token)


def token_sorted(normalized_text: str) -> str:
    return " ".join(sorted(tokenize(normalized_text)))


def extract_number_tokens(normalized_text: str) -> tuple[str, ...]:
    return tuple(_NUMBER_RE.findall(normalized_text))


def normalize_record(record: Mapping[str, object]) -> NormalizedRecord:
    """Normalize one source record using the shared project contract."""

    missing = [
        key
        for key in ("entity_id", "business_name", "business_address", "country")
        if key not in record
    ]
    if missing:
        raise ValueError(f"source record is missing required keys: {missing}")

    name_unicode = normalize_unicode(record["business_name"])
    address_unicode = normalize_unicode(record["business_address"])
    return NormalizedRecord(
        entity_id=str(record["entity_id"]).strip(),
        country=str(record["country"]).strip(),
        name_unicode=name_unicode,
        name_ascii=fold_ascii(record["business_name"]),
        name_token_sorted=token_sorted(name_unicode),
        address_unicode=address_unicode,
        address_ascii=fold_ascii(record["business_address"]),
        address_token_sorted=token_sorted(address_unicode),
        name_tokens=tokenize(name_unicode),
        address_tokens=tokenize(address_unicode),
        address_numbers=extract_number_tokens(address_unicode),
        address_missing=not address_unicode,
    )
