"""Record-level text views used by the pair-feature module (Person 3).

Everything here is deterministic, hand-written and country-agnostic. No external
dictionaries, geocoders or registries are used. All views build on the shared
``normalization.normalize_unicode`` contract; the Unicode view is never
discarded, the Latin view is an *additional* comparison surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from business_entity_resolution.normalization import normalize_unicode

# ---------------------------------------------------------------------------
# Indic-script romanisation.
# The nine major Indic Unicode blocks share the ISCII layout, so one offset
# table romanises Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
# Telugu, Kannada and Malayalam. The output is approximate on purpose: it is
# only compared through fuzzy and phonetic-skeleton similarity.
# ---------------------------------------------------------------------------

_INDIC_BLOCKS = tuple(range(0x0900, 0x0D80, 0x80))

_INDIC_VOWELS = {
    0x04: "a", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_INDIC_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "rh",
    0x5E: "f", 0x5F: "y",
}
_INDIC_MATRAS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x62: "li", 0x63: "li",
}
_INDIC_NASALS = {0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA = 0x4D
_NUKTA = 0x3C


def _indic_offset(character: str) -> int | None:
    code = ord(character)
    if 0x0900 <= code < 0x0D80:
        return code - (code & ~0x7F)
    return None


def romanize_indic(text: str) -> str:
    """Approximate romanisation for the nine ISCII-aligned Indic scripts."""

    out: list[str] = []
    pending_inherent = False
    for character in text:
        offset = _indic_offset(character)
        if offset is None:
            # Word-final inherent vowels are usually silent ("bharat", not "bharata").
            pending_inherent = False
            out.append(character)
            continue
        if offset in _INDIC_CONSONANTS:
            if pending_inherent:
                out.append("a")
            out.append(_INDIC_CONSONANTS[offset])
            pending_inherent = True
        elif offset in _INDIC_MATRAS:
            out.append(_INDIC_MATRAS[offset])
            pending_inherent = False
        elif offset == _VIRAMA:
            pending_inherent = False
        elif offset == _NUKTA:
            continue
        elif offset in _INDIC_VOWELS:
            if pending_inherent:
                out.append("a")
                pending_inherent = False
            out.append(_INDIC_VOWELS[offset])
        elif offset in _INDIC_NASALS:
            if pending_inherent:
                out.append("a")
                pending_inherent = False
            out.append(_INDIC_NASALS[offset])
        elif 0x66 <= offset <= 0x6F:
            if pending_inherent:
                out.append("a")
                pending_inherent = False
            out.append(str(offset - 0x66))
        # Other signs (avagraha, dandas, length marks) carry no comparison value.
    return "".join(out)


def _latin_view(unicode_text: str) -> tuple[str, bool]:
    """Return (ASCII Latin view, has_non_latin_letters)."""

    has_indic = any(0x0900 <= ord(c) < 0x0D80 for c in unicode_text)
    text = romanize_indic(unicode_text) if has_indic else unicode_text
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_chars: list[str] = []
    non_latin = has_indic
    for character in decomposed:
        if unicodedata.combining(character):
            continue
        if ord(character) < 128:
            ascii_chars.append(character)
        elif character.isalpha():
            non_latin = True
            ascii_chars.append(" ")
        else:
            ascii_chars.append(" ")
    return " ".join("".join(ascii_chars).split()), non_latin


# ---------------------------------------------------------------------------
# Phonetic skeleton: aggressive, script-neutral key used to compare romanised
# Indic names with Latin names and to absorb vowel/voicing typos.
# ---------------------------------------------------------------------------

_SKELETON_RULES = (
    # "ch" -> "s" because Tamil/Bengali scripts use one letter for ch/s sounds;
    # v/w/b and voiced/unvoiced stops merge because several scripts do not
    # distinguish them.
    ("ph", "f"), ("ch", "s"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"),
    ("w", "v"), ("v", "p"), ("z", "j"), ("b", "p"), ("d", "t"), ("g", "k"),
    ("h", ""),
)
_VOWELS_RE = re.compile(r"[aeiouy]")
_REPEAT_RE = re.compile(r"(.)\1+")
_OCR_DIGITS = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "8": "b"})


def skeleton_token(token: str) -> str:
    if token.isdigit():
        return token
    text = token.translate(_OCR_DIGITS) if any(c.isalpha() for c in token) else token
    for old, new in _SKELETON_RULES:
        text = text.replace(old, new)
    first = text[:1]
    rest = _VOWELS_RE.sub("", text[1:])
    text = first + rest
    return _REPEAT_RE.sub(r"\1", text)


def skeleton(tokens: tuple[str, ...]) -> str:
    return " ".join(s for s in (skeleton_token(t) for t in tokens) if s)


# ---------------------------------------------------------------------------
# Name vocabularies (hand-written, multi-country, not tied to one country).
# ---------------------------------------------------------------------------

LEGAL_TOKENS = frozenset(
    """
    llc inc incorporated corp corporation co company companies ltd limited pvt private
    llp lp plc pllc pc pa na ltda gmbh ag bv nv sa sas sasu sarl eurl snc sci scop
    sprl srl spa cie et associates assoc opc trust the and of
    """.split()
)
HONORIFIC_TOKENS = frozenset("mr mrs ms miss dr shri sri smt m s messrs mme mlle m".split())
ALIAS_MARKERS = re.compile(
    r"\b(?:d ?b ?a|f ?k ?a|a ?k ?a|t ?a|formerly|trading as|doing business as)\b"
)
# Legal-form families; a pair whose families conflict (e.g. llc vs inc) is weak
# negative evidence, while a family absent on one side is neutral.
LEGAL_FAMILY = {
    "llc": "llc", "pllc": "llc",
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "ltd": "ltd", "limited": "ltd",
    "llp": "llp", "lp": "lp", "plc": "plc",
    "pvt": "pvt", "private": "pvt",
    "sa": "sa", "sas": "sas", "sasu": "sas", "sarl": "sarl", "eurl": "eurl",
    "gmbh": "gmbh",
}

# Skeletons of legal words so that romanised Indic forms ("praivet limited")
# are recognised as legal suffixes as well.
_LEGAL_SKELETONS = frozenset(
    skeleton_token(t) for t in LEGAL_TOKENS | {"praivet", "limited", "limitid"} if len(t) > 2
)

# ---------------------------------------------------------------------------
# Address vocabularies.
# ---------------------------------------------------------------------------

ADDRESS_ABBREVIATIONS = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "cir": "circle", "pl": "place", "pkwy": "parkway", "hwy": "highway", "trl": "trail",
    "ter": "terrace", "sq": "square", "apt": "apartment", "ste": "suite", "fl": "floor",
    "flr": "floor", "bldg": "building", "no": "number", "nr": "near", "opp": "opposite",
    "mkt": "market", "ext": "extension", "sec": "sector", "ph": "phase", "hno": "house",
    "h": "house", "e": "east", "w": "west", "n": "north", "s": "south", "mt": "mount",
    "ft": "fort", "pt": "point", "twp": "township", "cres": "crescent", "sqr": "square",
    "chs": "society", "soc": "society", "nagr": "nagar", "marg": "marg",
    "fg": "faubourg", "fbg": "faubourg", "imp": "impasse", "che": "chemin",
    "rte": "route", "all": "allee", "pce": "place",
}
# Region names collapse to short codes when a whole comma-separated component is
# the region name, so "Tennessee"/"TN" and "Tamil Nadu"/"TN" both become "tn".
# Code collisions across countries are harmless because pairs never cross
# countries. Ordinary words inside a street ("rue de la paix") are untouched.
_REGION_CODE_NAMES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
    "mh": "maharashtra", "hr": "haryana", "wb": "west bengal", "ka": "karnataka",
    "tg": "telangana", "ts": "telangana", "up": "uttar pradesh", "mp": "madhya pradesh",
    "ap": "andhra pradesh", "dl": "delhi", "gj": "gujarat", "rj": "rajasthan",
    "kl": "kerala", "pb": "punjab", "od": "odisha", "br": "bihar", "jh": "jharkhand",
    "cg": "chhattisgarh", "uk": "uttarakhand", "hp": "himachal pradesh", "as": "assam",
    "jk": "jammu and kashmir",
}
REGION_NAME_TO_CODE = {name: code for code, name in _REGION_CODE_NAMES.items()}
REGION_NAME_TO_CODE.update({"tamil nadu": "tn", "orissa": "od", "odisha": "od",
                            "uttaranchal": "uk", "pondicherry": "py", "puducherry": "py",
                            "goa": "ga", "chandigarh": "ch", "new delhi": "new delhi"})
MISSING_TOKENS = frozenset({"none", "null", "nan", "na", "n a", "nil", "unknown", "not available"})
_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th|er|e|eme|ere)$")
_NUM_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class PreparedRecord:
    """Record-level derived views; computed once per record, reused per pair."""

    name_unicode: str
    name_latin: str
    name_core: str
    name_core_sorted: str
    name_nospace: str
    name_skeleton: str
    name_core_skeleton: str
    name_acronym: str
    name_alias_parts: tuple[str, ...]
    name_legal_families: frozenset[str]
    name_tokens: frozenset[str]
    name_core_tokens: tuple[str, ...]
    name_core_skeleton_tokens: tuple[str, ...]
    name_non_latin: bool
    address_unicode: str
    address_latin: str
    address_sorted: str
    address_skeleton: str
    address_tokens: frozenset[str]
    address_alpha_tokens: frozenset[str]
    address_numbers: frozenset[str]
    address_longest_number: str
    address_first_component: str
    address_missing: bool
    address_non_latin: bool


def _strip_leading_zeros(number: str) -> str:
    stripped = number.lstrip("0")
    return stripped or "0"


def prepare_name(raw: object) -> dict[str, object]:
    unicode_text = normalize_unicode(raw if raw is not None else "")
    latin, non_latin = _latin_view(unicode_text)
    tokens = tuple(latin.split())

    def is_core(token: str) -> bool:
        if token in LEGAL_TOKENS or token in HONORIFIC_TOKENS:
            return False
        # Romanised local-script legal words ("praivet limitet") only.
        return not (non_latin and skeleton_token(token) in _LEGAL_SKELETONS)

    alias_split = ALIAS_MARKERS.split(latin)
    core_tokens = tuple(t for part in alias_split for t in part.split() if is_core(t))
    if not core_tokens:
        core_tokens = tokens
    alias_parts = tuple(" ".join(t for t in part.split() if is_core(t)) for part in alias_split)
    alias_parts = tuple(p for p in alias_parts if p) or (" ".join(core_tokens),)
    families = frozenset(LEGAL_FAMILY[t] for t in tokens if t in LEGAL_FAMILY)
    core = " ".join(core_tokens)
    return {
        "name_unicode": unicode_text,
        "name_latin": latin,
        "name_core": core,
        "name_core_sorted": " ".join(sorted(core_tokens)),
        "name_nospace": "".join(core_tokens),
        "name_skeleton": skeleton(tokens),
        "name_core_skeleton": skeleton(core_tokens),
        "name_acronym": "".join(t[0] for t in core_tokens if not t.isdigit()),
        "name_alias_parts": alias_parts,
        "name_legal_families": families,
        "name_tokens": frozenset(tokens),
        "name_core_tokens": core_tokens,
        "name_core_skeleton_tokens": tuple(skeleton_token(t) for t in core_tokens),
        "name_non_latin": non_latin,
    }


def _normalize_address_component(component: str) -> list[str]:
    code = REGION_NAME_TO_CODE.get(component)
    if code is not None:
        return [code]
    tokens = component.split()
    out: list[str] = []
    for token in tokens:
        match = _ORDINAL_RE.match(token)
        if match:
            out.append(_strip_leading_zeros(match.group(1)))
        elif token.isdigit():
            out.append(_strip_leading_zeros(token))
        else:
            out.append(ADDRESS_ABBREVIATIONS.get(token, token))
    return out


def prepare_address(raw: object) -> dict[str, object]:
    raw_text = "" if raw is None else str(raw)
    components_unicode: list[str] = []
    for component in raw_text.split(","):
        normalized = normalize_unicode(component)
        if normalized and normalized not in MISSING_TOKENS:
            components_unicode.append(normalized)
    unicode_text = " ".join(components_unicode)
    latin_components: list[list[str]] = []
    non_latin = False
    for component in components_unicode:
        latin, component_non_latin = _latin_view(component)
        non_latin = non_latin or component_non_latin
        if latin:
            latin_components.append(_normalize_address_component(latin))
    tokens = [t for comp in latin_components for t in comp]
    numbers = [t for t in tokens if t.isdigit()]
    # Mixed tokens such as "5n", "b3" or "22a" still carry numeric evidence.
    for t in tokens:
        if not t.isdigit():
            numbers.extend(_strip_leading_zeros(n) for n in _NUM_RE.findall(t))
    latin_text = " ".join(tokens)
    alpha = frozenset(t for t in tokens if not any(c.isdigit() for c in t))
    return {
        "address_unicode": unicode_text,
        "address_latin": latin_text,
        "address_sorted": " ".join(sorted(tokens)),
        "address_skeleton": skeleton(tuple(tokens)),
        "address_tokens": frozenset(tokens),
        "address_alpha_tokens": alpha,
        "address_numbers": frozenset(numbers),
        "address_longest_number": max(numbers, key=lambda n: (len(n), n)) if numbers else "",
        "address_first_component": " ".join(latin_components[0]) if latin_components else "",
        "address_missing": not tokens,
        "address_non_latin": non_latin,
    }


def prepare_record(name: object, address: object) -> PreparedRecord:
    return PreparedRecord(**prepare_name(name), **prepare_address(address))
