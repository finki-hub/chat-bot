import re
from typing import Final
from unicodedata import normalize

_WORD_RE: Final = re.compile(r"[^\W_]+", re.UNICODE)
_DIGRAPHS: Final[tuple[tuple[str, str], ...]] = (
    ("dzh", "џ"),
    ("gj", "ѓ"),
    ("kj", "ќ"),
    ("zh", "ж"),
    ("dz", "ѕ"),
    ("lj", "љ"),
    ("nj", "њ"),
    ("ch", "ч"),
    ("sh", "ш"),
)
_LETTERS: Final[dict[str, str]] = {
    "a": "а",
    "b": "б",
    "c": "ц",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "ј",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "z": "з",
    "š": "ш",
    "č": "ч",
    "ž": "ж",
    "ǵ": "ѓ",
    "ḱ": "ќ",
}


def _transliterate_token(match: re.Match[str]) -> str:
    token = match.group()
    if (
        any(character.isdigit() for character in token)
        or token.isupper()
        or any(character.isupper() for character in token[1:])
    ):
        return token

    source = token.lower()
    transliterated: list[str] = []
    index = 0
    while index < len(source):
        for latin, cyrillic in _DIGRAPHS:
            if source.startswith(latin, index):
                transliterated.append(cyrillic)
                index += len(latin)
                break
        else:
            character = source[index]
            mapped_character = _LETTERS.get(character)
            if mapped_character is None:
                return token
            transliterated.append(mapped_character)
            index += 1

    result = "".join(transliterated)
    return result.capitalize() if token.istitle() else result


def query_search_variants(query: str) -> tuple[str, ...]:
    """Return the original query and a distinct local Cyrillic variant."""
    normalized_query = normalize("NFC", query)
    transliterated = _WORD_RE.sub(_transliterate_token, normalized_query)
    return (query, transliterated) if transliterated != normalized_query else (query,)


# Romanized Macedonian function words that are not English words, so English queries
# (which transliterate just as readily) are not mistaken for Macedonian.
_ROMANIZED_MACEDONIAN_MARKERS: Final = frozenset(
    {
        "e",
        "na",
        "da",
        "se",
        "za",
        "od",
        "po",
        "vo",
        "li",
        "ili",
        "ne",
        "ke",
        "kje",
        "ima",
        "nema",
        "kade",
        "koga",
        "kako",
        "zosto",
        "zoshto",
        "zasto",
        "sto",
        "shto",
        "sho",
        "koj",
        "koja",
        "koe",
        "koi",
        "kolku",
        "dali",
        "mozam",
        "mozhe",
        "moze",
        "treba",
        "sakam",
        "kaj",
        "ako",
        "samo",
        "sme",
        "ste",
        "gi",
    }
)
_MACEDONIAN_LATIN_LETTERS: Final = frozenset("šžčǵḱ")
_CYRILLIC_RE: Final = re.compile(r"[\u0400-\u04ff]")
_WORD_EDGE_PUNCTUATION: Final = "?!.,;:\"'()"


def romanized_macedonian_variant(query: str) -> str | None:
    """Return the Cyrillic transliteration of a romanized Macedonian query, else None."""
    normalized_query = normalize("NFC", query)
    if _CYRILLIC_RE.search(normalized_query):
        return None
    has_marker = not _MACEDONIAN_LATIN_LETTERS.isdisjoint(
        normalized_query.casefold()
    ) or any(
        token.strip(_WORD_EDGE_PUNCTUATION).casefold() in _ROMANIZED_MACEDONIAN_MARKERS
        for token in normalized_query.split()
    )
    if not has_marker:
        return None
    transliterated = _WORD_RE.sub(_transliterate_token, normalized_query)
    return transliterated if transliterated != normalized_query else None
