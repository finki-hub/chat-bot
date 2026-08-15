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
    return (query, transliterated) if transliterated != query else (query,)
