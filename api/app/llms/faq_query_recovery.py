import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

FaqQueryIntent = Literal["location", "hours", "contact"]
FaqQueryMatchKind = Literal["title", "alias"]
FaqContactKind = Literal["email", "phone"]
_QueryPosition = Literal["prefix", "suffix"]

_MAX_QUERY_LENGTH = 160
_MAX_QUERY_TOKENS = 12
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_UNSUPPORTED_ENTITY_WORDS = frozenset(
    {
        "and",
        "ili",
        "i",
        "ne",
        "no",
        "not",
        "or",
        "vo",
        "v",
        "nedela",
        "ponedelnik",
        "vtornik",
        "sreda",
        "cetvrtok",
        "petok",
        "sabota",
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    },
)

_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "ѓ": "gj",
        "е": "e",
        "ж": "zh",
        "з": "z",
        "ѕ": "dz",
        "и": "i",
        "ј": "j",
        "к": "k",
        "л": "l",
        "љ": "lj",
        "м": "m",
        "н": "n",
        "њ": "nj",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "ќ": "kj",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "c",
        "ч": "ch",
        "џ": "dzh",
        "ш": "sh",
    },
)

# Reviewed full forms only; there is no generic suffix-stripping or stemming.
_REVIEWED_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("служба",), ("студентска", "служба")),
    (("службата",), ("студентска", "служба")),
    (("студентската", "служба"), ("студентска", "служба")),
    (("студентската", "службата"), ("студентска", "служба")),
    (("деканатот",), ("деканат",)),
    (("компјутерскиот", "центар"), ("компјутерски", "центар")),
    (("архивата",), ("архива",)),
)


@dataclass(frozen=True, slots=True)
class _QueryForm:
    tokens: tuple[str, ...]
    intent: FaqQueryIntent
    position: _QueryPosition = "prefix"
    contact_kind: FaqContactKind | None = None


_QUERY_FORMS: tuple[_QueryForm, ...] = (
    _QueryForm(("каде", "е"), "location"),
    _QueryForm(("каде", "се", "наоѓа"), "location"),
    _QueryForm(("каде", "се", "наоѓа"), "location", position="suffix"),
    _QueryForm(("кое", "е", "работното", "време", "на"), "hours"),
    _QueryForm(("работно", "време", "на"), "hours"),
    _QueryForm(("до", "колку", "работи"), "hours"),
    _QueryForm(("во", "колку", "часот", "работи"), "hours"),
    _QueryForm(
        ("која", "е", "е", "поштата", "на"),
        "contact",
        contact_kind="email",
    ),
    _QueryForm(("која", "е", "email", "на"), "contact", contact_kind="email"),
    _QueryForm(("е", "пошта", "на"), "contact", contact_kind="email"),
    _QueryForm(("email", "на"), "contact", contact_kind="email"),
    _QueryForm(("email", "за"), "contact", contact_kind="email"),
    _QueryForm(("мејл", "на"), "contact", contact_kind="email"),
    _QueryForm(("мејл", "за"), "contact", contact_kind="email"),
    _QueryForm(
        ("кој", "е", "телефонскиот", "број", "на"),
        "contact",
        contact_kind="phone",
    ),
    _QueryForm(("телефон", "на"), "contact", contact_kind="phone"),
    _QueryForm(("телефонски", "број", "на"), "contact", contact_kind="phone"),
    _QueryForm(("телефонски", "број", "за"), "contact", contact_kind="phone"),
)


@dataclass(frozen=True, slots=True)
class FaqQueryRecovery:
    title: str
    intent: FaqQueryIntent
    canonical_query: str
    match_kind: FaqQueryMatchKind
    contact_kind: FaqContactKind | None = None
    institution_qualified: bool = False


def _fold_token(token: str) -> str:
    transliterated = token.casefold().translate(_CYRILLIC_TO_LATIN)
    decomposed = unicodedata.normalize("NFKD", transliterated)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _comparison_token(token: str) -> str:
    key = _fold_token(token)
    # Only the reviewed Student Service spelling pair is equivalenced.
    if key in {"sluzhba", "sluzba"}:
        return "sluzba"
    if key in {"sluzhbata", "sluzbata"}:
        return "sluzbata"
    return key


def _phrase_key(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_comparison_token(token) for token in tokens)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.casefold()))


def _grammar_tokens(query: str) -> tuple[str, ...] | None:
    if not query or len(query) > _MAX_QUERY_LENGTH:
        return None
    trimmed = query.strip()
    grammar_text = trimmed[:-1] if trimmed.endswith(("?", "!", ".")) else trimmed
    grammar_text = grammar_text.casefold().replace("е-поштата", "е поштата")
    grammar_text = grammar_text.replace("е-пошта", "е пошта")
    if not grammar_text or any(
        not char.isalpha() and not char.isspace() for char in grammar_text
    ):
        return None
    if any(char in grammar_text for char in "?!. ".strip()):
        return None
    tokens = _tokens(grammar_text)
    if not 3 <= len(tokens) <= _MAX_QUERY_TOKENS:
        return None
    # Numeric/technical identifiers are not generalized into natural-language aliases.
    if any(any(char.isdigit() for char in token) for token in tokens):
        return None
    return tokens


def _form_entity(
    tokens: tuple[str, ...],
    form: _QueryForm,
) -> tuple[str, ...] | None:
    folded = tuple(_fold_token(token) for token in tokens)
    form_tokens = tuple(_fold_token(token) for token in form.tokens)
    if form.position == "prefix":
        if folded[: len(form_tokens)] != form_tokens:
            return None
        entity = tokens[len(form_tokens) :]
    else:
        if (
            len(folded) <= len(form_tokens)
            or folded[-len(form_tokens) :] != form_tokens
        ):
            return None
        entity = tokens[: -len(form_tokens)]
    return entity or None


def is_faq_query_recovery_eligible(query: str) -> bool:
    tokens = _grammar_tokens(query)
    return tokens is not None and any(
        _form_entity(tokens, form) for form in _QUERY_FORMS
    )


def _title_match(
    entity_tokens: tuple[str, ...],
    titles_by_key: dict[tuple[str, ...], list[str]],
) -> tuple[str, FaqQueryMatchKind] | None:
    exact_matches = set(titles_by_key.get(_phrase_key(entity_tokens), ()))
    alias_targets: set[str] = set()
    entity_key = _phrase_key(entity_tokens)
    for alias_tokens, target_tokens in _REVIEWED_ALIASES:
        if entity_key != _phrase_key(alias_tokens):
            continue
        alias_targets.update(titles_by_key.get(_phrase_key(target_tokens), ()))

    candidates = exact_matches | alias_targets
    if len(candidates) != 1:
        return None
    title = next(iter(candidates))
    return title, "title" if title in exact_matches else "alias"


def _title_match_with_qualifier(
    entity_tokens: tuple[str, ...],
    titles_by_key: dict[tuple[str, ...], list[str]],
) -> tuple[str, FaqQueryMatchKind, bool] | None:
    folded = tuple(_fold_token(token) for token in entity_tokens)
    institution_qualified = len(folded) > 2 and folded[-2:] == ("na", "finki")
    base_tokens = entity_tokens[:-2] if institution_qualified else entity_tokens
    base_folded = folded[:-2] if institution_qualified else folded
    if any(token in _UNSUPPORTED_ENTITY_WORDS for token in base_folded):
        return None
    if "na" in base_folded:
        return None
    title_match = _title_match(base_tokens, titles_by_key)
    if title_match is None:
        return None
    title, match_kind = title_match
    return title, match_kind, institution_qualified


def resolve_faq_query(
    query: str,
    titles: list[str] | tuple[str, ...],
) -> FaqQueryRecovery | None:
    query_tokens = _grammar_tokens(query)
    if query_tokens is None:
        return None

    titles_by_key: dict[tuple[str, ...], list[str]] = {}
    for faq_title in titles:
        title_tokens = _tokens(faq_title)
        if not title_tokens or len(title_tokens) > _MAX_QUERY_TOKENS:
            continue
        titles_by_key.setdefault(_phrase_key(title_tokens), []).append(faq_title)

    matches: set[
        tuple[FaqQueryIntent, str, FaqQueryMatchKind, FaqContactKind | None, bool]
    ] = set()
    for form in _QUERY_FORMS:
        entity = _form_entity(query_tokens, form)
        if entity is None:
            continue
        title_match = _title_match_with_qualifier(entity, titles_by_key)
        if title_match is None:
            continue
        title, match_kind, institution_qualified = title_match
        matches.add(
            (form.intent, title, match_kind, form.contact_kind, institution_qualified)
        )

    if len(matches) != 1:
        return None
    intent, title, match_kind, contact_kind, institution_qualified = next(iter(matches))
    entity_suffix = " на ФИНКИ" if institution_qualified else ""

    if intent == "location":
        canonical_query = f"Каде се наоѓа {title}{entity_suffix}?"
    elif intent == "hours":
        canonical_query = f"Кое е работното време на {title}{entity_suffix}?"
    elif contact_kind == "phone":
        canonical_query = f"Кој е телефонскиот број на {title}{entity_suffix}?"
    elif contact_kind == "email":
        canonical_query = f"Која е е-поштата на {title}{entity_suffix}?"
    else:
        return None

    return FaqQueryRecovery(
        title=title,
        intent=intent,
        canonical_query=canonical_query,
        match_kind=match_kind,
        contact_kind=contact_kind,
        institution_qualified=institution_qualified,
    )
