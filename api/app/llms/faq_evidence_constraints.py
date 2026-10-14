import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from app.data.connection import Database
from app.data.questions import get_question_names_query
from app.llms.faq_query_recovery import (
    _fold_token as _fold_faq_token,
)
from app.llms.faq_query_recovery import (
    faq_query_entity_forms,
    faq_query_institution_forms,
    match_faq_entity,
    match_faq_entity_with_status,
)

logger = logging.getLogger(__name__)

_CATALOG_LIMIT = 513
_CATALOG_CAP = 512
_MAX_QUERY_LENGTH = 320
_MAX_TOKENS = 36
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_EXCLUSION_RE = re.compile(
    r"\b(?:не|ne)\s+(?:прашувам|prasuvam|prashuvam)\s+(?:за|za)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[,.!?;]")
_INSTITUTION_TERMS = frozenset(
    {
        "fakultet",
        "fakultetot",
        "faculty",
        "univerzitet",
        "univerzitetot",
        "university",
    },
)
_INSTITUTION_SUBJECT_BLOCKERS = frozenset(
    {
        "od",
        "vo",
        "v",
        "na",
        "at",
        "in",
        "on",
        "profesor",
        "profesorot",
        "profesorka",
        "profesorkata",
        "professor",
        "student",
        "studentot",
        "studentka",
        "studentkata",
        "dekan",
        "dekanot",
        "nastavnik",
        "nastavnikot",
        "teacher",
        "lecturer",
    },
)
_INSTITUTION_PREPOSITIONS = frozenset(
    {"od", "vo", "v", "na", "at", "in", "on", "of", "za"},
)
_DIRECT_FACT_SOURCES_BY_EXCLUDED_TITLE = {
    "Студентска служба": ("Контакт и локација",),
    "Деканат": ("Контакт и локација",),
}
_OFFICE_TITLES = (
    "Студентска служба",
    "Деканат",
    "Компјутерски центар",
    "Архива",
    "Контакт и локација",
)
_FINKI_NAMES = (
    ("finki",),
    ("fakultet", "za", "informatichki", "nauki", "i", "kompjutersko", "inzhenerstvo"),
    ("fakultetot", "za", "informatichki", "nauki", "i", "kompjutersko", "inzhenerstvo"),
    ("faculty", "of", "computer", "science", "and", "engineering"),
    ("faculty", "of", "information", "science", "and", "computer", "engineering"),
    ("faculty", "of", "information", "sciences", "and", "computer", "engineering"),
)
_INSTITUTION_CONNECTORS = frozenset({"i", "ili", "and", "or", "vs", "versus"})
_INSTITUTION_MODIFIER_TOKENS = frozenset(
    {
        "denes",
        "deneska",
        "today",
        "tomorrow",
        "utre",
        "vo",
        "v",
        "ponedelnik",
        "vtornik",
        "sreda",
        "cetvrtok",
        "petok",
        "sabota",
        "nedela",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    },
)

FaqEvidenceConstraintReason = Literal[
    "none",
    "institution_scope",
    "entity_exclusion",
    "catalog_error",
]
FaqEvidenceConstraintKind = Literal["institution", "exclusion"]


@dataclass(frozen=True, slots=True)
class FaqConstraints:
    kind: FaqEvidenceConstraintKind
    reason: Literal["institution_scope", "entity_exclusion"]
    office_titles: tuple[str, ...]
    institution_is_finki: bool | None = None
    excluded_title: str | None = None
    target_title: str | None = None
    direct_fact_source_titles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FaqEvidenceConstraintPlan:
    outcome: Literal["none", "matched", "error"]
    constraints: FaqConstraints | None = None
    catalog_titles: tuple[str, ...] | None = None


class _CandidateSource(Protocol):
    @property
    def title(self) -> str: ...


class _FaqCandidate(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def retrieval_source(self) -> _CandidateSource: ...


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.casefold()))


def _fold_token(token: str) -> str:
    # Share the conservative transliteration used by the FAQ entity resolver.
    return _fold_faq_token(token)


def _folded(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_fold_token(token) for token in tokens)


def _is_finki(institution: tuple[str, ...]) -> bool:
    return _folded(institution) in _FINKI_NAMES


def _has_finki_prefix_with_extra(institution_key: tuple[str, ...]) -> bool:
    return any(
        len(institution_key) > len(name) and institution_key[: len(name)] == name
        for name in _FINKI_NAMES
    )


def _office_titles(titles: tuple[str, ...]) -> tuple[tuple[str, ...], bool]:
    found: list[str] = []
    ambiguous = False
    for office_title in _OFFICE_TITLES:
        status, resolved = match_faq_entity_with_status(office_title, titles)
        ambiguous = ambiguous or status == "ambiguous"
        if resolved is not None and resolved not in found:
            found.append(resolved)
    return tuple(found), ambiguous


def _is_reviewed_definition(head: str) -> bool:
    tokens = _folded(_tokens(head))
    return (
        (
            len(tokens) == 4
            and tokens[:3]
            in {
                ("shto", "znachi", "zborot"),
                ("sto", "znaci", "zborot"),
            }
        )
        or (len(tokens) == 3 and tokens[:2] in {("shto", "znachi"), ("sto", "znaci")})
        or (
            len(tokens) == 4
            and tokens[0:2] == ("what", "does")
            and tokens[-1] == "mean"
        )
    )


def _unquoted_exclusion_matches(
    query: str,
) -> list[re.Match[str]] | None:
    matches = list(_EXCLUSION_RE.finditer(query))
    if len(matches) != 1:
        return None

    quote_pairs = {'"': '"', "'": "'", "«": "»", "„": "“", "“": "”"}
    closing_quote: str | None = None
    quoted_positions = [False] * len(query)
    for index, character in enumerate(query):
        if closing_quote is None:
            if character in quote_pairs:
                closing_quote = quote_pairs[character]
                quoted_positions[index] = True
            continue
        quoted_positions[index] = True
        if character == closing_quote:
            closing_quote = None
        elif character in quote_pairs:
            # Nested or mismatched quotation is intentionally ambiguous here.
            return None
    if closing_quote is not None:
        return None

    match = matches[0]
    if any(quoted_positions[match.start() : match.end()]):
        return None
    return matches


def _exclusion_constraint(
    query: str,
    titles: tuple[str, ...],
    office_titles: tuple[str, ...],
) -> FaqConstraints | None:
    matches = _unquoted_exclusion_matches(query)
    if matches is None:
        return None
    match = matches[0]
    left = query[: match.start()]
    right = query[match.end() :]
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(left))
    clause_start = boundaries[-1].end() if boundaries else 0
    if left[clause_start:].strip():
        return None

    next_boundary = _CLAUSE_BOUNDARY_RE.search(right)
    excluded_phrase = (
        right[: next_boundary.start()] if next_boundary else right
    ).strip()
    request_clauses = [left[:clause_start]]
    if next_boundary:
        request_clauses.append(right[next_boundary.end() :])
    request_clauses = [clause.strip() for clause in request_clauses if clause.strip()]
    if not excluded_phrase or len(request_clauses) != 1:
        return None

    request_clause = request_clauses[0].strip().strip("?!.;,: ")
    recognized_request = _is_reviewed_definition(request_clause)
    if not recognized_request:
        recognized_request = bool(
            faq_query_entity_forms(
                f"{request_clause}?",
                max_query_length=_MAX_QUERY_LENGTH,
                max_query_tokens=_MAX_TOKENS,
            )
        )
    if not recognized_request:
        return None

    excluded_title = match_faq_entity(excluded_phrase, titles)
    if excluded_title is None or excluded_title not in office_titles:
        return None
    direct_fact_sources: list[str] = []
    for source_title in _DIRECT_FACT_SOURCES_BY_EXCLUDED_TITLE.get(excluded_title, ()):
        source_status, resolved_source = match_faq_entity_with_status(
            source_title, titles
        )
        if source_status == "ambiguous":
            return None
        if resolved_source is not None and resolved_source not in direct_fact_sources:
            direct_fact_sources.append(resolved_source)
    return FaqConstraints(
        kind="exclusion",
        reason="entity_exclusion",
        office_titles=office_titles,
        excluded_title=excluded_title,
        direct_fact_source_titles=tuple(direct_fact_sources),
    )


def _institution_contact_constraint(
    query: str,
    catalog: tuple[str, ...],
    office_titles: tuple[str, ...],
) -> FaqConstraints | None:
    forms = faq_query_institution_forms(
        query,
        max_query_length=_MAX_QUERY_LENGTH,
        max_query_tokens=_MAX_TOKENS,
    )
    for _intent, institution, contact_kind in forms:
        institution_key = _folded(institution)
        if contact_kind != "email" or not institution:
            continue
        if len(institution_key) > 10 or any(
            not token.isalpha() for token in institution_key
        ):
            continue
        if _has_finki_prefix_with_extra(institution_key):
            continue
        if any(token in _INSTITUTION_MODIFIER_TOKENS for token in institution_key):
            continue
        if (
            any(token in _INSTITUTION_CONNECTORS for token in institution_key)
            and institution_key not in _FINKI_NAMES
        ):
            continue
        if any(token in _INSTITUTION_SUBJECT_BLOCKERS for token in institution_key):
            continue
        if not _is_reviewed_institution_subject(institution_key):
            continue
        target_title = match_faq_entity("Контакт и локација", catalog)
        if target_title is None or target_title not in office_titles:
            return None
        return FaqConstraints(
            kind="institution",
            reason="institution_scope",
            office_titles=office_titles,
            institution_is_finki=_is_finki(institution),
            target_title=target_title,
        )
    return None


def _is_reviewed_institution_subject(institution: tuple[str, ...]) -> bool:
    if institution in _FINKI_NAMES:
        return True
    if len(institution) < 2:
        return False
    markers = {
        index for index, token in enumerate(institution) if token in _INSTITUTION_TERMS
    }
    if len(markers) != 1:
        return False
    marker_index = next(iter(markers))
    if marker_index == len(institution) - 1:
        return not any(token in _INSTITUTION_PREPOSITIONS for token in institution)
    if marker_index != 0 or len(institution) < 2:
        return False
    # A leading institution type may take the reviewed "for/of" linker, then
    # a bounded proper-name phrase; other prepositions are rejected above.
    linker_index = 1 if institution[1] in {"of", "za"} else None
    return not any(
        token in _INSTITUTION_PREPOSITIONS and index != linker_index
        for index, token in enumerate(institution)
    ) and all(token not in _INSTITUTION_CONNECTORS for token in institution)


def parse_explicit_faq_constraints(
    query: str,
    titles: list[str] | tuple[str, ...] | None = None,
) -> FaqConstraints | None:
    """Parse only reviewed, complete office-attribute or request-exclusion forms."""
    if not query or len(query) > _MAX_QUERY_LENGTH:
        return None
    if len(_tokens(query)) > _MAX_TOKENS:
        return None
    catalog = tuple(titles) if titles is not None else _OFFICE_TITLES
    if len(catalog) > _CATALOG_CAP:
        return None
    office_titles, office_titles_ambiguous = _office_titles(catalog)
    if not office_titles:
        return None

    exclusion = _exclusion_constraint(query, catalog, office_titles)
    if exclusion is not None:
        return exclusion
    if office_titles_ambiguous:
        return None

    institution_contact = _institution_contact_constraint(query, catalog, office_titles)
    if institution_contact is not None:
        return institution_contact

    forms = faq_query_entity_forms(
        query,
        max_query_length=_MAX_QUERY_LENGTH,
        max_query_tokens=_MAX_TOKENS,
    )
    if not forms:
        return None
    for _intent, entity_tokens, _contact_kind in forms:
        folded = _folded(entity_tokens)
        for separator_index, separator in enumerate(folded):
            if separator not in {"na", "at"} or separator_index == 0:
                continue
            entity = entity_tokens[:separator_index]
            institution = entity_tokens[separator_index + 1 :]
            if not institution or len(institution) > 10:
                continue
            if any(not token.isalpha() for token in entity + institution):
                continue
            institution_key = _folded(institution)
            if _has_finki_prefix_with_extra(institution_key):
                continue
            if any(token in _INSTITUTION_MODIFIER_TOKENS for token in institution_key):
                continue
            if (
                any(token in _INSTITUTION_CONNECTORS for token in institution_key)
                and institution_key not in _FINKI_NAMES
            ):
                continue
            target_title = match_faq_entity(entity, catalog)
            if target_title is None or target_title not in office_titles:
                continue
            return FaqConstraints(
                kind="institution",
                reason="institution_scope",
                office_titles=office_titles,
                institution_is_finki=_is_finki(institution),
                target_title=target_title,
            )
    return None


def apply_faq_constraints_to_candidates[T: _FaqCandidate](
    candidates: Sequence[T],
    constraints: FaqConstraints,
) -> list[T]:
    """Filter only incompatible mapped FAQ evidence; preserve order and all documents."""
    if constraints.kind == "institution" and constraints.institution_is_finki:
        return list(candidates)

    filtered: list[T] = []
    for candidate in candidates:
        title = candidate.retrieval_source.title
        if (
            candidate.source != "faq"
            or (
                constraints.kind == "institution"
                and title not in constraints.office_titles
            )
            or (
                constraints.kind == "exclusion"
                and title
                not in (
                    constraints.excluded_title,
                    *constraints.direct_fact_source_titles,
                )
            )
        ):
            filtered.append(candidate)
    return filtered


async def plan_faq_evidence_constraints(
    db: Database,
    query: str,
) -> FaqEvidenceConstraintPlan:
    """Resolve explicit constraints against a complete, bounded FAQ title catalog."""
    preliminary = parse_explicit_faq_constraints(query)
    if preliminary is None:
        return FaqEvidenceConstraintPlan(outcome="none")
    try:
        titles = await get_question_names_query(db, limit=_CATALOG_LIMIT)
        if len(titles) > _CATALOG_CAP:
            return FaqEvidenceConstraintPlan(outcome="error")
        constraints = parse_explicit_faq_constraints(query, titles)
        if constraints is None:
            return FaqEvidenceConstraintPlan(outcome="error")
        return FaqEvidenceConstraintPlan(
            outcome="matched",
            constraints=constraints,
            catalog_titles=tuple(titles),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # ruff: ignore[BLE001] -- explicit constraint fails closed
        logger.warning(
            "FAQ evidence-constraint catalog unavailable error_type=%s",
            type(exc).__name__,
        )
        return FaqEvidenceConstraintPlan(outcome="error")
