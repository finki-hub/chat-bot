import logging
import unicodedata
from dataclasses import dataclass
from typing import Final

from asyncpg import Record

from app.data.connection import Database
from app.data.links import fetch_links_for_context
from app.llms.reranker import post_rerank
from app.utils.settings import Settings
from app.utils.timing import timed

logger = logging.getLogger(__name__)
settings = Settings()

# Bound the user-editable catalog's footprint in every prompt.
_LINKS_RENDER_MAX: Final[int] = 50
_LINK_NAME_MAX: Final[int] = 80
_LINK_URL_MAX: Final[int] = 2048
_LINK_DESC_MAX: Final[int] = 200
_RELEVANT_LINKS_MAX: Final[int] = 5
_FINKI_HUB_BOOST: Final[float] = 0.08
_FINKI_HUB_DOMAIN: Final[str] = "finki-hub.com"
_FINKI_HUB_QUERY_TERMS: Final[tuple[str, ...]] = (
    "finki hub",
    "финки хаб",
    "предмет",
    "predmet",
    "course",
    "курс",
    "сним",
    "recording",
    "распоред",
    "schedule",
    "timetable",
    "диплом",
    "diploma",
)


@dataclass(frozen=True, slots=True)
class _LinkCandidate:
    name: str
    url: str
    description: str


@dataclass(frozen=True, slots=True)
class _RankedLink:
    candidate: _LinkCandidate
    score: float


def _sanitize_inline(text: str, max_len: int) -> str:
    """Flatten a field to a bounded inline span so stored values cannot fabricate prompt structure."""
    spaced = "".join(" " if ch.isspace() else ch for ch in text)
    cleaned = "".join(
        ch for ch in spaced if not unicodedata.category(ch).startswith("C")
    )
    collapsed = " ".join(cleaned.split())
    if len(collapsed) > max_len:
        collapsed = collapsed[:max_len].rstrip() + "…"
    return collapsed


def _candidate_from_row(row: Record) -> _LinkCandidate | None:
    name = _sanitize_inline(row["name"] or "", _LINK_NAME_MAX)
    url = _sanitize_inline(row["url"] or "", _LINK_URL_MAX)
    if not name or not url:
        return None
    return _LinkCandidate(
        name=name,
        url=url,
        description=_sanitize_inline(row["description"] or "", _LINK_DESC_MAX),
    )


def _rerank_text(candidate: _LinkCandidate) -> str:
    return "\n".join(
        part
        for part in [
            f"Име: {candidate.name}",
            f"URL: {candidate.url}",
            f"Опис: {candidate.description}" if candidate.description else None,
        ]
        if part is not None
    )


def _line(candidate: _LinkCandidate) -> str:
    rendered = f"- {candidate.name}: {candidate.url}"
    if candidate.description:
        rendered += f" ({candidate.description})"
    return rendered


def _query_can_use_finki_hub(query: str) -> bool:
    normalized = query.casefold()
    return any(term in normalized for term in _FINKI_HUB_QUERY_TERMS)


def _boosted_score(ranked: _RankedLink, query: str) -> float:
    if _FINKI_HUB_DOMAIN in ranked.candidate.url and _query_can_use_finki_hub(query):
        return ranked.score + _FINKI_HUB_BOOST
    return ranked.score


async def _rank_relevant_links(
    query: str,
    candidates: list[_LinkCandidate],
) -> list[_LinkCandidate]:
    response = await post_rerank(
        {"query": query, "documents": [_rerank_text(c) for c in candidates]},
    )
    ranked_items = response.json()["reranked_documents"]
    ranked: list[_RankedLink] = []
    for item in ranked_items:
        idx = item["index"]
        if not 0 <= idx < len(candidates):
            logger.warning(
                "Link reranker returned out-of-range index %d (have %d candidates)",
                idx,
                len(candidates),
            )
            continue
        score = item["score"]
        if score >= settings.RERANKER_MIN_SCORE:
            ranked.append(_RankedLink(candidate=candidates[idx], score=score))

    ranked.sort(key=lambda link: _boosted_score(link, query), reverse=True)
    return [link.candidate for link in ranked[:_RELEVANT_LINKS_MAX]]


async def get_links_context(db: Database, *, query: str | None = None) -> str:
    """Render relevant links first, followed by the capped link catalog."""
    try:
        with timed("links"):
            rows = await fetch_links_for_context(db)
    except Exception:
        logger.exception("Failed to load the link catalog for context")
        return ""

    candidates = [
        candidate for row in rows if (candidate := _candidate_from_row(row)) is not None
    ]
    if not candidates:
        return ""

    relevant: list[_LinkCandidate] = []
    if query:
        try:
            with timed("links.rerank"):
                relevant = await _rank_relevant_links(query, candidates)
        except Exception:
            logger.exception("Failed to rerank link catalog; rendering unranked links")

    if not relevant:
        return "Корисни линкови:\n" + "\n".join(
            _line(c) for c in candidates[:_LINKS_RENDER_MAX]
        )

    relevant_keys = {c.url for c in relevant}
    other_links = [c for c in candidates if c.url not in relevant_keys][
        : _LINKS_RENDER_MAX - len(relevant)
    ]
    sections = ["Најрелевантни линкови:\n" + "\n".join(_line(c) for c in relevant)]
    if other_links:
        sections.append(
            "Други корисни линкови:\n" + "\n".join(_line(c) for c in other_links),
        )
    return "\n\n".join(sections)
