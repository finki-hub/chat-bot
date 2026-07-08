import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from app.api.recommendation_presenters import (
    committee_alternative,
    confidence,
    mentor_score,
    person_score,
    proposal_text,
    recommendation_evidence,
)
from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.data import StaffDirectoryUnavailableError, get_active_staff_names
from app.data.connection import Database
from app.data.diplomas import get_defended_committees
from app.data.professor_documents import get_all_paper_authors
from app.llms.diploma_retrieval import retrieve_similar_diplomas
from app.llms.embeddings import generate_embeddings
from app.llms.professor_retrieval import retrieve_professor_papers
from app.llms.text_utils import _prepare_text_for_embedding
from app.recommenders.config import ScoringWeights
from app.recommenders.recommend import (
    CoauthorIndex,
    MentorPriorIndex,
    Mode,
    SelectionConstraints,
    _accumulate_coauthor_edges,
    build_coauthor_prior,
    build_expertise_index,
    build_mentor_prior,
    score_people,
    select_committee,
)
from app.schemas.recommendations import (
    RecommendationRequestSchema,
    RecommendationResponseSchema,
)

_INITIAL_K = 30
_TOP_K = 10
_PAPER_K = 50

_PRIOR_CACHE_TTL = 3600.0
_mentor_prior_cache: dict[str, tuple[float, MentorPriorIndex]] = {}


class RecommendationServiceError(Exception):
    pass


class ActiveStaffDirectoryUnavailableError(RecommendationServiceError):
    pass


@dataclass(frozen=True, slots=True)
class InactiveStaffRequestedError(RecommendationServiceError):
    field: Literal["mentor", "include_professors"]
    names: tuple[str, ...]


class NoSimilarDefensesError(RecommendationServiceError):
    pass


async def _get_mentor_prior(db: Database) -> MentorPriorIndex:
    cached = _mentor_prior_cache.get("prior")
    now = time.monotonic()
    if cached is not None and now - cached[0] < _PRIOR_CACHE_TTL:
        return cached[1]
    committees = await get_defended_committees(db)
    prior = build_mentor_prior(
        (row["mentor"], row["member1"], row["member2"]) for row in committees
    )
    _mentor_prior_cache["prior"] = (now, prior)
    return prior


async def _get_coauthor_prior(db: Database) -> MentorPriorIndex:
    return build_coauthor_prior(await get_all_paper_authors(db))


def _inactive_requested_names(
    active_staff: frozenset[str],
    names: list[str],
) -> tuple[str, ...]:
    return tuple(name for name in names if name not in active_staff)


async def _active_staff_names() -> frozenset[str]:
    try:
        return await get_active_staff_names()
    except StaffDirectoryUnavailableError as exc:
        raise ActiveStaffDirectoryUnavailableError from exc


def _check_active_staff(
    payload: RecommendationRequestSchema,
    active_staff: frozenset[str],
) -> None:
    if payload.mentor is not None and payload.mentor not in active_staff:
        raise InactiveStaffRequestedError("mentor", (payload.mentor,))

    inactive_includes = _inactive_requested_names(
        active_staff,
        payload.include_professors,
    )
    if inactive_includes:
        raise InactiveStaffRequestedError("include_professors", inactive_includes)


async def _cancel_background_tasks(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def generate_committee_recommendation(
    payload: RecommendationRequestSchema,
    db: Database,
) -> RecommendationResponseSchema:
    mode = Mode.MEMBERS_ONLY if payload.mentor else Mode.FULL
    active_staff = await _active_staff_names()
    _check_active_staff(payload, active_staff)

    text = proposal_text(payload)
    query_embedding = await generate_embeddings(
        _prepare_text_for_embedding(text, DEFAULT_EMBEDDINGS_MODEL, is_document=False),
        DEFAULT_EMBEDDINGS_MODEL,
    )
    weights = ScoringWeights()
    use_papers = (
        weights.expertise_weight > 0
        or weights.coauthor_weight > 0
        or weights.coauthor_member_boost > 0
    )

    retrieved_task = asyncio.create_task(
        retrieve_similar_diplomas(
            db,
            text,
            DEFAULT_EMBEDDINGS_MODEL,
            _INITIAL_K,
            _TOP_K,
            query_embedding=query_embedding,
        ),
    )
    paper_task = (
        asyncio.create_task(
            retrieve_professor_papers(
                db,
                text,
                DEFAULT_EMBEDDINGS_MODEL,
                _PAPER_K,
                query_embedding=query_embedding,
            ),
        )
        if use_papers
        else None
    )
    mentor_prior_task = asyncio.create_task(_get_mentor_prior(db))
    coauthor_prior_task = (
        asyncio.create_task(_get_coauthor_prior(db))
        if weights.coauthor_prior_weight > 0
        else None
    )
    background_tasks: list[asyncio.Task] = [mentor_prior_task]
    if paper_task is not None:
        background_tasks.append(paper_task)
    if coauthor_prior_task is not None:
        background_tasks.append(coauthor_prior_task)

    try:
        retrieved = await retrieved_task
    except asyncio.CancelledError:
        await _cancel_background_tasks(background_tasks)
        raise
    except Exception:
        await _cancel_background_tasks(background_tasks)
        raise

    if mode is Mode.FULL and not retrieved:
        await _cancel_background_tasks(background_tasks)
        raise NoSimilarDefensesError

    try:
        papers = await paper_task if paper_task is not None else []
        mentor_prior = await mentor_prior_task
        coauthor_prior = await coauthor_prior_task if coauthor_prior_task else None
    except asyncio.CancelledError:
        await _cancel_background_tasks(background_tasks)
        raise
    except Exception:
        await _cancel_background_tasks(background_tasks)
        raise

    now = datetime.now(UTC).date()
    expertise = build_expertise_index(papers, weights, now_year=now.year)
    coauthors = (
        _accumulate_coauthor_edges(papers, weights)
        if weights.coauthor_weight > 0 or weights.coauthor_member_boost > 0
        else CoauthorIndex({}, {})
    )
    ranked = score_people(
        retrieved,
        expertise,
        coauthors,
        weights,
        mode,
        given_mentor=payload.mentor,
        now=now,
    )
    rec = select_committee(
        ranked,
        mode,
        given_mentor=payload.mentor,
        mentor_topk=payload.mentor_topk,
        mentor_prior=mentor_prior,
        coauthor_prior=coauthor_prior,
        constraints=SelectionConstraints(
            exclude=frozenset(payload.exclude_professors),
            include=frozenset(payload.include_professors),
            allowed=active_staff,
            alternative_count=payload.alternatives,
        ),
    )
    confidence_score, confidence_reasons = confidence(retrieved, rec, _TOP_K)

    return RecommendationResponseSchema(
        mode=mode.value,
        mentor=mentor_score(rec, ranked),
        mentor_is_given=rec.mentor_is_given,
        members=[person_score(name, ranked, rec) for name in rec.members],
        supporting_diploma_ids=rec.supporting_diploma_ids,
        confidence=confidence_score,
        confidence_reasons=confidence_reasons,
        evidence=recommendation_evidence(retrieved, rec, ranked),
        alternatives=[committee_alternative(alt) for alt in rec.alternatives],
    )
