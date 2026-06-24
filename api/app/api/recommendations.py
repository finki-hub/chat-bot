import logging

from fastapi import APIRouter, Depends, status

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.data.connection import Database
from app.data.db import get_db
from app.data.diplomas import get_defended_committees
from app.data.professor_documents import get_all_paper_authors
from app.llms.diploma_retrieval import retrieve_similar_diplomas
from app.llms.professor_retrieval import retrieve_professor_papers
from app.recommenders.config import ScoringWeights
from app.recommenders.recommend import (
    CoauthorIndex,
    Mode,
    RankedPeople,
    Recommendation,
    _accumulate_coauthor_edges,
    build_coauthor_prior,
    build_expertise_index,
    build_mentor_prior,
    score_people,
    select_committee,
)
from app.schemas.diplomas import (
    PersonScoreSchema,
    RecommendationRequestSchema,
    RecommendationResponseSchema,
)

logger = logging.getLogger(__name__)

# Retrieval breadth: pull a wide vector neighborhood, then rerank down to the scoring set.
_INITIAL_K = 30
_TOP_K = 10
# Paper KNN breadth for the expertise signal (only used when a paper weight is on).
_PAPER_K = 50

db_dep = Depends(get_db)

router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
    dependencies=[db_dep],
)


def _person_score(
    name: str,
    ranked: RankedPeople,
    rec: Recommendation,
) -> PersonScoreSchema:
    """`score` is the final candidate score (blended + mentor-prior); for a prior-surfaced
    member (absent from `blended`) this is the only place the real score lives. `prior_score`
    is the prior's contribution = final - blended.
    """
    blended = ranked.blended.get(name)
    final = rec.member_scores.get(name, blended)
    prior = (final - (blended or 0.0)) if final is not None else 0.0
    return PersonScoreSchema(
        name=name,
        score=final,
        defense_score=ranked.defense.get(name, 0.0),
        expertise_score=ranked.expertise.get(name, 0.0),
        buddy_score=0.0,
        prior_score=prior,
        supporting_diploma_ids=ranked.supporting.get(name, []),
    )


def _mentor_score(
    rec: Recommendation,
    ranked: RankedPeople,
) -> PersonScoreSchema:
    """The mentor slot: chosen-with-scores in FULL; an echo of the given mentor in MEMBERS-ONLY."""
    name = rec.mentor or ""
    if rec.mentor_is_given:
        # MEMBERS-ONLY: the mentor is given, never scored — echo it with empty scores.
        return PersonScoreSchema(
            name=name,
            score=None,
            defense_score=0.0,
            expertise_score=0.0,
            buddy_score=0.0,
            supporting_diploma_ids=[],
        )
    return _person_score(name, ranked, rec)


@router.post(
    "/",
    summary="Recommend a thesis committee",
    description=(
        "Given a proposed thesis title (and optionally a known mentor), recommend a "
        "committee grounded in the most similar historical defenses. The mode is inferred "
        "from the payload: a mentor present -> MEMBERS-ONLY (recommend only the two "
        "members; the mentor is fixed and excluded); omitted -> FULL (recommend a mentor "
        "and two members)."
    ),
    status_code=status.HTTP_200_OK,
    operation_id="recommendCommittee",
)
async def recommend_committee(
    payload: RecommendationRequestSchema,
    db: Database = db_dep,
) -> RecommendationResponseSchema:
    mode = Mode.MEMBERS_ONLY if payload.mentor else Mode.FULL
    text = payload.title.strip()

    retrieved = await retrieve_similar_diplomas(
        db,
        text,
        DEFAULT_EMBEDDINGS_MODEL,
        _INITIAL_K,
        _TOP_K,
    )

    weights = ScoringWeights()

    # Paper-derived signals from the professor_document corpus. The expertise signal gives
    # members a gentle topical lift (and an explainability trail of supporting papers); the
    # buddy/co-author signal is built but OFF by default (coauthor_weight=0) — GATE B found
    # co-authorship does not predict committee co-membership. Skip the paper KNN entirely
    # when no paper signal is enabled.
    papers = (
        await retrieve_professor_papers(db, text, DEFAULT_EMBEDDINGS_MODEL, _PAPER_K)
        if weights.expertise_weight > 0 or weights.coauthor_weight > 0
        else []
    )
    expertise = build_expertise_index(papers, weights)
    coauthors = (
        _accumulate_coauthor_edges(papers, weights)
        if weights.coauthor_weight > 0
        else CoauthorIndex({}, {})
    )

    ranked = score_people(
        retrieved,
        expertise,
        coauthors,
        weights,
        mode,
        given_mentor=payload.mentor,
    )

    # Mentor-conditioned habitual co-membership prior, built from the WHOLE defense graph
    # (title-independent). It surfaces the resolved mentor's frequent collaborators even when
    # the topical retrieval missed them — the dominant member signal until the paper/buddy
    # corpus lands. No leave-one-out needed: the proposed thesis is not in the corpus.
    committees = await get_defended_committees(db)
    mentor_prior = build_mentor_prior(
        (row["mentor"], row["member1"], row["member2"]) for row in committees
    )

    # Global co-author prior (the mentor's frequent co-authors). Built + ablatable but OFF by
    # default (coauthor_prior_weight=0): co-authorship correlates with committee co-membership
    # but adds noise on top of the defense prior, which already records who actually serves.
    coauthor_prior = (
        build_coauthor_prior(await get_all_paper_authors(db))
        if weights.coauthor_prior_weight > 0
        else None
    )

    rec = select_committee(
        ranked,
        mode,
        given_mentor=payload.mentor,
        mentor_topk=payload.mentor_topk,
        mentor_prior=mentor_prior,
        coauthor_prior=coauthor_prior,
    )

    return RecommendationResponseSchema(
        mode=mode.value,
        mentor=_mentor_score(rec, ranked),
        mentor_is_given=rec.mentor_is_given,
        members=[_person_score(name, ranked, rec) for name in rec.members],
        supporting_diploma_ids=rec.supporting_diploma_ids,
    )
