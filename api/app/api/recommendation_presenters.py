from app.recommenders.recommend import (
    CommitteeAlternative,
    Mode,
    RankedPeople,
    Recommendation,
    RetrievedDiploma,
)
from app.schemas.recommendations import (
    CommitteeAlternativeSchema,
    PersonScoreSchema,
    RecommendationEvidenceSchema,
    RecommendationRequestSchema,
    SupportingDiplomaSchema,
)

_EVIDENCE_K = 5


def person_score(
    name: str,
    ranked: RankedPeople,
    rec: Recommendation,
) -> PersonScoreSchema:
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
        supporting_paper_titles=ranked.expertise_supporting.get(name, [])[:_EVIDENCE_K],
    )


def mentor_score(rec: Recommendation, ranked: RankedPeople) -> PersonScoreSchema:
    name = rec.mentor or ""
    if rec.mentor_is_given:
        return PersonScoreSchema(
            name=name,
            score=None,
            defense_score=0.0,
            expertise_score=0.0,
            buddy_score=0.0,
            supporting_diploma_ids=[],
            supporting_paper_titles=ranked.expertise_supporting.get(name, [])[
                :_EVIDENCE_K
            ],
        )
    return person_score(name, ranked, rec)


def proposal_text(payload: RecommendationRequestSchema) -> str:
    parts = [payload.title]
    if payload.abstract:
        parts.append(payload.abstract)
    if payload.keywords:
        parts.append("Клучни зборови: " + ", ".join(payload.keywords))
    if payload.study_program:
        parts.append("Студиска програма: " + payload.study_program)
    if payload.research_area:
        parts.append("Област: " + payload.research_area)
    return "\n".join(parts)


def recommendation_evidence(
    retrieved: list[RetrievedDiploma],
    rec: Recommendation,
    ranked: RankedPeople,
) -> RecommendationEvidenceSchema:
    return RecommendationEvidenceSchema(
        similar_diplomas=[supporting_diploma(d) for d in retrieved[:_EVIDENCE_K]],
        supporting_paper_titles=supporting_papers(rec, ranked),
    )


def supporting_diploma(diploma: RetrievedDiploma) -> SupportingDiplomaSchema:
    return SupportingDiplomaSchema(
        id=diploma.id,
        external_id=diploma.external_id,
        title=diploma.title,
        mentor=diploma.mentor,
        members=[m for m in (diploma.member1, diploma.member2) if m],
        similarity=diploma.similarity,
        rerank_score=diploma.rerank_score,
    )


def committee_alternative(
    alternative: CommitteeAlternative,
) -> CommitteeAlternativeSchema:
    return CommitteeAlternativeSchema(
        mentor=alternative.mentor,
        members=list(alternative.members),
        score=alternative.score,
        supporting_diploma_ids=alternative.supporting_diploma_ids,
    )


def supporting_papers(rec: Recommendation, ranked: RankedPeople) -> list[str]:
    seen: dict[str, None] = {}
    for name in (rec.mentor, *rec.members):
        if name is None:
            continue
        for title in ranked.expertise_supporting.get(name, []):
            seen.setdefault(title, None)
    return list(seen)[:_EVIDENCE_K]


def confidence(
    retrieved: list[RetrievedDiploma],
    rec: Recommendation,
    top_k: int,
) -> tuple[float, list[str]]:
    top_similarity = max((item.similarity for item in retrieved), default=0.0)
    retrieval_coverage = min(len(retrieved) / top_k, 1.0)
    committee_slots = 3 if rec.mode is Mode.FULL else 2
    filled_slots = len(rec.members) + (
        1 if rec.mentor is not None and rec.mode is Mode.FULL else 0
    )
    committee_coverage = min(filled_slots / committee_slots, 1.0)
    score = (
        (0.45 * top_similarity)
        + (0.35 * retrieval_coverage)
        + (0.20 * committee_coverage)
    )

    reasons: list[str] = []
    if top_similarity >= 0.75:
        reasons.append("strong_topic_match")
    elif top_similarity >= 0.45:
        reasons.append("moderate_topic_match")
    else:
        reasons.append("weak_topic_match")

    if retrieval_coverage >= 0.8:
        reasons.append("enough_similar_defenses")
    else:
        reasons.append("limited_similar_defenses")

    if committee_coverage < 1.0:
        reasons.append("incomplete_committee")
        score = min(score, 0.35)
    if score < 0.45:
        reasons.append("review_manually")

    return min(score, 1.0), reasons
