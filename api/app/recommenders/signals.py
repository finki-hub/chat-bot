from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from uuid import UUID

from app.recommenders.config import ScoringWeights
from app.recommenders.types import (
    CoauthorIndex,
    ExpertiseIndex,
    MentorPriorIndex,
    Mode,
    RankedPeople,
    RetrievedDiploma,
    RetrievedPaper,
)


def build_mentor_prior(
    defenses: Iterable[tuple[str | None, str | None, str | None]],
) -> MentorPriorIndex:
    by_mentor: dict[str, dict[str, float]] = {}
    for mentor, member1, member2 in defenses:
        if not mentor:
            continue
        bucket = by_mentor.setdefault(mentor, {})
        for member in (member1, member2):
            if member:
                bucket[member] = bucket.get(member, 0.0) + 1.0
    return MentorPriorIndex(by_mentor=by_mentor)


def build_coauthor_prior(papers: Iterable[Sequence[str]]) -> MentorPriorIndex:
    by_person: dict[str, dict[str, float]] = {}
    for authors in papers:
        uniq = sorted({a for a in authors if a})
        for a, b in itertools.combinations(uniq, 2):
            a_bucket = by_person.setdefault(a, {})
            a_bucket[b] = a_bucket.get(b, 0.0) + 1.0
            b_bucket = by_person.setdefault(b, {})
            b_bucket[a] = b_bucket.get(a, 0.0) + 1.0
    return MentorPriorIndex(by_mentor=by_person)


def _minmax[K](scores: Mapping[K, float]) -> dict[K, float]:
    if not scores:
        return {}
    values = scores.values()
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return dict.fromkeys(scores, 0.0)
    span = hi - lo
    return {key: (value - lo) / span for key, value in scores.items()}


def _half_life_decay(delta_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (delta_days / half_life_days)


def members(diploma: RetrievedDiploma) -> list[str]:
    return [m for m in (diploma.member1, diploma.member2) if m]


def _diploma_base_weight(
    diploma: RetrievedDiploma,
    weights: ScoringWeights,
    *,
    now: date | None,
) -> float:
    weight = weights.similarity_weight * diploma.similarity
    if diploma.rerank_score is not None:
        weight += weights.rerank_weight * diploma.rerank_score
    if (
        weights.recency_half_life_days > 0
        and diploma.date_of_submission is not None
        and now is not None
    ):
        delta_days = (now - diploma.date_of_submission).days
        weight *= _half_life_decay(float(delta_days), weights.recency_half_life_days)
    return weight


def _defense_counts(retrieved: Sequence[RetrievedDiploma]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for diploma in retrieved:
        for name in {diploma.mentor, *members(diploma)}:
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _accumulate_defense_scores(
    retrieved: Sequence[RetrievedDiploma],
    weights: ScoringWeights,
    *,
    mentor_hint: str | None = None,
    now: date | None = None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[frozenset[str], float],
    dict[str, list[UUID]],
]:
    person_score: dict[str, float] = {}
    mentor_score: dict[str, float] = {}
    pair_score: dict[frozenset[str], float] = {}
    supporting: dict[str, list[UUID]] = {}

    for diploma in retrieved:
        weight = _diploma_base_weight(diploma, weights, now=now)
        if mentor_hint is not None and diploma.mentor == mentor_hint:
            weight *= 1.0 + weights.mentor_match_boost

        diploma_members = members(diploma)
        for name in {diploma.mentor, *diploma_members}:
            person_score[name] = person_score.get(name, 0.0) + weight
            supporting.setdefault(name, []).append(diploma.id)

        mentor_score[diploma.mentor] = mentor_score.get(diploma.mentor, 0.0) + weight

        if len(diploma_members) == 2:
            key = frozenset(diploma_members)
            if len(key) == 2:
                pair_score[key] = pair_score.get(key, 0.0) + weight

    return person_score, mentor_score, pair_score, supporting


def _paper_similarity(
    paper: RetrievedPaper,
    half_life_days: float,
    now_year: int | None,
) -> float:
    similarity = 1.0 - paper.distance
    if half_life_days and paper.year and now_year:
        similarity *= _half_life_decay(
            (now_year - paper.year) * 365.0,
            half_life_days,
        )
    return similarity


def build_expertise_index(
    papers: Sequence[RetrievedPaper],
    weights: ScoringWeights,
    *,
    canonical_set: Iterable[str] | None = None,
    now_year: int | None = None,
) -> ExpertiseIndex:
    canonical = set(canonical_set) if canonical_set is not None else None
    sims: dict[str, list[float]] = {}
    supporting: dict[str, list[str]] = {}
    for paper in papers:
        similarity = _paper_similarity(
            paper,
            weights.expertise_recency_half_life_days,
            now_year,
        )
        for author in paper.coauthors:
            if canonical is not None and author not in canonical:
                continue
            sims.setdefault(author, []).append(similarity)
            supporting.setdefault(author, []).append(paper.title)
    by_professor = {
        name: sum(sorted(values, reverse=True)[: weights.expertise_top_papers])
        for name, values in sims.items()
    }
    return ExpertiseIndex(by_professor=by_professor, supporting=supporting)


def _accumulate_coauthor_edges(
    papers: Sequence[RetrievedPaper],
    weights: ScoringWeights,
    *,
    canonical_set: Iterable[str] | None = None,
    now_year: int | None = None,
) -> CoauthorIndex:
    canonical = set(canonical_set) if canonical_set is not None else None
    edges: dict[frozenset[str], float] = {}
    supporting: dict[frozenset[str], list[str]] = {}
    for paper in papers:
        weight = _paper_similarity(
            paper,
            weights.coauthor_recency_half_life_days,
            now_year,
        )
        canon = [c for c in paper.coauthors if canonical is None or c in canonical]
        for a, b in itertools.combinations(sorted(set(canon)), 2):
            key = frozenset((a, b))
            edges[key] = edges.get(key, 0.0) + weight
            supporting.setdefault(key, []).append(paper.title)
    return CoauthorIndex(edges=edges, supporting=supporting)


def score_people(
    retrieved: Sequence[RetrievedDiploma],
    expertise: ExpertiseIndex,
    coauthors: CoauthorIndex | None,
    weights: ScoringWeights,
    mode: Mode,
    given_mentor: str | None = None,
    *,
    now: date | None = None,
) -> RankedPeople:
    defense, mentor_score, pair_score, supporting = _accumulate_defense_scores(
        retrieved,
        weights,
        mentor_hint=given_mentor,
        now=now,
    )
    n_def = _defense_counts(retrieved)

    d_norm = _minmax(defense)
    e_norm = _minmax(expertise.by_professor)
    c_norm: dict[frozenset[str], float] = (
        _minmax(coauthors.edges) if coauthors is not None else {}
    )

    blended: dict[str, float] = {}
    for name in set(d_norm) | set(e_norm):
        w_exp = weights.expertise_weight
        if n_def.get(name, 0) < weights.cold_start_defense_floor:
            w_exp *= weights.cold_start_expertise_boost
        blended[name] = d_norm.get(name, 0.0) + w_exp * e_norm.get(name, 0.0)

    if mode is Mode.MEMBERS_ONLY and given_mentor is not None and coauthors is not None:
        b_raw = {
            n: coauthors.edges.get(frozenset((given_mentor, n)), 0.0)
            for n in blended
            if n != given_mentor
        }
        b_norm = _minmax(b_raw)
        for n, s in b_norm.items():
            blended[n] += weights.coauthor_member_boost * s

    if mode is Mode.MEMBERS_ONLY and given_mentor is not None:
        blended.pop(given_mentor, None)

    return RankedPeople(
        blended=blended,
        defense=d_norm,
        expertise=e_norm,
        coauthor=c_norm,
        mentor_score=mentor_score,
        pair_score=pair_score,
        supporting=supporting,
        expertise_supporting=expertise.supporting,
        pair_affinity_weight=weights.pair_affinity_weight,
        coauthor_weight=weights.coauthor_weight,
        mentor_prior_weight=weights.mentor_prior_weight,
        coauthor_prior_weight=weights.coauthor_prior_weight,
    )
