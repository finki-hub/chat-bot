from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from uuid import UUID

from app.recommenders.config import ScoringWeights


class Mode(Enum):
    FULL = "full"
    MEMBERS_ONLY = "members_only"


@dataclass(frozen=True)
class RetrievedDiploma:
    id: UUID
    external_id: str
    mentor: str
    member1: str | None
    member2: str | None
    similarity: float
    rerank_score: float | None = None
    date_of_submission: date | None = None


@dataclass(frozen=True)
class RetrievedPaper:
    external_id: str
    title: str
    coauthors: tuple[str, ...]
    distance: float
    year: int | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class ExpertiseIndex:
    by_professor: dict[str, float]
    supporting: dict[str, list[str]]


@dataclass(frozen=True)
class CoauthorIndex:
    edges: dict[frozenset[str], float]
    supporting: dict[
        frozenset[str],
        list[str],
    ]


@dataclass(frozen=True)
class MentorPriorIndex:
    by_mentor: dict[str, dict[str, float]]


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


@dataclass(frozen=True)
class RankedPeople:
    blended: dict[str, float]
    defense: dict[str, float]
    expertise: dict[str, float]
    coauthor: dict[frozenset[str], float]
    mentor_score: dict[str, float]
    pair_score: dict[frozenset[str], float]
    supporting: dict[str, list[UUID]]
    pair_affinity_weight: float = 0.0
    coauthor_weight: float = 0.0
    mentor_prior_weight: float = 0.0
    coauthor_prior_weight: float = 0.0


@dataclass(frozen=True)
class Recommendation:
    mode: Mode
    mentor: str | None
    members: tuple[str, ...]
    mentor_is_given: bool
    supporting_diploma_ids: list[UUID] = field(default_factory=list)
    member_scores: dict[str, float] = field(default_factory=dict)


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
    if half_life_days <= 0:  # recency OFF -> no-op weight
        return 1.0
    return 0.5 ** (delta_days / half_life_days)


def _members(diploma: RetrievedDiploma) -> list[str]:
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
        for name in {diploma.mentor, *_members(diploma)}:
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

        members = _members(diploma)
        for name in {diploma.mentor, *members}:
            person_score[name] = person_score.get(name, 0.0) + weight
            supporting.setdefault(name, []).append(diploma.id)

        mentor_score[diploma.mentor] = mentor_score.get(diploma.mentor, 0.0) + weight

        if len(members) == 2:
            key = frozenset(members)
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


def _best_unordered_pair(
    candidates: Mapping[str, float],
    ranked: RankedPeople,
    *,
    exclude: Iterable[str] = (),
) -> tuple[str, ...]:
    excluded = set(exclude)
    pool = sorted(
        (n for n in candidates if n not in excluded),
        key=lambda n: candidates[n],
        reverse=True,
    )

    if len(pool) < 2:
        return tuple(pool)

    best_pair: tuple[str, str] | None = None
    best_objective = float("-inf")
    for a, b in itertools.combinations(pool, 2):
        key = frozenset((a, b))
        objective = (
            candidates[a]
            + candidates[b]
            + ranked.pair_affinity_weight * ranked.pair_score.get(key, 0.0)
            + ranked.coauthor_weight * ranked.coauthor.get(key, 0.0)
        )
        if objective > best_objective:
            best_objective = objective
            best_pair = (a, b)

    return best_pair if best_pair is not None else ()


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
            w_exp *= weights.cold_start_expertise_boost  # junior/cold-start boost
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
        pair_affinity_weight=weights.pair_affinity_weight,
        coauthor_weight=weights.coauthor_weight,
        mentor_prior_weight=weights.mentor_prior_weight,
        coauthor_prior_weight=weights.coauthor_prior_weight,
    )


def _apply_prior(
    candidates: dict[str, float],
    prior_index: MentorPriorIndex | None,
    mentor: str | None,
    weight: float,
) -> dict[str, float]:
    if prior_index is None or mentor is None or weight <= 0.0:
        return candidates
    prior = prior_index.by_mentor.get(mentor)
    if not prior:
        return candidates
    out = dict(candidates)
    for name, score in _minmax(prior).items():
        if name == mentor:
            continue
        out[name] = out.get(name, 0.0) + weight * score
    return out


def select_committee(
    ranked: RankedPeople,
    mode: Mode,
    given_mentor: str | None,
    *,
    mentor_topk: int,
    exclude: Iterable[str] = (),
    mentor_prior: MentorPriorIndex | None = None,
    coauthor_prior: MentorPriorIndex | None = None,
) -> Recommendation:
    excluded = set(exclude)

    if mode is Mode.MEMBERS_ONLY:
        mentor = given_mentor
        candidates = {n: s for n, s in ranked.blended.items() if n != given_mentor}
    else:
        top_mentors = sorted(
            ranked.mentor_score.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:mentor_topk]
        top_mentors = [
            (name, value) for name, value in top_mentors if name not in excluded
        ]
        mentor = top_mentors[0][0] if top_mentors else None
        candidates = {n: s for n, s in ranked.blended.items() if n != mentor}

    candidates = _apply_prior(
        candidates,
        mentor_prior,
        mentor,
        ranked.mentor_prior_weight,
    )
    candidates = _apply_prior(
        candidates,
        coauthor_prior,
        mentor,
        ranked.coauthor_prior_weight,
    )

    pair_exclude = (*excluded, mentor) if mentor is not None else tuple(excluded)
    members = _best_unordered_pair(candidates, ranked, exclude=pair_exclude)

    supporting_ids = _collect_supporting(ranked, mentor, members)

    return Recommendation(
        mode=mode,
        mentor=mentor,
        members=members,
        mentor_is_given=(mode is Mode.MEMBERS_ONLY),
        supporting_diploma_ids=supporting_ids,
        member_scores={name: candidates.get(name, 0.0) for name in members},
    )


def _collect_supporting(
    ranked: RankedPeople,
    mentor: str | None,
    members: Sequence[str],
) -> list[UUID]:
    seen: dict[UUID, None] = {}
    for name in (mentor, *members):
        if name is None:
            continue
        for diploma_id in ranked.supporting.get(name, []):
            seen.setdefault(diploma_id, None)
    return list(seen)
