from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import assert_never
from uuid import UUID

from app.recommenders.signals import _minmax
from app.recommenders.types import (
    CommitteeAlternative,
    MentorPriorIndex,
    Mode,
    RankedPeople,
    Recommendation,
    SelectionConstraints,
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


def _apply_prior(
    candidates: dict[str, float],
    prior_index: MentorPriorIndex | None,
    mentor: str | None,
    weight: float,
    constraints: SelectionConstraints,
) -> dict[str, float]:
    if prior_index is None or mentor is None or weight <= 0.0:
        return candidates
    prior = prior_index.by_mentor.get(mentor)
    if not prior:
        return candidates
    out = dict(candidates)
    for name, score in _minmax(prior).items():
        allowed = constraints.allowed
        if name == mentor or name in constraints.exclude:
            continue
        if allowed is not None and name not in allowed:
            continue
        out[name] = out.get(name, 0.0) + weight * score
    return out


def _members_objective(
    members: tuple[str, ...],
    candidates: Mapping[str, float],
    ranked: RankedPeople,
) -> float:
    if not members:
        return 0.0
    if len(members) == 1:
        return candidates[members[0]]

    a, b = members
    key = frozenset((a, b))
    return (
        candidates[a]
        + candidates[b]
        + ranked.pair_affinity_weight * ranked.pair_score.get(key, 0.0)
        + ranked.coauthor_weight * ranked.coauthor.get(key, 0.0)
    )


def _candidate_member_sets(
    candidates: Mapping[str, float],
    constraints: SelectionConstraints,
    mentor: str | None,
) -> list[tuple[str, ...]]:
    pool = sorted(
        (
            name
            for name in candidates
            if name not in constraints.exclude and name != mentor
        ),
        key=lambda name: candidates[name],
        reverse=True,
    )
    return [tuple(pool)] if len(pool) < 2 else list(itertools.combinations(pool, 2))


def _member_alternatives(
    candidates: Mapping[str, float],
    ranked: RankedPeople,
    constraints: SelectionConstraints,
    mentor: str | None,
    mentor_component: float,
) -> list[CommitteeAlternative]:
    alternatives: list[CommitteeAlternative] = []
    for members in _candidate_member_sets(candidates, constraints, mentor):
        if constraints.include and not constraints.include <= {
            name for name in {mentor, *members} if name is not None
        }:
            continue
        alternatives.append(
            CommitteeAlternative(
                mentor=mentor,
                members=members,
                score=mentor_component
                + _members_objective(members, candidates, ranked),
                supporting_diploma_ids=_collect_supporting(ranked, mentor, members),
            ),
        )
    alternatives.sort(key=lambda alt: alt.score, reverse=True)
    return alternatives


def _candidates_for_mentor(
    ranked: RankedPeople,
    mentor: str | None,
    mentor_prior: MentorPriorIndex | None,
    coauthor_prior: MentorPriorIndex | None,
    constraints: SelectionConstraints,
) -> dict[str, float]:
    candidates = {
        name: score
        for name, score in ranked.blended.items()
        if name != mentor
        and name not in constraints.exclude
        and (constraints.allowed is None or name in constraints.allowed)
    }
    # A required person may have no retrieval evidence for this topic.
    for name in constraints.include:
        if (
            name != mentor
            and name not in constraints.exclude
            and (constraints.allowed is None or name in constraints.allowed)
        ):
            candidates.setdefault(name, 0.0)
    candidates = _apply_prior(
        candidates,
        mentor_prior,
        mentor,
        ranked.mentor_prior_weight,
        constraints,
    )
    return _apply_prior(
        candidates,
        coauthor_prior,
        mentor,
        ranked.coauthor_prior_weight,
        constraints,
    )


def _can_mentor(name: str, constraints: SelectionConstraints) -> bool:
    return (
        name not in constraints.exclude
        and (constraints.allowed is None or name in constraints.allowed)
        and (constraints.mentor_allowed is None or name in constraints.mentor_allowed)
        and not (constraints.include_as_members and name in constraints.include)
    )


def _eligible_paper_mentors(
    ranked: RankedPeople,
    constraints: SelectionConstraints,
) -> list[str]:
    scores = {
        name: score
        for name, score in ranked.paper_scores.items()
        if _can_mentor(name, constraints)
    }
    top = max(scores.values(), default=0.0)
    if top <= 0.0:
        return []
    return [
        name
        for name, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if score / top >= ranked.paper_mentor_min_relative
        and ranked.paper_similarity.get(name, 0.0) >= ranked.paper_mentor_min_similarity
    ]


def _paper_mentor_alternatives(
    ranked: RankedPeople,
    slots: int,
    taken: frozenset[str],
    constraints: SelectionConstraints,
    committees_for: Callable[[str], list[CommitteeAlternative]],
) -> list[CommitteeAlternative]:
    alternatives: list[CommitteeAlternative] = []
    for mentor in _eligible_paper_mentors(ranked, constraints):
        if len(alternatives) >= slots:
            break
        if mentor in taken:
            continue
        committees = committees_for(mentor)
        if committees:
            alternatives.append(replace(committees[0], mentor_basis="papers"))
    return alternatives


def _prior_for_members(
    mentor: str | None,
    mentor_prior: MentorPriorIndex | None,
    member_prior: MentorPriorIndex | None,
) -> MentorPriorIndex | None:
    if member_prior is not None and mentor and member_prior.by_mentor.get(mentor):
        return member_prior
    return mentor_prior


def _committees_for_mentor(
    ranked: RankedPeople,
    mentor: str | None,
    mentor_component: float,
    priors: tuple[
        MentorPriorIndex | None,
        MentorPriorIndex | None,
        MentorPriorIndex | None,
    ],
    constraints: SelectionConstraints,
) -> tuple[list[CommitteeAlternative], dict[str, float]]:
    """Rank a mentor with the full prior, then pick members with the member prior."""
    mentor_prior, member_prior, coauthor_prior = priors
    ranking_candidates = _candidates_for_mentor(
        ranked,
        mentor,
        mentor_prior,
        coauthor_prior,
        constraints,
    )
    ranking = _member_alternatives(
        ranking_candidates,
        ranked,
        constraints,
        mentor,
        mentor_component,
    )
    prior = _prior_for_members(mentor, mentor_prior, member_prior)
    if prior is mentor_prior or not ranking:
        return ranking, ranking_candidates

    candidates = _candidates_for_mentor(
        ranked,
        mentor,
        prior,
        coauthor_prior,
        constraints,
    )
    committees = _member_alternatives(
        candidates,
        ranked,
        constraints,
        mentor,
        mentor_component,
    )
    if not committees:
        return ranking, ranking_candidates
    # Keep this mentor's rank against the others unchanged.
    offset = ranking[0].score - committees[0].score
    return [
        replace(committee, score=committee.score + offset) for committee in committees
    ], candidates


def _full_alternatives(
    ranked: RankedPeople,
    mentor_topk: int,
    priors: tuple[
        MentorPriorIndex | None,
        MentorPriorIndex | None,
        MentorPriorIndex | None,
    ],
    constraints: SelectionConstraints,
) -> tuple[list[CommitteeAlternative], dict[str, float]]:
    mentor_norm = _minmax(ranked.mentor_score)
    mentors = [
        name
        for name, _score in sorted(
            ranked.mentor_score.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if _can_mentor(name, constraints)
    ][:mentor_topk]
    alternatives: list[CommitteeAlternative] = []
    candidates_by_mentor: dict[str, dict[str, float]] = {}

    def committees_for(mentor: str, component: float) -> list[CommitteeAlternative]:
        committees, candidates = _committees_for_mentor(
            ranked,
            mentor,
            component,
            priors,
            constraints,
        )
        candidates_by_mentor[mentor] = candidates
        return committees

    for mentor in mentors:
        alternatives.extend(committees_for(mentor, mentor_norm.get(mentor, 0.0)))

    alternatives.sort(key=lambda alt: alt.score, reverse=True)
    if constraints.distinct_mentors:
        paper_alternatives = _paper_mentor_alternatives(
            ranked,
            mentor_topk - len(mentors),
            frozenset(mentors),
            constraints,
            lambda mentor: committees_for(mentor, 0.0),
        )
        # Cap scores so defense-backed mentors stay first for any consumer.
        ceiling = alternatives[-1].score if alternatives else float("inf")
        for alternative in paper_alternatives:
            ceiling = min(ceiling, alternative.score)
            alternatives.append(replace(alternative, score=ceiling))
    if alternatives and alternatives[0].mentor is not None:
        return alternatives, candidates_by_mentor[alternatives[0].mentor]
    first_mentor = next(iter(candidates_by_mentor), None)
    return alternatives, (
        candidates_by_mentor[first_mentor] if first_mentor is not None else {}
    )


def _best_per_mentor(
    alternatives: Sequence[CommitteeAlternative],
) -> list[CommitteeAlternative]:
    best: dict[str | None, CommitteeAlternative] = {}
    for alternative in alternatives:
        best.setdefault(alternative.mentor, alternative)
    return list(best.values())


def select_committee(
    ranked: RankedPeople,
    mode: Mode,
    given_mentor: str | None,
    *,
    mentor_topk: int,
    exclude: Iterable[str] = (),
    mentor_prior: MentorPriorIndex | None = None,
    coauthor_prior: MentorPriorIndex | None = None,
    constraints: SelectionConstraints | None = None,
    member_prior: MentorPriorIndex | None = None,
) -> Recommendation:
    active_constraints = constraints or SelectionConstraints(exclude=frozenset(exclude))
    if constraints is None and exclude:
        active_constraints = SelectionConstraints(exclude=frozenset(exclude))
    priors = (mentor_prior, member_prior, coauthor_prior)

    match mode:
        case Mode.MEMBERS_ONLY:
            alternatives, candidates = _committees_for_mentor(
                ranked,
                given_mentor,
                0.0,
                priors,
                active_constraints,
            )
        case Mode.FULL:
            alternatives, candidates = _full_alternatives(
                ranked,
                mentor_topk,
                priors,
                active_constraints,
            )
        case unreachable:
            assert_never(unreachable)

    selected = alternatives[0] if alternatives else CommitteeAlternative(None, (), 0.0)
    shortlist = (
        _best_per_mentor(alternatives)
        if active_constraints.distinct_mentors and mode is Mode.FULL
        else alternatives
    )
    selected_alternatives = tuple(shortlist[: active_constraints.alternative_count])
    mentor = given_mentor if mode is Mode.MEMBERS_ONLY else selected.mentor
    members = selected.members

    return Recommendation(
        mode=mode,
        mentor=mentor,
        members=members,
        mentor_is_given=(mode is Mode.MEMBERS_ONLY),
        supporting_diploma_ids=selected.supporting_diploma_ids,
        member_scores={name: candidates.get(name, 0.0) for name in members},
        alternatives=selected_alternatives,
    )
