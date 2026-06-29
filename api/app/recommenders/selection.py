from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
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


def _committee_contains_required(
    mentor: str | None,
    members: Sequence[str],
    required: frozenset[str],
) -> bool:
    if not required:
        return True
    committee = {mentor, *members}
    return required <= {name for name in committee if name is not None}


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


def _pair_objective(
    pair: tuple[str, str],
    candidates: Mapping[str, float],
    ranked: RankedPeople,
) -> float:
    a, b = pair
    key = frozenset((a, b))
    return (
        candidates[a]
        + candidates[b]
        + ranked.pair_affinity_weight * ranked.pair_score.get(key, 0.0)
        + ranked.coauthor_weight * ranked.coauthor.get(key, 0.0)
    )


def _candidate_pairs(
    candidates: Mapping[str, float],
    constraints: SelectionConstraints,
    mentor: str | None,
) -> list[tuple[str, str]]:
    pool = sorted(
        (
            name
            for name in candidates
            if name not in constraints.exclude and name != mentor
        ),
        key=lambda name: candidates[name],
        reverse=True,
    )
    if len(pool) < 2:
        return []
    return list(itertools.combinations(pool, 2))


def _member_alternatives(
    candidates: Mapping[str, float],
    ranked: RankedPeople,
    constraints: SelectionConstraints,
    mentor: str | None,
    mentor_component: float,
) -> list[CommitteeAlternative]:
    alternatives: list[CommitteeAlternative] = []
    for pair in _candidate_pairs(candidates, constraints, mentor):
        if not _committee_contains_required(mentor, pair, constraints.include):
            continue
        alternatives.append(
            CommitteeAlternative(
                mentor=mentor,
                members=pair,
                score=mentor_component + _pair_objective(pair, candidates, ranked),
                supporting_diploma_ids=_collect_supporting(ranked, mentor, pair),
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
        if name != mentor and name not in constraints.exclude
    }
    candidates = _apply_prior(
        candidates,
        mentor_prior,
        mentor,
        ranked.mentor_prior_weight,
    )
    return _apply_prior(
        candidates,
        coauthor_prior,
        mentor,
        ranked.coauthor_prior_weight,
    )


def _full_alternatives(
    ranked: RankedPeople,
    mentor_topk: int,
    mentor_prior: MentorPriorIndex | None,
    coauthor_prior: MentorPriorIndex | None,
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
        if name not in constraints.exclude
    ][:mentor_topk]
    alternatives: list[CommitteeAlternative] = []
    selected_member_scores: dict[str, float] = {}

    for mentor in mentors:
        candidates = _candidates_for_mentor(
            ranked,
            mentor,
            mentor_prior,
            coauthor_prior,
            constraints,
        )
        alternatives.extend(
            _member_alternatives(
                candidates,
                ranked,
                constraints,
                mentor,
                mentor_norm.get(mentor, 0.0),
            ),
        )
        if not selected_member_scores:
            selected_member_scores = candidates

    alternatives.sort(key=lambda alt: alt.score, reverse=True)
    if alternatives:
        best_mentor = alternatives[0].mentor
        selected_member_scores = _candidates_for_mentor(
            ranked,
            best_mentor,
            mentor_prior,
            coauthor_prior,
            constraints,
        )
    return alternatives, selected_member_scores


def _members_only_alternatives(
    ranked: RankedPeople,
    given_mentor: str | None,
    mentor_prior: MentorPriorIndex | None,
    coauthor_prior: MentorPriorIndex | None,
    constraints: SelectionConstraints,
) -> tuple[list[CommitteeAlternative], dict[str, float]]:
    candidates = _candidates_for_mentor(
        ranked,
        given_mentor,
        mentor_prior,
        coauthor_prior,
        constraints,
    )
    alternatives = _member_alternatives(
        candidates,
        ranked,
        constraints,
        given_mentor,
        0.0,
    )
    return alternatives, candidates


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
) -> Recommendation:
    active_constraints = constraints or SelectionConstraints(exclude=frozenset(exclude))
    if constraints is None and exclude:
        active_constraints = SelectionConstraints(exclude=frozenset(exclude))

    match mode:
        case Mode.MEMBERS_ONLY:
            alternatives, candidates = _members_only_alternatives(
                ranked,
                given_mentor,
                mentor_prior,
                coauthor_prior,
                active_constraints,
            )
        case Mode.FULL:
            alternatives, candidates = _full_alternatives(
                ranked,
                mentor_topk,
                mentor_prior,
                coauthor_prior,
                active_constraints,
            )
        case unreachable:
            assert_never(unreachable)

    selected = alternatives[0] if alternatives else CommitteeAlternative(None, (), 0.0)
    selected_alternatives = tuple(alternatives[: active_constraints.alternative_count])
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
