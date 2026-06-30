from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from uuid import UUID


class Mode(Enum):
    FULL = "full"
    MEMBERS_ONLY = "members_only"


@dataclass(frozen=True)
class RetrievedDiploma:
    id: UUID
    external_id: str
    title: str
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
    supporting: dict[frozenset[str], list[str]]


@dataclass(frozen=True)
class MentorPriorIndex:
    by_mentor: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RankedPeople:
    blended: dict[str, float]
    defense: dict[str, float]
    expertise: dict[str, float]
    coauthor: dict[frozenset[str], float]
    mentor_score: dict[str, float]
    pair_score: dict[frozenset[str], float]
    supporting: dict[str, list[UUID]]
    expertise_supporting: dict[str, list[str]]
    pair_affinity_weight: float = 0.0
    coauthor_weight: float = 0.0
    mentor_prior_weight: float = 0.0
    coauthor_prior_weight: float = 0.0


@dataclass(frozen=True)
class SelectionConstraints:
    exclude: frozenset[str] = frozenset()
    include: frozenset[str] = frozenset()
    alternative_count: int = 3


@dataclass(frozen=True)
class CommitteeAlternative:
    mentor: str | None
    members: tuple[str, ...]
    score: float
    supporting_diploma_ids: list[UUID] = field(default_factory=list)


@dataclass(frozen=True)
class Recommendation:
    mode: Mode
    mentor: str | None
    members: tuple[str, ...]
    mentor_is_given: bool
    supporting_diploma_ids: list[UUID] = field(default_factory=list)
    member_scores: dict[str, float] = field(default_factory=dict)
    alternatives: tuple[CommitteeAlternative, ...] = ()
