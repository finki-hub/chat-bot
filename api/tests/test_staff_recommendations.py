from dataclasses import replace
from uuid import uuid4

import httpx
import pytest

from app.api.recommendation_presenters import mentor_score
from app.data import StaffDirectoryUnavailableError, get_active_staff_names
from app.recommenders.config import ScoringWeights
from app.recommenders.selection import select_committee
from app.recommenders.signals import build_expertise_index, score_people
from app.recommenders.types import (
    MentorPriorIndex,
    Mode,
    RankedPeople,
    Recommendation,
    RetrievedPaper,
    SelectionConstraints,
)
from app.schemas.recommendations import RecommendationRequestSchema
from app.services.recommendations import (
    InactiveStaffRequestedError,
    _check_active_staff,
)


class FakeStaffClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response

    async def get(self, url: str) -> httpx.Response:
        return self.response


@pytest.mark.anyio
async def test_active_staff_names_include_only_active_entries(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    monkeypatch.setattr(
        "app.data.staff.get_http_client",
        lambda: FakeStaffClient(
            httpx.Response(
                200,
                request=httpx.Request("GET", "https://staff.example.test"),
                json={
                    "data": [
                        {"name": "Active Professor", "active": "1"},
                        {"name": "Inactive Professor", "active": "0"},
                        {"name": "  Spaced Active  ", "active": "1"},
                    ],
                },
            ),
        ),
    )

    active_names = await get_active_staff_names()

    assert active_names == frozenset({"Active Professor", "Spaced Active"})


@pytest.mark.anyio
async def test_active_staff_names_reject_empty_active_directory(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    monkeypatch.setattr(
        "app.data.staff.get_http_client",
        lambda: FakeStaffClient(
            httpx.Response(
                200,
                request=httpx.Request("GET", "https://staff.example.test"),
                json={"data": [{"name": "Inactive Professor", "active": "0"}]},
            ),
        ),
    )

    with pytest.raises(StaffDirectoryUnavailableError, match="no active staff"):
        await get_active_staff_names()


def test_select_committee_filters_inactive_members_and_prior_candidates() -> None:
    ranked = RankedPeople(
        blended={
            "Active One": 0.6,
            "Active Two": 0.5,
            "Inactive High Score": 100.0,
        },
        defense={},
        expertise={},
        coauthor={},
        mentor_score={"Active Mentor": 1.0},
        pair_score={},
        supporting={},
        expertise_supporting={},
        mentor_prior_weight=1.5,
    )

    rec = select_committee(
        ranked,
        Mode.MEMBERS_ONLY,
        given_mentor="Active Mentor",
        mentor_topk=3,
        mentor_prior=MentorPriorIndex(
            by_mentor={
                "Active Mentor": {
                    "Inactive Prior": 100.0,
                    "Inactive High Score": 90.0,
                },
            },
        ),
        constraints=SelectionConstraints(
            allowed=frozenset({"Active Mentor", "Active One", "Active Two"}),
        ),
    )

    assert rec.mentor == "Active Mentor"
    assert rec.members == ("Active One", "Active Two")


def test_select_committee_filters_inactive_mentor_candidates() -> None:
    ranked = RankedPeople(
        blended={"Active Member One": 0.4, "Active Member Two": 0.3},
        defense={},
        expertise={},
        coauthor={},
        mentor_score={"Inactive Mentor": 100.0, "Active Mentor": 1.0},
        pair_score={},
        supporting={},
        expertise_supporting={},
    )

    rec = select_committee(
        ranked,
        Mode.FULL,
        given_mentor=None,
        mentor_topk=3,
        constraints=SelectionConstraints(
            allowed=frozenset(
                {"Active Mentor", "Active Member One", "Active Member Two"},
            ),
        ),
    )

    assert rec.mentor == "Active Mentor"
    assert rec.members == ("Active Member One", "Active Member Two")


def _shortlist_ranked() -> RankedPeople:
    return RankedPeople(
        blended={"Member One": 0.9, "Member Two": 0.8, "Member Three": 0.7},
        defense={},
        expertise={},
        coauthor={},
        mentor_score={"Assistant": 9.0, "Mentor A": 3.0, "Mentor B": 2.0},
        pair_score={},
        supporting={},
        expertise_supporting={},
    )


def test_select_committee_never_recommends_an_ineligible_mentor() -> None:
    rec = select_committee(
        _shortlist_ranked(),
        Mode.FULL,
        given_mentor=None,
        mentor_topk=3,
        constraints=SelectionConstraints(
            mentor_allowed=frozenset({"Mentor A", "Mentor B"}),
            alternative_count=10,
        ),
    )

    assert rec.mentor == "Mentor A"
    assert {alternative.mentor for alternative in rec.alternatives} == {
        "Mentor A",
        "Mentor B",
    }


def test_select_committee_can_shortlist_distinct_mentors() -> None:
    rec = select_committee(
        _shortlist_ranked(),
        Mode.FULL,
        given_mentor=None,
        mentor_topk=3,
        constraints=SelectionConstraints(
            mentor_allowed=frozenset({"Mentor A", "Mentor B"}),
            alternative_count=5,
            distinct_mentors=True,
        ),
    )

    assert [alternative.mentor for alternative in rec.alternatives] == [
        "Mentor A",
        "Mentor B",
    ]
    assert rec.alternatives[0].members == ("Member One", "Member Two")


def test_select_committee_keeps_required_member_without_evidence() -> None:
    rec = select_committee(
        _shortlist_ranked(),
        Mode.MEMBERS_ONLY,
        given_mentor="Mentor A",
        mentor_topk=3,
        constraints=SelectionConstraints(include=frozenset({"No Evidence"})),
    )

    assert "No Evidence" in rec.members
    assert "Member One" in rec.members


def test_select_committee_can_seat_required_person_as_member_only() -> None:
    rec = select_committee(
        _shortlist_ranked(),
        Mode.FULL,
        given_mentor=None,
        mentor_topk=3,
        constraints=SelectionConstraints(
            include=frozenset({"Mentor A"}),
            mentor_allowed=frozenset({"Mentor A", "Mentor B"}),
            include_as_members=True,
        ),
    )

    assert rec.mentor == "Mentor B"
    assert "Mentor A" in rec.members


def test_shortlist_fills_with_paper_backed_mentors_after_defense_mentors() -> None:
    ranked = replace(
        _shortlist_ranked(),
        paper_mentors={"Paper Mentor": 1.0, "Assistant": 0.9, "Mentor A": 0.8},
    )

    rec = select_committee(
        ranked,
        Mode.FULL,
        given_mentor=None,
        mentor_topk=3,
        constraints=SelectionConstraints(
            mentor_allowed=frozenset({"Mentor A", "Mentor B", "Paper Mentor"}),
            alternative_count=5,
            distinct_mentors=True,
        ),
    )

    assert rec.mentor == "Mentor A"
    assert [(alt.mentor, alt.mentor_basis) for alt in rec.alternatives] == [
        ("Mentor A", "defenses"),
        ("Mentor B", "defenses"),
        ("Paper Mentor", "papers"),
    ]


def test_paper_mentors_are_ignored_without_shortlist_mode() -> None:
    ranked = replace(_shortlist_ranked(), paper_mentors={"Paper Mentor": 1.0})

    rec = select_committee(
        ranked,
        Mode.FULL,
        given_mentor=None,
        mentor_topk=5,
        constraints=SelectionConstraints(alternative_count=20),
    )

    assert "Paper Mentor" not in {alt.mentor for alt in rec.alternatives}


def test_paper_backed_mentors_require_topical_and_relative_strength() -> None:
    weights = ScoringWeights(expertise_recency_half_life_days=0.0)
    expertise = build_expertise_index(
        [
            RetrievedPaper("1", "Strong", ("Strong Fit",), distance=0.3),
            RetrievedPaper("2", "Strong 2", ("Strong Fit",), distance=0.35),
            RetrievedPaper("3", "Off topic", ("Weak Match",), distance=0.5),
            RetrievedPaper("4", "Off topic 2", ("Weak Match",), distance=0.5),
            RetrievedPaper("5", "Close", ("Minor Author",), distance=0.42),
            RetrievedPaper("6", "Far", ("Far Author",), distance=0.46),
        ],
        weights,
    )
    ranked = score_people([], expertise, None, weights, Mode.FULL)

    assert set(ranked.paper_mentors) == {"Strong Fit", "Minor Author"}

    strict = replace(weights, paper_mentor_min_similarity=0.6)
    strict_ranked = score_people([], expertise, None, strict, Mode.FULL)

    assert set(strict_ranked.paper_mentors) == {"Strong Fit"}


def test_given_mentor_keeps_defense_evidence() -> None:
    diploma_id = uuid4()
    ranked = replace(_shortlist_ranked(), supporting={"Mentor A": [diploma_id]})
    rec = Recommendation(
        mode=Mode.MEMBERS_ONLY,
        mentor="Mentor A",
        members=("Member One", "Member Two"),
        mentor_is_given=True,
    )

    assert mentor_score(rec, ranked).supporting_diploma_ids == [diploma_id]


def test_check_staff_rejects_active_but_ineligible_mentor() -> None:
    payload = RecommendationRequestSchema(title="Test", mentor="Secretary")
    active_staff = frozenset({"Professor", "Secretary"})
    mentor_eligible = frozenset({"Professor"})

    with pytest.raises(InactiveStaffRequestedError) as exc_info:
        _check_active_staff(payload, active_staff, mentor_eligible)

    assert exc_info.value.field == "mentor"
    assert exc_info.value.names == ("Secretary",)


def test_check_staff_accepts_mentor_eligible_name() -> None:
    payload = RecommendationRequestSchema(title="Test", mentor="Professor")

    _check_active_staff(
        payload,
        frozenset({"Professor", "Secretary"}),
        frozenset({"Professor"}),
    )
