import httpx
import pytest

from app.data import StaffDirectoryUnavailableError, get_active_staff_names
from app.recommenders.selection import select_committee
from app.recommenders.types import (
    MentorPriorIndex,
    Mode,
    RankedPeople,
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
