import httpx
import pytest
from fastapi import HTTPException

from app.api.professors import list_professors
from app.data.staff import (
    StaffDirectoryUnavailableError,
    StaffMember,
    get_active_staff_names,
    get_mentor_eligible_names,
    get_staff_directory,
)


def _response(
    url: str,
    status_code: int,
    *,
    content: bytes | None = None,
    json: object | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", url)
    if content is not None:
        return httpx.Response(status_code, content=content, request=request)
    return httpx.Response(status_code, json=json, request=request)


class FakeStaffClient:
    def __init__(
        self,
        *,
        directus: httpx.Response,
        legacy: httpx.Response,
    ) -> None:
        self.directus = directus
        self.legacy = legacy

    async def get(self, url: str) -> httpx.Response:
        if "items/staff" in url:
            return self.directus
        return self.legacy


@pytest.mark.anyio
async def test_staff_directory_prefers_directus(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        directus=_response(
            "https://data.example.test/items/staff",
            200,
            json={
                "data": [
                    {
                        "id": 7,
                        "name": "Directus Professor",
                        "active": True,
                        "title": "д-р",
                        "position": "Доцент",
                        "photo": "https://example.test/photo.jpg",
                    },
                    {
                        "id": 8,
                        "name": "Inactive Professor",
                        "active": False,
                    },
                ],
            },
        ),
        legacy=_response(
            "https://staff.example.test",
            200,
            content=b'[{"name":"Legacy Professor","active":"1"}]',
        ),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    staff = await get_staff_directory()

    assert [member.name for member in staff] == ["Directus Professor"]
    assert staff[0].id == 7
    assert staff[0].position == "Доцент"
    assert staff[0].mentor_eligible is True
    assert staff[0].photo == "https://example.test/photo.jpg"


@pytest.mark.anyio
async def test_staff_directory_falls_back_to_legacy(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        directus=_response("https://data.example.test/items/staff", 503),
        legacy=_response(
            "https://staff.example.test",
            200,
            content=b'[{"name":"Legacy Professor","active":"1"}]',
        ),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    staff = await get_staff_directory()

    assert [member.name for member in staff] == ["Legacy Professor"]


@pytest.mark.anyio
async def test_staff_directory_raises_when_both_sources_fail(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        directus=_response("https://data.example.test/items/staff", 503),
        legacy=_response("https://staff.example.test", 503),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    with pytest.raises(StaffDirectoryUnavailableError):
        await get_staff_directory()


@pytest.mark.anyio
async def test_active_staff_names_use_directus_active_flag(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        directus=_response(
            "https://data.example.test/items/staff",
            200,
            json={
                "data": [
                    {"name": "Active Professor", "active": True},
                    {"name": "Inactive Professor", "active": False},
                ],
            },
        ),
        legacy=_response("https://staff.example.test", 503),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    assert await get_active_staff_names() == frozenset({"Active Professor"})


@pytest.mark.anyio
async def test_mentor_eligible_names_exclude_non_teaching_roles(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        directus=_response(
            "https://data.example.test/items/staff",
            200,
            json={
                "data": [
                    {
                        "name": "Eligible Professor",
                        "active": True,
                        "position": "Доцент",
                    },
                    {
                        "name": "Secretary",
                        "active": True,
                        "position": "Секретар",
                    },
                    {
                        "name": "Assistant",
                        "active": True,
                        "position": "Асистент докторанд",
                    },
                    {
                        "name": "Visiting Professor",
                        "active": True,
                        "position": "Визитинг професор",
                    },
                    {
                        "name": "Titled Associate Professor",
                        "active": True,
                        "position": "Насловен вонреден професор",
                    },
                    {
                        "name": "Titled Regular Professor",
                        "active": True,
                        "position": "Насловен редовен професор",
                    },
                ],
            },
        ),
        legacy=_response("https://staff.example.test", 503),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    assert await get_mentor_eligible_names() == frozenset({"Eligible Professor"})


@pytest.mark.anyio
async def test_list_professors_maps_directory_fields(monkeypatch) -> None:
    async def fake_directory() -> tuple[StaffMember, ...]:
        return (
            StaffMember(
                id=2,
                name="Александар Стојменски",
                active=True,
                title="д-р",
                position="Доцент",
                profile="https://finki.ukim.mk/kadar/aleksandar-stojmenski/",
                photo="https://example.test/photo.jpg",
            ),
        )

    monkeypatch.setattr(
        "app.api.professors.get_staff_directory",
        fake_directory,
    )

    professors = await list_professors()

    assert len(professors) == 1
    assert professors[0].name == "Александар Стојменски"
    assert professors[0].position == "Доцент"
    assert professors[0].mentor_eligible is True
    assert professors[0].profile is not None


@pytest.mark.anyio
async def test_list_professors_returns_503_when_directory_is_unavailable(
    monkeypatch,
) -> None:
    async def failing_directory() -> tuple[StaffMember, ...]:
        raise StaffDirectoryUnavailableError

    monkeypatch.setattr(
        "app.api.professors.get_staff_directory",
        failing_directory,
    )

    with pytest.raises(HTTPException) as exc_info:
        await list_professors()

    assert exc_info.value.status_code == 503
