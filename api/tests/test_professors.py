import httpx
import pytest
from fastapi import HTTPException

from app.api.professors import list_professors
from app.data.staff import (
    StaffDirectoryUnavailableError,
    StaffMember,
    get_active_staff_names,
    get_mentor_eligible_names,
    get_staff_aliases,
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
    def __init__(self, response: httpx.Response) -> None:
        self.response = response

    async def get(self, url: str) -> httpx.Response:
        return self.response


@pytest.mark.anyio
async def test_staff_directory_reads_directus(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        _response(
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
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    staff = await get_staff_directory()

    assert [member.name for member in staff] == ["Directus Professor"]
    assert staff[0].id == 7
    assert staff[0].position == "Доцент"
    assert staff[0].mentor_eligible is True
    assert staff[0].photo == "https://example.test/photo.jpg"


@pytest.mark.anyio
async def test_staff_directory_raises_when_directus_fails(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        _response("https://data.example.test/items/staff", 503),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    with pytest.raises(StaffDirectoryUnavailableError):
        await get_staff_directory()


@pytest.mark.anyio
async def test_active_staff_names_use_directus_active_flag(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        _response(
            "https://data.example.test/items/staff",
            200,
            json={
                "data": [
                    {"name": "Active Professor", "active": True},
                    {"name": "Inactive Professor", "active": False},
                ],
            },
        ),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    assert await get_active_staff_names() == frozenset({"Active Professor"})


@pytest.mark.anyio
async def test_mentor_eligible_names_exclude_non_teaching_roles(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    client = FakeStaffClient(
        _response(
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
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    assert await get_mentor_eligible_names() == frozenset({"Eligible Professor"})


def test_non_finki_professor_is_not_mentor_eligible() -> None:
    member = StaffMember(
        name="External Professor",
        active=True,
        position="Редовен професор",
        finki=False,
    )

    assert member.mentor_eligible is False


class FakeRoutedStaffClient:
    def __init__(self, thesis_response: httpx.Response) -> None:
        self.thesis_response = thesis_response

    async def get(self, url: str) -> httpx.Response:
        if "bachelor_thesis_staff" in url:
            return self.thesis_response
        return _response(
            url,
            200,
            json={
                "data": [
                    {"id": 1, "name": "Ana Long Surname", "active": True},
                    {"id": 2, "name": "Same Name", "active": True},
                    {"id": 3, "name": "Shared One", "active": True},
                    {"id": 4, "name": "Shared Two", "active": True},
                ],
            },
        )


@pytest.mark.anyio
async def test_staff_aliases_resolve_only_unambiguous_active_links(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    monkeypatch.setattr("app.data.staff._staff_aliases_cache", None)
    client = FakeRoutedStaffClient(
        _response(
            "https://data.example.test/items/bachelor_thesis_staff",
            200,
            json={
                "data": [
                    {"name": "Ana Long", "staff": 1, "count": 9},
                    {"name": "Same Name", "staff": 2, "count": 4},
                    {"name": "Shared", "staff": 3, "count": 2},
                    {"name": "Shared", "staff": 4, "count": 1},
                    {"name": "Former", "staff": 99, "count": 1},
                    {"name": "Unlinked", "staff": None, "count": 1},
                ],
            },
        ),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    assert await get_staff_aliases() == {"Ana Long": "Ana Long Surname"}


@pytest.mark.anyio
async def test_staff_aliases_fall_back_when_links_fail(monkeypatch) -> None:
    monkeypatch.setattr("app.data.staff._active_staff_cache", None)
    monkeypatch.setattr("app.data.staff._staff_aliases_cache", None)
    client = FakeRoutedStaffClient(
        _response("https://data.example.test/items/bachelor_thesis_staff", 503),
    )
    monkeypatch.setattr("app.data.staff.get_http_client", lambda: client)

    assert await get_staff_aliases() == {}


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
