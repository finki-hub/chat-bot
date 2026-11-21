import time
from typing import Final

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from app.utils.http_client import get_http_client
from app.utils.settings import Settings


class StaffDirectoryUnavailableError(Exception):
    pass


# FINKI's diploma procedure allows a mentor who is a teacher elected to a
# teaching-scientific title at the Faculty. The confirmed mentor-eligible
# titles are regular professor, associate professor, and docent. Visiting
# and titled professors are not mentors; they may still serve on committees.
MENTOR_ELIGIBLE_POSITIONS: Final[frozenset[str]] = frozenset(
    {
        "Редовен професор",
        "Вонреден професор",
        "Доцент",
    },
)


def is_mentor_eligible(position: str | None) -> bool:
    return position is not None and position in MENTOR_ELIGIBLE_POSITIONS


class StaffMember(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int | None = None
    name: str
    active: bool
    title: str | None = None
    position: str | None = None
    cabinet: str | None = None
    profile: str | None = None
    consultations: str | None = None
    photo: str | None = None

    @property
    def mentor_eligible(self) -> bool:
        return is_mentor_eligible(self.position)

    @field_validator("active", mode="before")
    @classmethod
    def _parse_active(cls, value: object) -> bool:
        # Directus returns a boolean; the legacy export uses "0"/"1".
        return value in (True, 1, "1")


class _DirectusStaffResponse(BaseModel):
    data: list[StaffMember]


_LEGACY_ADAPTER: Final = TypeAdapter(list[StaffMember])
_DIRECTUS_ADAPTER: Final = TypeAdapter(_DirectusStaffResponse)

settings = Settings()

_DIRECTUS_FIELDS: Final = (
    "id,name,active,title,position,cabinet,profile,consultations,photo"
)

_active_staff_cache: tuple[float, tuple[StaffMember, ...]] | None = None


async def _fetch_directus_staff(
    client: httpx.AsyncClient,
) -> list[StaffMember] | None:
    url = (
        f"{settings.DIRECTUS_STAFF_URL}?limit=-1"
        "&filter[active][_eq]=true"
        f"&fields={_DIRECTUS_FIELDS}"
        "&sort=name"
    )
    try:
        response = await client.get(url)
        response.raise_for_status()
        return _DIRECTUS_ADAPTER.validate_json(response.content).data
    except httpx.HTTPError, ValidationError:
        return None


async def _fetch_legacy_staff(client: httpx.AsyncClient) -> list[StaffMember]:
    try:
        response = await client.get(settings.STAFF_API_URL)
        response.raise_for_status()
        return _LEGACY_ADAPTER.validate_json(response.content)
    except httpx.HTTPStatusError as exc:
        msg = "Staff directory returned an error status"
        raise StaffDirectoryUnavailableError(msg) from exc
    except httpx.RequestError as exc:
        msg = "Staff directory is unreachable"
        raise StaffDirectoryUnavailableError(msg) from exc
    except ValidationError as exc:
        msg = "Staff directory returned an invalid payload"
        raise StaffDirectoryUnavailableError(msg) from exc


async def get_staff_directory() -> tuple[StaffMember, ...]:
    global _active_staff_cache  # ruff: ignore[PLW0603] -- this function owns the process-local TTL cache

    now = time.monotonic()
    cached = _active_staff_cache
    if cached is not None and now - cached[0] < settings.STAFF_CACHE_TTL:
        return cached[1]

    client = get_http_client()
    staff = await _fetch_directus_staff(client)
    if staff is None or not any(member.active for member in staff):
        try:
            staff = await _fetch_legacy_staff(client)
        except StaffDirectoryUnavailableError:
            if cached is not None:
                return cached[1]
            raise

    active_staff = tuple(
        member.model_copy(update={"name": member.name.strip()})
        for member in staff
        if member.active and member.name.strip()
    )
    if not active_staff:
        if cached is not None:
            return cached[1]
        msg = "Staff directory returned no active staff"
        raise StaffDirectoryUnavailableError(msg)

    _active_staff_cache = (now, active_staff)
    return active_staff


async def get_active_staff_names() -> frozenset[str]:
    return frozenset(member.name for member in await get_staff_directory())


async def get_mentor_eligible_names() -> frozenset[str]:
    return frozenset(
        member.name for member in await get_staff_directory() if member.mentor_eligible
    )
