import logging
import time
from collections.abc import Mapping
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

logger = logging.getLogger(__name__)


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
    finki: bool = True

    @property
    def mentor_eligible(self) -> bool:
        return self.finki and is_mentor_eligible(self.position)

    @field_validator("active", mode="before")
    @classmethod
    def _parse_active(cls, value: object) -> bool:
        # Directus returns a boolean; accept the legacy "0"/"1" shape too.
        return value in (True, 1, "1")


class _DirectusStaffResponse(BaseModel):
    data: list[StaffMember]


class _ThesisStaffName(BaseModel):
    name: str
    staff: int | None


class _DirectusThesisStaffResponse(BaseModel):
    data: list[_ThesisStaffName]


_DIRECTUS_ADAPTER: Final = TypeAdapter(_DirectusStaffResponse)
_THESIS_STAFF_ADAPTER: Final = TypeAdapter(_DirectusThesisStaffResponse)

settings = Settings()

_DIRECTUS_FIELDS: Final = (
    "id,name,active,title,position,cabinet,profile,consultations,photo,finki"
)

_active_staff_cache: tuple[float, tuple[StaffMember, ...]] | None = None
_staff_aliases_cache: tuple[float, dict[str, str]] | None = None


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


async def get_staff_directory() -> tuple[StaffMember, ...]:
    global _active_staff_cache  # ruff: ignore[PLW0603] -- this function owns the process-local TTL cache

    now = time.monotonic()
    cached = _active_staff_cache
    if cached is not None and now - cached[0] < settings.STAFF_CACHE_TTL:
        return cached[1]

    client = get_http_client()
    staff = await _fetch_directus_staff(client)
    if staff is None:
        if cached is not None:
            return cached[1]
        msg = "Staff directory is unavailable"
        raise StaffDirectoryUnavailableError(msg)

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


async def _fetch_thesis_staff_names(
    client: httpx.AsyncClient,
) -> list[_ThesisStaffName] | None:
    url = (
        f"{settings.DIRECTUS_THESIS_STAFF_URL}?limit=-1"
        "&aggregate[count]=id&groupBy[]=name&groupBy[]=staff"
    )
    try:
        response = await client.get(url)
        response.raise_for_status()
        return _THESIS_STAFF_ADAPTER.validate_json(response.content).data
    except httpx.HTTPError, ValidationError:
        return None


def _build_staff_aliases(
    links: list[_ThesisStaffName],
    staff: tuple[StaffMember, ...],
) -> dict[str, str]:
    current_names = {member.id: member.name for member in staff}
    resolved: dict[str, set[str]] = {}
    for link in links:
        current = current_names.get(link.staff)
        if current is not None:
            resolved.setdefault(link.name.strip(), set()).add(current)
    return {
        written: next(iter(names))
        for written, names in resolved.items()
        if len(names) == 1 and written not in names
    }


async def get_staff_aliases() -> Mapping[str, str]:
    """Map names written on theses to current directory names, best effort."""
    global _staff_aliases_cache  # ruff: ignore[PLW0603] -- this function owns the process-local TTL cache

    now = time.monotonic()
    cached = _staff_aliases_cache
    if cached is not None and now - cached[0] < settings.STAFF_CACHE_TTL:
        return cached[1]

    links = await _fetch_thesis_staff_names(get_http_client())
    if links is None:
        logger.warning("Staff name links are unavailable; using written names")
        return cached[1] if cached is not None else {}

    aliases = _build_staff_aliases(links, await get_staff_directory())
    _staff_aliases_cache = (now, aliases)
    return aliases
