import time
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from app.utils.http_client import get_http_client
from app.utils.settings import Settings


class StaffDirectoryUnavailableError(Exception):
    pass


class StaffMember(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    active: Literal["0", "1"]


_STAFF_ADAPTER: Final = TypeAdapter(list[StaffMember])

settings = Settings()

_active_staff_cache: tuple[float, frozenset[str]] | None = None


async def get_active_staff_names() -> frozenset[str]:
    global _active_staff_cache  # noqa: PLW0603

    now = time.monotonic()
    cached = _active_staff_cache
    if cached is not None and now - cached[0] < settings.STAFF_CACHE_TTL:
        return cached[1]

    client = get_http_client()
    try:
        response = await client.get(settings.STAFF_API_URL)
        response.raise_for_status()
        staff = _STAFF_ADAPTER.validate_json(response.content)
    except httpx.HTTPStatusError as exc:
        if cached is not None:
            return cached[1]
        msg = "Staff directory returned an error status"
        raise StaffDirectoryUnavailableError(msg) from exc
    except httpx.RequestError as exc:
        if cached is not None:
            return cached[1]
        msg = "Staff directory is unreachable"
        raise StaffDirectoryUnavailableError(msg) from exc
    except ValidationError as exc:
        if cached is not None:
            return cached[1]
        msg = "Staff directory returned an invalid payload"
        raise StaffDirectoryUnavailableError(msg) from exc

    active_names = frozenset(
        member.name.strip()
        for member in staff
        if member.active == "1" and member.name.strip()
    )
    _active_staff_cache = (now, active_names)
    return active_names
