import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

import httpx

from app.data.connection import Database
from app.data.diplomas import (
    DiplomaUpsert,
    count_diplomas,
    upsert_changed_diplomas,
)
from app.utils.http_client import get_http_client
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

_DIPLOMAS_TIMEOUT: Final = httpx.Timeout(timeout=60.0)
_DEFENDED_STATUS: Final = "Одбрана"
_SYNC_LOCK_KEY: Final = 7_241_806_133
# An edited upstream title gets a new key and leaves its old row behind, so a
# small gap against the stored count is normal.
_MAX_SHRINK_FRACTION: Final = 0.05


class DiplomaSyncError(Exception):
    pass


class DiplomaUpstreamError(DiplomaSyncError):
    pass


class InvalidDiplomaDataError(DiplomaSyncError):
    pass


class DiplomaShrinkError(DiplomaSyncError):
    pass


@dataclass(frozen=True, slots=True)
class DiplomaSyncResult:
    fetched: int
    defended: int
    inserted: int
    updated: int


def _compute_external_id(record: dict[str, object]) -> str:
    raw = (
        f"{record.get('fileId')}|{record.get('student')}"
        f"|{record.get('title')}|{record.get('dateOfSubmission')}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_submission_date(value: object, position: int) -> date | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        msg = f"Diploma record {position} has an invalid submission date"
        raise InvalidDiplomaDataError(msg)
    try:
        # ruff: ignore[DTZ007] -- the source is a calendar date with no time zone
        return datetime.strptime(value.strip(), "%d.%m.%Y").date()
    except ValueError as exc:
        msg = f"Diploma record {position} has an invalid submission date"
        raise InvalidDiplomaDataError(msg) from exc


def _required_text(
    record: dict[str, object],
    field: str,
    position: int,
    *,
    allow_blank: bool = False,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        msg = f"Diploma record {position} has an invalid {field}"
        raise InvalidDiplomaDataError(msg)
    return value


def _optional_text(
    record: dict[str, object],
    field: str,
    position: int,
) -> str | None:
    value = record.get(field)
    if value is not None and not isinstance(value, str):
        msg = f"Diploma record {position} has an invalid {field}"
        raise InvalidDiplomaDataError(msg)
    return value


def parse_defended_diplomas(payload: object) -> tuple[int, list[DiplomaUpsert]]:
    """Validate the whole upstream payload and return its defended diplomas."""
    if not isinstance(payload, list) or not payload:
        msg = "Diplomas API returned an empty or unexpected list"
        raise InvalidDiplomaDataError(msg)

    diplomas: dict[str, DiplomaUpsert] = {}
    for position, record in enumerate(payload):
        if not isinstance(record, dict):
            msg = f"Diploma record {position} is not an object"
            raise InvalidDiplomaDataError(msg)
        if record.get("status") != _DEFENDED_STATUS:
            continue
        external_id = _compute_external_id(record)
        diplomas[external_id] = DiplomaUpsert(
            external_id=external_id,
            title=_required_text(record, "title", position),
            description=_required_text(
                record,
                "description",
                position,
                allow_blank=True,
            ),
            mentor=_required_text(record, "mentor", position),
            member1=_optional_text(record, "member1", position),
            member2=_optional_text(record, "member2", position),
            status=_DEFENDED_STATUS,
            date_of_submission=_parse_submission_date(
                record.get("dateOfSubmission"),
                position,
            ),
        )

    if not diplomas:
        msg = "Diplomas API returned no defended diplomas"
        raise InvalidDiplomaDataError(msg)
    return len(payload), list(diplomas.values())


async def _fetch_diploma_payload() -> object:
    try:
        response = await get_http_client().get(
            settings.DIPLOMAS_API_URL,
            timeout=_DIPLOMAS_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        msg = f"Diplomas API returned status {exc.response.status_code}"
        raise DiplomaUpstreamError(msg) from exc
    except httpx.RequestError as exc:
        msg = "Failed to reach the Diplomas API"
        raise DiplomaUpstreamError(msg) from exc
    except ValueError as exc:
        msg = "Diplomas API returned invalid JSON"
        raise InvalidDiplomaDataError(msg) from exc


async def sync_defended_diplomas(
    db: Database,
    *,
    accept_fewer: bool = False,
) -> DiplomaSyncResult:
    """Atomically apply a validated upstream snapshot without deleting rows."""
    fetched, diplomas = parse_defended_diplomas(await _fetch_diploma_payload())

    async with db.transaction() as connection:
        await connection.execute("SELECT pg_advisory_xact_lock($1)", _SYNC_LOCK_KEY)
        stored = await count_diplomas(connection)
        if not accept_fewer and len(diplomas) < stored * (1 - _MAX_SHRINK_FRACTION):
            msg = (
                f"Diplomas API returned {len(diplomas)} defended diplomas, "
                f"far fewer than the {stored} stored"
            )
            raise DiplomaShrinkError(msg)
        inserted, updated = await upsert_changed_diplomas(connection, diplomas)

    result = DiplomaSyncResult(fetched, len(diplomas), inserted, updated)
    logger.info(
        "Diploma sync complete: fetched=%d defended=%d inserted=%d updated=%d",
        result.fetched,
        result.defended,
        result.inserted,
        result.updated,
    )
    return result
