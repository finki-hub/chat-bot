from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal, Protocol
from uuid import UUID

from asyncpg import Record
from asyncpg.pool import PoolConnectionProxy

from app.data.connection import Database

type QueryArgument = UUID | date | datetime | int
type RowValue = UUID | date | datetime | int | None


class SponsoredUsageConnection(Protocol):
    async def execute(self, query: str, *args: QueryArgument) -> str: ...

    async def fetchrow(
        self,
        query: str,
        *args: QueryArgument,
    ) -> Mapping[str, RowValue] | None: ...


type TransactionConnection = SponsoredUsageConnection | PoolConnectionProxy[Record]


class SponsoredUsageDatabase(Protocol):
    def transaction(self) -> AbstractAsyncContextManager[SponsoredUsageConnection]: ...


@dataclass(frozen=True, slots=True)
class SponsoredUsageSnapshot:
    user_id: UUID
    usage_date: date
    user_request_count: int
    global_request_count: int
    user_limit: int
    global_limit: int
    remaining_user_requests: int
    remaining_global_requests: int
    reset_at: datetime


@dataclass(frozen=True, slots=True)
class SponsoredUsageAdmission:
    request_id: UUID
    expires_at: datetime
    snapshot: SponsoredUsageSnapshot


class SponsoredUsageError(Exception):
    pass


class SponsoredRequestInProgressError(SponsoredUsageError):
    code: Literal["sponsored_request_in_progress"] = "sponsored_request_in_progress"

    def __init__(self, user_id: UUID) -> None:
        self.user_id = user_id

    def __str__(self) -> str:
        return f"sponsored request already in progress for user {self.user_id}"


class SponsoredQuotaExceededError(SponsoredUsageError):
    code: Literal["free_quota_exhausted"] = "free_quota_exhausted"

    def __init__(self, scope: Literal["user", "global"], reset_at: datetime) -> None:
        self.scope = scope
        self.reset_at = reset_at

    def __str__(self) -> str:
        return (
            f"sponsored {self.scope} quota exhausted until {self.reset_at.isoformat()}"
        )


_SNAPSHOT_SQL = "SELECT $1::date AS usage_date, COALESCE((SELECT request_count FROM sponsored_user_usage WHERE user_id = $2 AND usage_date = $1), 0) AS user_request_count, COALESCE((SELECT request_count FROM sponsored_global_usage WHERE usage_date = $1), 0) AS global_request_count"


async def get_sponsored_usage_snapshot(
    db: Database | SponsoredUsageDatabase,
    *,
    user_id: UUID,
    user_limit: int,
    global_limit: int,
    now: datetime | None = None,
) -> SponsoredUsageSnapshot:
    """Read advisory usage counters in a single database transaction."""
    current = _utc_now(now)
    usage_date, reset_at = _utc_day_bounds(current)
    async with db.transaction() as connection:
        return await _snapshot_in_transaction(
            connection,
            user_id=user_id,
            usage_date=usage_date,
            reset_at=reset_at,
            user_limit=user_limit,
            global_limit=global_limit,
        )


async def admit_sponsored_request(
    db: Database | SponsoredUsageDatabase,
    *,
    user_id: UUID,
    request_id: UUID,
    user_limit: int,
    global_limit: int,
    lease_ttl: timedelta,
    now: datetime | None = None,
) -> SponsoredUsageAdmission:
    """Atomically reserve a lease and consume both sponsored usage counters."""
    if lease_ttl <= timedelta(0):
        raise SponsoredUsageError
    current = _utc_now(now)
    usage_date, reset_at = _utc_day_bounds(current)
    expires_at = current + lease_ttl

    async with db.transaction() as connection:
        await connection.execute(
            "INSERT INTO sponsored_global_usage (usage_date, request_count) VALUES ($1, 0) ON CONFLICT (usage_date) DO NOTHING",
            usage_date,
        )
        global_row = await connection.fetchrow(
            "SELECT request_count FROM sponsored_global_usage WHERE usage_date = $1 FOR UPDATE",
            usage_date,
        )
        if global_row is None:
            raise SponsoredUsageError

        await connection.execute(
            "INSERT INTO sponsored_user_usage (user_id, usage_date, request_count) VALUES ($1, $2, 0) ON CONFLICT (user_id, usage_date) DO NOTHING",
            user_id,
            usage_date,
        )
        user_row = await connection.fetchrow(
            "SELECT request_count FROM sponsored_user_usage WHERE user_id = $1 AND usage_date = $2 FOR UPDATE",
            user_id,
            usage_date,
        )
        if user_row is None:
            raise SponsoredUsageError

        await connection.execute(
            "DELETE FROM sponsored_request_leases WHERE user_id = $1 AND expires_at <= $2",
            user_id,
            current,
        )
        lease_row = await connection.fetchrow(
            """
            INSERT INTO sponsored_request_leases (user_id, request_id, expires_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id) DO NOTHING
            RETURNING user_id
            """,
            user_id,
            request_id,
            expires_at,
        )
        if lease_row is None:
            raise SponsoredRequestInProgressError(user_id)

        global_counter = await connection.fetchrow(
            """
            UPDATE sponsored_global_usage
            SET request_count = request_count + 1
            WHERE usage_date = $1 AND request_count < $2
            RETURNING request_count
            """,
            usage_date,
            global_limit,
        )
        if global_counter is None:
            raise SponsoredQuotaExceededError("global", reset_at)

        user_counter = await connection.fetchrow(
            """
            UPDATE sponsored_user_usage
            SET request_count = request_count + 1
            WHERE user_id = $1 AND usage_date = $2 AND request_count < $3
            RETURNING request_count
            """,
            user_id,
            usage_date,
            user_limit,
        )
        if user_counter is None:
            raise SponsoredQuotaExceededError("user", reset_at)

        snapshot = await _snapshot_in_transaction(
            connection,
            user_id=user_id,
            usage_date=usage_date,
            reset_at=reset_at,
            user_limit=user_limit,
            global_limit=global_limit,
        )
        return SponsoredUsageAdmission(
            request_id=request_id,
            expires_at=expires_at,
            snapshot=snapshot,
        )


async def release_sponsored_request(
    db: Database | SponsoredUsageDatabase,
    *,
    user_id: UUID,
    request_id: UUID,
) -> None:
    """Release only the matching lease without refunding committed usage."""
    async with db.transaction() as connection:
        await connection.execute(
            """
            DELETE FROM sponsored_request_leases
            WHERE user_id = $1 AND request_id = $2
            """,
            user_id,
            request_id,
        )


async def _snapshot_in_transaction(
    connection: TransactionConnection,
    *,
    user_id: UUID,
    usage_date: date,
    reset_at: datetime,
    user_limit: int,
    global_limit: int,
) -> SponsoredUsageSnapshot:
    row = await connection.fetchrow(_SNAPSHOT_SQL, usage_date, user_id)
    if row is None:
        raise SponsoredUsageError
    user_request_count = _row_int(row, "user_request_count")
    global_request_count = _row_int(row, "global_request_count")
    return SponsoredUsageSnapshot(
        user_id=user_id,
        usage_date=usage_date,
        user_request_count=user_request_count,
        global_request_count=global_request_count,
        user_limit=user_limit,
        global_limit=global_limit,
        remaining_user_requests=max(user_limit - user_request_count, 0),
        remaining_global_requests=max(global_limit - global_request_count, 0),
        reset_at=reset_at,
    )


def _row_int(row: Mapping[str, RowValue] | Record, column: str) -> int:
    value = row[column]
    if not isinstance(value, int):
        raise SponsoredUsageError
    return value


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        raise SponsoredUsageError
    return now.astimezone(UTC)


def _utc_day_bounds(now: datetime) -> tuple[date, datetime]:
    usage_date = now.date()
    reset_at = datetime.combine(usage_date + timedelta(days=1), time.min, tzinfo=UTC)
    return usage_date, reset_at
