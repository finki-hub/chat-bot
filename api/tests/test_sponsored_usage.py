from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import anyio
import pytest

from app.data.sponsored_usage import (
    QueryArgument,
    RowValue,
    SponsoredUsageAdmission,
    SponsoredUsageError,
    SponsoredUsageSnapshot,
    admit_sponsored_request,
    get_sponsored_usage_snapshot,
    release_sponsored_request,
)

USER_ID = UUID("11111111-1111-1111-1111-111111111111")
REQUEST_ID = UUID("22222222-2222-2222-2222-222222222222")
NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)


class ScriptedFailureError(Exception):
    pass


class ScriptedConnection:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on

    async def execute(self, query: str, *args: QueryArgument) -> str:
        del args
        self.calls.append(query)
        if self.fail_on is not None and self.fail_on in query:
            raise ScriptedFailureError
        return "EXECUTE"

    async def fetchrow(
        self,
        query: str,
        *args: QueryArgument,
    ) -> dict[str, RowValue] | None:
        del args
        self.calls.append(query)
        if self.fail_on is not None and self.fail_on in query:
            raise ScriptedFailureError
        if "AS usage_date" in query and "COALESCE" in query:
            return {
                "usage_date": date(2026, 7, 18),
                "user_request_count": 2,
                "global_request_count": 3,
            }
        if "FROM sponsored_global_usage" in query:
            return {"request_count": 2}
        if "FROM sponsored_user_usage" in query:
            return {"request_count": 1}
        if "INSERT INTO sponsored_request_leases" in query:
            return {"user_id": USER_ID}
        if "UPDATE sponsored_global_usage" in query:
            return {"request_count": 3}
        if "UPDATE sponsored_user_usage" in query:
            return {"request_count": 2}
        return None


class ScriptedDatabase:
    def __init__(self, connection: ScriptedConnection) -> None:
        self.connection = connection
        self.committed = False
        self.rolled_back = False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[ScriptedConnection]:
        try:
            yield self.connection
        except (
            ScriptedFailureError,
            SponsoredUsageError,
        ):
            self.rolled_back = True
            raise
        else:
            self.committed = True


def test_admission_locks_global_before_user_and_commits_snapshot() -> None:
    async def run() -> tuple[ScriptedDatabase, SponsoredUsageAdmission]:
        connection = ScriptedConnection()
        database = ScriptedDatabase(connection)
        admission = await admit_sponsored_request(
            database,
            user_id=USER_ID,
            request_id=REQUEST_ID,
            user_limit=5,
            global_limit=100,
            lease_ttl=timedelta(minutes=10),
            now=NOW,
        )
        return database, admission

    database, admission = anyio.run(run)

    assert admission.snapshot.user_request_count == 2
    assert admission.snapshot.global_request_count == 3
    assert admission.snapshot.remaining_user_requests == 3
    assert admission.expires_at == NOW + timedelta(minutes=10)
    assert database.committed
    assert not database.rolled_back
    global_lock = next(
        index
        for index, query in enumerate(database.connection.calls)
        if "FROM sponsored_global_usage" in query
    )
    user_lock = next(
        index
        for index, query in enumerate(database.connection.calls)
        if "FROM sponsored_user_usage" in query
    )
    assert global_lock < user_lock


def test_admission_rolls_back_lease_and_global_counter_when_user_update_fails() -> None:
    async def run() -> ScriptedDatabase:
        connection = ScriptedConnection(fail_on="UPDATE sponsored_user_usage")
        database = ScriptedDatabase(connection)
        with pytest.raises(ScriptedFailureError):
            await admit_sponsored_request(
                database,
                user_id=USER_ID,
                request_id=REQUEST_ID,
                user_limit=5,
                global_limit=100,
                lease_ttl=timedelta(minutes=10),
                now=NOW,
            )
        return database

    database = anyio.run(run)

    assert database.rolled_back
    assert not database.committed
    assert any(
        "INSERT INTO sponsored_request_leases" in q for q in database.connection.calls
    )
    assert any("UPDATE sponsored_global_usage" in q for q in database.connection.calls)


def test_snapshot_and_release_use_transaction_boundaries() -> None:
    async def run() -> tuple[
        ScriptedDatabase,
        ScriptedDatabase,
        SponsoredUsageSnapshot,
    ]:
        snapshot_database = ScriptedDatabase(ScriptedConnection())
        snapshot = await get_sponsored_usage_snapshot(
            snapshot_database,
            user_id=USER_ID,
            user_limit=5,
            global_limit=100,
            now=NOW,
        )
        release_database = ScriptedDatabase(ScriptedConnection())
        await release_sponsored_request(
            release_database,
            user_id=USER_ID,
            request_id=REQUEST_ID,
        )
        return snapshot_database, release_database, snapshot

    snapshot_database, release_database, snapshot = anyio.run(run)

    assert snapshot.user_id == USER_ID
    assert snapshot.usage_date == date(2026, 7, 18)
    assert snapshot_database.committed
    assert release_database.committed
    assert (
        "DELETE FROM sponsored_request_leases" in release_database.connection.calls[0]
    )


def test_new_sponsored_usage_migration_is_additive_and_idempotent() -> None:
    migration = Path("resources/migrations/0008_sponsored_luna_usage.sql").read_text()

    assert "CREATE TABLE IF NOT EXISTS sponsored_user_usage" in migration
    assert "CREATE TABLE IF NOT EXISTS sponsored_global_usage" in migration
    assert "CREATE TABLE IF NOT EXISTS sponsored_request_leases" in migration
    assert "PRIMARY KEY (user_id, usage_date)" in migration
    assert "PRIMARY KEY (usage_date)" in migration
    assert "PRIMARY KEY (user_id)" in migration
    assert "request_count >= 0" in migration
