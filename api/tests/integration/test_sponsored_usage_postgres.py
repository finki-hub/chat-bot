import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import anyio
import pytest

from app.data.connection import Database
from app.data.sponsored_usage import (
    SponsoredQuotaExceededError,
    SponsoredRequestInProgressError,
    SponsoredUsageError,
    admit_sponsored_request,
    release_sponsored_request,
)

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL sponsored usage tests",
)


async def _database() -> AsyncIterator[Database]:
    if DATABASE_URL is None:
        pytest.skip(
            "set TEST_DATABASE_URL to run real-PostgreSQL sponsored usage tests",
        )
    database = Database(DATABASE_URL, min_size=1, max_size=20)
    await database.init()
    await database.run_migrations()
    await database.execute(
        """
        TRUNCATE sponsored_user_usage,
                 sponsored_global_usage,
                 sponsored_request_leases
        """,
    )
    try:
        yield database
    finally:
        await database.disconnect()


def test_real_postgres_allows_five_admissions_and_rolls_back_the_sixth() -> None:
    async def run() -> None:
        async for database in _database():
            user_id = uuid4()
            now = datetime(2026, 7, 18, 12, tzinfo=UTC)
            for _ in range(5):
                admission = await admit_sponsored_request(
                    database,
                    user_id=user_id,
                    request_id=uuid4(),
                    user_limit=5,
                    global_limit=100,
                    lease_ttl=timedelta(minutes=10),
                    now=now,
                )
                await release_sponsored_request(
                    database,
                    user_id=user_id,
                    request_id=admission.request_id,
                )

            with pytest.raises(SponsoredQuotaExceededError) as error:
                await admit_sponsored_request(
                    database,
                    user_id=user_id,
                    request_id=uuid4(),
                    user_limit=5,
                    global_limit=100,
                    lease_ttl=timedelta(minutes=10),
                    now=now,
                )

            assert error.value.scope == "user"
            count = await database.fetchval(
                """
                SELECT request_count
                FROM sponsored_user_usage
                WHERE user_id = $1 AND usage_date = $2
                """,
                user_id,
                now.date(),
            )
            lease_count = await database.fetchval(
                """
                SELECT COUNT(*)
                FROM sponsored_request_leases
                WHERE user_id = $1
                """,
                user_id,
            )
            assert count == 5
            assert lease_count == 0

    anyio.run(run)


def test_real_postgres_parallel_tasks_never_exceed_global_limit() -> None:
    async def run() -> None:
        async for database in _database():
            now = datetime(2026, 7, 18, 12, tzinfo=UTC)
            outcomes: list[bool] = []

            async def attempt(
                database: Database,
                now: datetime,
                outcomes: list[bool],
                user_id: UUID,
            ) -> None:
                try:
                    admission = await admit_sponsored_request(
                        database,
                        user_id=user_id,
                        request_id=uuid4(),
                        user_limit=5,
                        global_limit=5,
                        lease_ttl=timedelta(minutes=10),
                        now=now,
                    )
                except SponsoredUsageError:
                    outcomes.append(False)
                else:
                    outcomes.append(True)
                    await release_sponsored_request(
                        database,
                        user_id=user_id,
                        request_id=admission.request_id,
                    )

            async with anyio.create_task_group() as task_group:
                for _ in range(20):
                    task_group.start_soon(attempt, database, now, outcomes, uuid4())

            assert sum(outcomes) == 5
            count = await database.fetchval(
                "SELECT request_count FROM sponsored_global_usage WHERE usage_date = $1",
                now.date(),
            )
            assert count == 5

    anyio.run(run)


def test_real_postgres_lease_blocks_active_request_and_expires_after_crash() -> None:
    async def run() -> None:
        async for database in _database():
            user_id = uuid4()
            start = datetime(2026, 7, 18, 12, tzinfo=UTC)
            first = await admit_sponsored_request(
                database,
                user_id=user_id,
                request_id=uuid4(),
                user_limit=5,
                global_limit=100,
                lease_ttl=timedelta(seconds=10),
                now=start,
            )

            with pytest.raises(SponsoredRequestInProgressError):
                await admit_sponsored_request(
                    database,
                    user_id=user_id,
                    request_id=uuid4(),
                    user_limit=5,
                    global_limit=100,
                    lease_ttl=timedelta(seconds=10),
                    now=start + timedelta(seconds=1),
                )

            second = await admit_sponsored_request(
                database,
                user_id=user_id,
                request_id=uuid4(),
                user_limit=5,
                global_limit=100,
                lease_ttl=timedelta(seconds=10),
                now=start + timedelta(seconds=11),
            )
            assert second.snapshot.user_request_count == 2
            await release_sponsored_request(
                database,
                user_id=user_id,
                request_id=second.request_id,
            )
            assert first.snapshot.user_request_count == 1

    anyio.run(run)


def test_real_postgres_global_limit_failure_rolls_back_user_and_lease() -> None:
    async def run() -> None:
        async for database in _database():
            now = datetime(2026, 7, 18, 12, tzinfo=UTC)
            first_user = uuid4()
            await admit_sponsored_request(
                database,
                user_id=first_user,
                request_id=uuid4(),
                user_limit=5,
                global_limit=1,
                lease_ttl=timedelta(minutes=10),
                now=now,
            )

            second_user = uuid4()
            with pytest.raises(SponsoredQuotaExceededError) as error:
                await admit_sponsored_request(
                    database,
                    user_id=second_user,
                    request_id=uuid4(),
                    user_limit=5,
                    global_limit=1,
                    lease_ttl=timedelta(minutes=10),
                    now=now,
                )

            assert error.value.scope == "global"
            user_row = await database.fetchrow(
                """
                SELECT request_count
                FROM sponsored_user_usage
                WHERE user_id = $1 AND usage_date = $2
                """,
                second_user,
                now.date(),
            )
            lease_count = await database.fetchval(
                "SELECT COUNT(*) FROM sponsored_request_leases WHERE user_id = $1",
                second_user,
            )
            global_count = await database.fetchval(
                "SELECT request_count FROM sponsored_global_usage WHERE usage_date = $1",
                now.date(),
            )
            assert user_row is None
            assert lease_count == 0
            assert global_count == 1

    anyio.run(run)
