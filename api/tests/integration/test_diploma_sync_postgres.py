import os
from collections.abc import AsyncIterator

import pytest

from app.data.connection import Database
from app.services import diploma_sync
from app.services.diploma_sync import DiplomaShrinkError, sync_defended_diplomas

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL diploma sync tests",
)


def _record(number: int, **overrides: object) -> dict[str, object]:
    return {
        "fileId": f"file-{number}",
        "student": f"{number} - Student",
        "title": f"Title {number}",
        "description": "Abstract",
        "mentor": "Mentor",
        "member1": "One",
        "member2": "Two",
        "status": "Одбрана",
        "dateOfSubmission": "21.10.2026",
    } | overrides


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database(os.environ["TEST_DATABASE_URL"], min_size=1, max_size=4)
    await db.init()
    await db.run_migrations()
    await db.execute("TRUNCATE diploma CASCADE")
    try:
        yield db
    finally:
        await db.execute("TRUNCATE diploma CASCADE")
        await db.disconnect()


def _serve(monkeypatch, payload: list[dict[str, object]]) -> None:
    async def fetch() -> object:
        return payload

    monkeypatch.setattr(diploma_sync, "_fetch_diploma_payload", fetch)


async def _state(db: Database) -> list[tuple[str, object, int]]:
    rows = await db.fetch(
        "SELECT external_id, updated_at, embedding_revision FROM diploma "
        "ORDER BY external_id",
    )
    return [
        (row["external_id"], row["updated_at"], row["embedding_revision"])
        for row in rows
    ]


@pytest.mark.anyio
async def test_sync_writes_only_new_or_changed_rows(monkeypatch, database) -> None:
    records = [_record(1), _record(2)]
    _serve(monkeypatch, records)
    first = await sync_defended_diplomas(database)
    before = await _state(database)

    second = await sync_defended_diplomas(database)

    assert (first.inserted, first.updated) == (2, 0)
    assert (second.inserted, second.updated) == (0, 0)
    assert await _state(database) == before

    _serve(monkeypatch, [_record(1, mentor="New Mentor"), _record(2), _record(3)])
    third = await sync_defended_diplomas(database)

    assert (third.inserted, third.updated) == (1, 1)
    assert await database.fetchval("SELECT count(*) FROM diploma") == 3
    assert (
        await database.fetchval(
            "SELECT count(*) FROM diploma WHERE mentor = 'New Mentor'",
        )
        == 1
    )


@pytest.mark.anyio
async def test_sync_refuses_a_large_shrink_without_writing(
    monkeypatch,
    database,
) -> None:
    _serve(monkeypatch, [_record(number) for number in range(40)])
    await sync_defended_diplomas(database)
    before = await _state(database)

    _serve(
        monkeypatch,
        [_record(0, mentor="Must Not Persist"), *[_record(n) for n in range(1, 30)]],
    )
    with pytest.raises(DiplomaShrinkError):
        await sync_defended_diplomas(database)

    assert await _state(database) == before

    accepted = await sync_defended_diplomas(database, accept_fewer=True)

    assert (accepted.inserted, accepted.updated) == (0, 1)
    assert await database.fetchval("SELECT count(*) FROM diploma") == 40


@pytest.mark.anyio
async def test_sync_tolerates_a_small_gap_from_edited_titles(
    monkeypatch,
    database,
) -> None:
    _serve(monkeypatch, [_record(number) for number in range(40)])
    await sync_defended_diplomas(database)

    _serve(monkeypatch, [_record(number) for number in range(39)])
    result = await sync_defended_diplomas(database)

    assert (result.inserted, result.updated) == (0, 0)
