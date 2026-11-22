import json
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.data.connection import Database
from app.sync_professor_papers import (
    PaperRefreshError,
    Profile,
    build_refresh_plan,
    refresh_papers,
)

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL paper refresh tests",
)
PROFILES = {"A1": Profile("A1", "Марко Петров")}


def _work(number: int) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/W{number}",
        "doi": f"https://doi.org/10.1000/{number}",
        "title": f"Paper {number}",
        "publication_year": 2026,
        "type": "article",
        "is_retracted": False,
        "abstract_inverted_index": {"Abstract": [0]},
        "topics": [],
        "authorships": [
            {
                "author": {"id": "https://openalex.org/A1"},
                "raw_affiliation_strings": [
                    "Faculty of Computer Science and Engineering"
                ],
                "raw_author_name": "Marko Petrov",
            },
        ],
    }


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database(os.environ["TEST_DATABASE_URL"], min_size=1, max_size=4)
    await db.init()
    await db.run_migrations()
    await db.execute("TRUNCATE professor_document CASCADE")
    await db.execute(
        "INSERT INTO professor_document "
        "(external_id, title, canonical_authors, sources) "
        "VALUES ('10.1000/1', 'Paper 1', $1::jsonb, $2::jsonb)",
        json.dumps(["Постоечки Автор"], ensure_ascii=False),
        json.dumps(["ukim_repo"]),
    )
    try:
        yield db
    finally:
        await db.execute("TRUNCATE professor_document CASCADE")
        await db.disconnect()


@pytest.mark.anyio
async def test_dry_run_writes_nothing_and_apply_writes_the_reviewed_plan(
    database,
) -> None:
    works = [_work(1), _work(2)]

    plan, _ = await refresh_papers(database, works, PROFILES, {}, expected_plan=None)

    assert await database.fetchval("SELECT count(*) FROM professor_document") == 1

    await refresh_papers(
        database,
        works,
        PROFILES,
        {},
        expected_plan=plan.fingerprint,
    )

    assert await database.fetchval("SELECT count(*) FROM professor_document") == 2
    authors = await database.fetchval(
        "SELECT canonical_authors FROM professor_document WHERE external_id = '10.1000/1'",
    )
    assert json.loads(authors) == ["Постоечки Автор", "Марко Петров"]

    second, _ = await refresh_papers(database, works, PROFILES, {}, expected_plan=None)
    assert second.new_papers == ()
    assert second.updates == ()


@pytest.mark.anyio
async def test_apply_refuses_a_plan_that_changed(database) -> None:
    reviewed = build_refresh_plan([_work(1)], PROFILES, [], {})
    changed_works = [_work(1), _work(2)]

    with pytest.raises(PaperRefreshError, match="changed since it was reviewed"):
        await refresh_papers(
            database,
            changed_works,
            PROFILES,
            {},
            expected_plan=reviewed.fingerprint,
        )

    assert await database.fetchval("SELECT count(*) FROM professor_document") == 1
