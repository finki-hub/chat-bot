from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest

from app.data.connection import Database
from app.data.links import create_link_query
from app.schemas.links import CreateLinkSchema

URL = "https://finki.ukim.mk/internacionalni-studenti/admissions/"


def test_create_link_passes_url_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a validated link, whose url is a pydantic HttpUrl.
    database = Database("postgresql://link-test")
    now = datetime.now(UTC)
    fetchrow = AsyncMock(
        return_value={
            "id": uuid4(),
            "name": "Admissions",
            "url": URL,
            "description": None,
            "user_id": None,
            "created_at": now,
            "updated_at": now,
        },
    )
    monkeypatch.setattr(database, "fetchrow", fetchrow)
    link = CreateLinkSchema.model_validate(
        {"name": "Admissions", "url": URL, "description": None, "user_id": None},
    )

    # When: the create data path runs.
    created = anyio.run(create_link_query, database, link)

    # Then: asyncpg gets a str, which it can encode (an HttpUrl raises TypeError).
    await_args = fetchrow.await_args
    assert await_args is not None
    assert await_args.args[2] == URL
    assert type(await_args.args[2]) is str
    assert created is not None
    assert str(created.url) == URL
