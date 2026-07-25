import json
from unittest.mock import AsyncMock

import anyio
import pytest

from app.data.connection import Database
from app.data.questions import create_question_query, update_question_query
from app.schemas.questions import CreateQuestionSchema, UpdateQuestionSchema


def test_create_question_serializes_validated_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a validated question containing a Discord profile link.
    database = Database("postgresql://question-links-test")
    fetchrow = AsyncMock(return_value=None)
    monkeypatch.setattr(database, "fetchrow", fetchrow)
    question = CreateQuestionSchema.model_validate(
        {
            "name": "Што е ФИНКИ Хаб",
            "content": "Започната од Delemangi.",
            "links": {
                "Discord профил": (
                    "https://discord.com/users/198249751001563136"
                ),
            },
        },
    )

    # When: the create data path prepares its database arguments.
    anyio.run(create_question_query, database, question)

    # Then: the link map reaches asyncpg as JSON containing string URLs.
    await_args = fetchrow.await_args
    assert await_args is not None
    links_json = await_args.args[4]
    assert json.loads(links_json) == {
        "Discord профил": "https://discord.com/users/198249751001563136",
    }


def test_update_question_serializes_validated_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a partial update containing a Discord profile link.
    database = Database("postgresql://question-links-test")
    fetchrow = AsyncMock(return_value=None)
    monkeypatch.setattr(database, "fetchrow", fetchrow)
    question = UpdateQuestionSchema.model_validate(
        {
            "links": {
                "Discord профил": (
                    "https://discord.com/users/198249751001563136"
                ),
            },
        },
    )

    # When: the update data path prepares its database arguments.
    anyio.run(update_question_query, database, "Што е ФИНКИ Хаб", question)

    # Then: the link map reaches asyncpg as JSON containing string URLs.
    await_args = fetchrow.await_args
    assert await_args is not None
    links_json = await_args.args[1]
    assert json.loads(links_json) == {
        "Discord профил": "https://discord.com/users/198249751001563136",
    }
