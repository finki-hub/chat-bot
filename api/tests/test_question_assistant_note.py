from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest
from pydantic import ValidationError

from app.data.connection import Database
from app.data.questions import (
    create_question_query,
    get_question_by_name_query,
    update_question_query,
)
from app.schemas.questions import CreateQuestionSchema, UpdateQuestionSchema

NOTE = "The statute counts only regular sessions, so both limits match."


def _row(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "name": "Exam attempts",
        "content": "A course can be taken at most four times.",
        "user_id": None,
        "links": None,
        "assistant_note": NOTE,
        "created_at": now,
        "updated_at": now,
        **overrides,
    }


@pytest.mark.parametrize(
    "schema",
    [CreateQuestionSchema, UpdateQuestionSchema],
)
def test_question_schemas_reject_discord_tokens_in_assistant_note(
    schema: type[CreateQuestionSchema | UpdateQuestionSchema],
) -> None:
    with pytest.raises(ValidationError, match="unresolved Discord token"):
        schema.model_validate(
            {
                "name": "Exam attempts",
                "content": "Four attempts.",
                "assistant_note": "Ask <@198249751001563136>.",
            },
        )


def test_question_schemas_default_assistant_note_to_none() -> None:
    created = CreateQuestionSchema.model_validate(
        {"name": "Exam attempts", "content": "Four attempts."},
    )
    update = UpdateQuestionSchema.model_validate({"content": "Four attempts."})

    assert created.assistant_note is None
    assert "assistant_note" not in update.model_dump(exclude_unset=True)


def test_create_question_persists_and_returns_assistant_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a new FAQ carrying internal guidance.
    database = Database("postgresql://question-note-test")
    fetchrow = AsyncMock(return_value=_row())
    monkeypatch.setattr(database, "fetchrow", fetchrow)
    question = CreateQuestionSchema.model_validate(
        {
            "name": "Exam attempts",
            "content": "A course can be taken at most four times.",
            "assistant_note": NOTE,
        },
    )

    # When: the create data path runs.
    created = anyio.run(create_question_query, database, question)

    # Then: the note is written as its own column and read back.
    await_args = fetchrow.await_args
    assert await_args is not None
    assert "assistant_note" in await_args.args[0]
    assert await_args.args[5] == NOTE
    assert created is not None
    assert created.assistant_note == NOTE


def test_update_question_can_clear_assistant_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an update that explicitly clears the note.
    database = Database("postgresql://question-note-test")
    fetchrow = AsyncMock(return_value=_row(assistant_note=None))
    monkeypatch.setattr(database, "fetchrow", fetchrow)
    question = UpdateQuestionSchema.model_validate({"assistant_note": None})

    # When: the update data path runs.
    updated = anyio.run(update_question_query, database, "Exam attempts", question)

    # Then: only the note column is set to NULL.
    await_args = fetchrow.await_args
    assert await_args is not None
    assert await_args.args[0].startswith("UPDATE question SET assistant_note = $1,")
    assert await_args.args[1:] == (None, "Exam attempts")
    assert updated is not None
    assert updated.assistant_note is None


def test_get_question_returns_assistant_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database("postgresql://question-note-test")
    monkeypatch.setattr(database, "fetchrow", AsyncMock(return_value=_row()))

    question = anyio.run(get_question_by_name_query, database, "Exam attempts")

    assert question is not None
    assert question.assistant_note == NOTE
