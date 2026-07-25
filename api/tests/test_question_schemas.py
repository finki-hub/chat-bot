import pytest
from pydantic import ValidationError

from app.schemas.questions import CreateQuestionSchema, UpdateQuestionSchema


@pytest.mark.parametrize(
    "token",
    [
        "<@198249751001563136>",
        "<@!198249751001563136>",
        "<@&942470742208049212>",
        "<#942470742208049212>",
    ],
)
def test_create_question_rejects_unresolved_discord_tokens(token: str) -> None:
    # Given: FAQ content containing a raw Discord platform token.
    payload = {
        "name": "Што е ФИНКИ Хаб",
        "content": f"ФИНКИ Хаб е започнат од {token}.",
    }

    # When/Then: parsing the API request rejects the user-facing token.
    with pytest.raises(ValidationError, match="unresolved Discord token"):
        CreateQuestionSchema.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Основач <@198249751001563136>"),
        (
            "links",
            {"Discord <#942470742208049212>": "https://discord.com/channels/1/2"},
        ),
    ],
)
def test_create_question_rejects_discord_tokens_in_labels(
    field: str,
    value: str | dict[str, str],
) -> None:
    # Given: a raw Discord token in a user-visible FAQ label.
    payload: dict[str, str | dict[str, str]] = {
        "name": "Што е ФИНКИ Хаб",
        "content": "Независна студентска иницијатива.",
        field: value,
    }

    # When/Then: parsing the API request rejects the label.
    with pytest.raises(ValidationError, match="unresolved Discord token"):
        CreateQuestionSchema.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Основач <@198249751001563136>"),
        ("content", "Започната од <@198249751001563136>."),
        (
            "links",
            {"Discord <#942470742208049212>": "https://discord.com/channels/1/2"},
        ),
    ],
)
def test_update_question_rejects_unresolved_discord_tokens(
    field: str,
    value: str | dict[str, str],
) -> None:
    # Given: an update containing a raw Discord token in a user-facing field.
    payload = {field: value}

    # When/Then: parsing the partial update rejects it.
    with pytest.raises(ValidationError, match="unresolved Discord token"):
        UpdateQuestionSchema.model_validate(payload)


def test_create_question_accepts_readable_discord_profile_link() -> None:
    # Given: a readable attribution that preserves the Discord ID in a URL.
    payload = {
        "name": "Што е ФИНКИ Хаб",
        "content": (
            "Започната од "
            "[Delemangi](https://discord.com/users/198249751001563136)."
        ),
        "user_id": "198249751001563136",
        "links": {
            "Discord профил": "https://discord.com/users/198249751001563136",
        },
    }

    # When: the API request is parsed.
    question = CreateQuestionSchema.model_validate(payload)

    # Then: readable content, profile URL, and author metadata remain valid.
    assert question.content == payload["content"]
    assert question.user_id == payload["user_id"]
    assert question.model_dump(mode="json")["links"] == payload["links"]
