import anyio
import pytest

from app.api import chat_title
from app.llms.models import Model
from app.llms.provider_credentials import LlmProviderCredentials
from app.schemas.chat import ConversationTurn
from app.schemas.chat_title import ChatTitleSchema

QUESTION = "kade e studentska sluzba"


def _payload(model: Model | str) -> ChatTitleSchema:
    return ChatTitleSchema(
        user_id=None,
        messages=[ConversationTurn(role="user", content=QUESTION)],
        query_transform_model=model,
    )


def _transform_returning(output: str | None):
    async def fake_transform_query(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        credentials: LlmProviderCredentials | None = None,
    ) -> str:
        # None stands for a failed provider call, which hands the prompt back.
        return query if output is None else output

    return fake_transform_query


@pytest.mark.parametrize(
    "case",
    [
        (None, False, "no_model", None),
        (None, True, "model_failed", "qwen3:14b-q4_K_M"),
        ("  ", True, "empty_output", "qwen3:14b-q4_K_M"),
    ],
)
def test_a_title_that_falls_back_to_the_question_is_reported(
    monkeypatch,
    caplog,
    case,
):
    output, has_credential, reason, model = case
    events: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        chat_title,
        "capture",
        lambda distinct_id, event, properties: events.append(
            (distinct_id, event, properties),
        ),
    )
    monkeypatch.setattr(chat_title, "transform_query", _transform_returning(output))
    monkeypatch.setattr(
        chat_title,
        "has_provider_credential",
        lambda *args: has_credential,
    )

    with caplog.at_level("WARNING", logger=chat_title.__name__):
        response = anyio.run(
            chat_title.generate_chat_title,
            _payload("qwen3:14b-q4_K_M"),
        )

    assert response.title == QUESTION
    assert events == [
        (
            "chat-title",
            "chat_title_fallback",
            {"reason": reason, "model": model, "$process_person_profile": False},
        ),
    ]
    assert f"reason={reason}" in caplog.text
    # The question itself never reaches the log or the event.
    assert QUESTION not in caplog.text


def test_a_generated_title_is_not_reported(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        chat_title,
        "capture",
        lambda distinct_id, event, properties: events.append(event),
    )
    monkeypatch.setattr(
        chat_title,
        "transform_query",
        _transform_returning("Локација на студентската служба"),
    )
    monkeypatch.setattr(chat_title, "has_provider_credential", lambda *args: True)

    response = anyio.run(
        chat_title.generate_chat_title,
        _payload("qwen3:14b-q4_K_M"),
    )

    assert response.title == "Локација на студентската служба"
    assert events == []
