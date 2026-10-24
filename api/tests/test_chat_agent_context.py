from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse

from app.api import chat as chat_api
from app.data.connection import Database
from app.llms.models import Model
from app.schemas.chat import ChatSchema
from tests.chat_models_access_support import credentials, settings

USER_ID = UUID("00000000-0000-4000-8000-000000000002")


def _payload() -> ChatSchema:
    return ChatSchema.model_validate(
        {
            "user_id": USER_ID,
            "interface": "web",
            "inference_model": Model.GPT_6_LUNA,
            "messages": [{"role": "user", "content": "Question"}],
        },
    )


@pytest.mark.anyio
async def test_chat_points_agent_to_search_without_pre_retrieval(
    monkeypatch,
) -> None:
    contexts: list[str] = []
    retrieval_credentials: list[object] = []
    resolved = credentials(openai=True)

    async def resolve_credentials(*args, **kwargs):
        return resolved

    async def answer(payload, context, **kwargs):
        contexts.append(context)
        retrieval_credentials.append(kwargs["retrieval_credentials"])

        async def body():
            yield 'event: token\ndata: {"text": "ok"}\n\n'
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(body())

    monkeypatch.setattr(chat_api, "resolve_provider_credentials", resolve_credentials)
    monkeypatch.setattr(chat_api, "handle_chat", answer)
    monkeypatch.setattr(chat_api, "capture", lambda *args, **kwargs: None)
    request = cast(
        "Request",
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(settings=settings())),
            headers={"X-Distinct-Id": "test-distinct-id"},
        ),
    )

    rendered = "".join(
        [
            str(chunk)
            async for chunk in chat_api._chat_response_stream(  # noqa: SLF001
                _payload(),
                request,
                Database.__new__(Database),
                uuid4(),
            )
        ],
    )

    assert '"text": "ok"' in rendered
    assert "event: sources" not in rendered
    assert "event: status" not in rendered
    [context] = contexts
    assert context.startswith("Денешен датум: ")
    assert retrieval_credentials == [resolved]
