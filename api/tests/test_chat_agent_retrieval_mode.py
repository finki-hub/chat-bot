from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
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
async def test_agent_mode_skips_pre_retrieval_and_points_agent_to_search(
    monkeypatch,
) -> None:
    contexts: list[str] = []

    async def resolve_credentials(*args, **kwargs):
        return credentials(openai=True)

    async def fail_if_retrieval_runs(*args, **kwargs):
        raise AssertionError("agent mode must not pre-retrieve")

    async def links(*args, **kwargs):
        return "Корисни линкови:\n- https://finki.ukim.mk"

    async def answer(payload, context, **kwargs):
        contexts.append(context)

        async def body():
            yield 'event: token\ndata: {"text": "ok"}\n\n'
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(body())

    monkeypatch.setattr(chat_api, "resolve_provider_credentials", resolve_credentials)
    monkeypatch.setattr(
        chat_api,
        "get_retrieved_context_with_sources",
        fail_if_retrieval_runs,
    )
    monkeypatch.setattr(chat_api, "get_links_context", links)
    monkeypatch.setattr(chat_api, "handle_chat", answer)
    monkeypatch.setattr(chat_api, "capture", lambda *args, **kwargs: None)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(settings=settings(retrieval_mode="agent")),
        ),
        headers={"X-Distinct-Id": "test-distinct-id"},
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
    [context] = contexts
    assert "пребарај ги со алатката" in context
    assert "https://finki.ukim.mk" in context
