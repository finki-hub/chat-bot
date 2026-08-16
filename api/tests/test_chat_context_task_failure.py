from types import SimpleNamespace
from uuid import UUID, uuid4

import anyio
import pytest

from app.api import chat as chat_api
from app.data.connection import Database
from app.llms.models import Model
from app.schemas.chat import ChatSchema
from tests.chat_models_access_support import credentials, settings

USER_ID = UUID("00000000-0000-4000-8000-000000000002")


def _request():
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=settings())),
        headers={"X-Distinct-Id": "test-distinct-id"},
    )


def _payload() -> ChatSchema:
    return ChatSchema.model_validate(
        {
            "user_id": USER_ID,
            "interface": "web",
            "inference_model": Model.GPT_5_6_LUNA,
            "messages": [{"role": "user", "content": "Question"}],
        },
    )


@pytest.mark.anyio
async def test_failed_links_context_wakes_stream_and_cleans_context_tasks(
    monkeypatch,
) -> None:
    retrieval_started = anyio.Event()
    retrieval_finished = anyio.Event()
    links_finished = anyio.Event()
    internal_message = "private link lookup failure"

    async def resolve_credentials(*args, **kwargs):
        return credentials(openai=True)

    async def retrieve(*args, **kwargs):
        retrieval_started.set()
        try:
            await anyio.sleep_forever()
        finally:
            retrieval_finished.set()

    async def fail_links(*args, **kwargs):
        try:
            raise RuntimeError(internal_message)
        finally:
            links_finished.set()

    async def fail_if_agent_starts(*args, **kwargs):
        raise AssertionError("agent must not start after link lookup fails")

    monkeypatch.setattr(
        chat_api,
        "resolve_provider_credentials",
        resolve_credentials,
    )
    monkeypatch.setattr(
        chat_api,
        "get_retrieved_context_with_sources",
        retrieve,
    )
    monkeypatch.setattr(chat_api, "get_links_context", fail_links)
    monkeypatch.setattr(chat_api, "handle_chat", fail_if_agent_starts)
    monkeypatch.setattr(chat_api, "capture", lambda *args, **kwargs: None)

    with anyio.fail_after(0.2):
        chunks = [
            str(chunk)
            async for chunk in chat_api._chat_response_stream(  # noqa: SLF001
                _payload(),
                _request(),
                Database.__new__(Database),
                uuid4(),
            )
        ]

    rendered = "".join(chunks)
    assert rendered.count('"code": "agent_error"') == 1
    assert internal_message not in rendered
    assert retrieval_started.is_set()
    assert retrieval_finished.is_set()
    assert links_finished.is_set()
