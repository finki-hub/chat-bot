from typing import cast

import pytest
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.llms import chat as chat_module
from app.llms.models import Model
from app.llms.prompts import AGENT_SEARCH_TOOLS_PROMPT
from app.llms.search_tools import KnowledgeSearch, SearchResult
from app.llms.tools import agent_request_tools, fallback_prompts
from app.schemas.chat import ChatSchema


@pytest.mark.anyio
async def test_fallback_prompts_are_unchanged_without_a_registered_builder() -> None:
    assert await fallback_prompts("system", "user") == ("system", "user")


@pytest.mark.anyio
async def test_fallback_prompts_keep_originals_when_the_builder_fails() -> None:
    async def failing() -> tuple[str, str]:
        raise RuntimeError

    with agent_request_tools([], failing):
        assert await fallback_prompts("system", "user") == ("system", "user")


@pytest.mark.anyio
async def test_tool_less_fallback_answers_from_a_search_on_the_question(
    monkeypatch,
) -> None:
    searched: list[str] = []
    prompts: list[tuple[str, str]] = []

    async def search(self, query: str) -> SearchResult:
        searched.append(query)
        return SearchResult(content="Тип на извор: FAQ", sources=[], hit=True)

    async def stream(user_prompt, model, *, system_prompt, **kwargs):
        prompts.append(await fallback_prompts(system_prompt, user_prompt))
        return StreamingResponse(iter(()))

    monkeypatch.setattr(KnowledgeSearch, "search", search)
    monkeypatch.setattr(chat_module, "stream_response_with_agent", stream)
    payload = ChatSchema.model_validate(
        {
            "interface": "web",
            "inference_model": Model.GPT_6_LUNA,
            "messages": [{"role": "user", "content": "Кога е уписот?"}],
        },
    )

    await chat_module.handle_chat(
        payload,
        "Денешен датум: 24.10.2026.",
        db=cast("Database", object()),
    )

    assert searched == ["Кога е уписот?"]
    [(system_prompt, user_prompt)] = prompts
    assert AGENT_SEARCH_TOOLS_PROMPT.split("\n", 1)[0] not in system_prompt
    assert "Тип на извор: FAQ" in user_prompt
    assert chat_module.AGENT_SEARCH_CONTEXT not in user_prompt
