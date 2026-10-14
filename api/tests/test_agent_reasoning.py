from typing import cast

import pytest
from langchain_core.messages import AIMessageChunk

from app.llms import agents, anthropic
from app.llms.agents import _chunk_reasoning
from app.llms.models import Model
from app.schemas.chat_credentials import ChatCredentialSecret
from tests.test_agent_error_log_privacy import TestAgentGraph


def test_chunk_reasoning_extracts_openrouter_text_details() -> None:
    chunk = AIMessageChunk(
        content="",
        additional_kwargs={
            "reasoning_details": [
                {"type": "reasoning.text", "text": "first"},
                {"type": "reasoning.encrypted", "data": "opaque"},
                {"type": "reasoning.text", "text": " second"},
            ],
        },
    )

    assert _chunk_reasoning(chunk) == "first second"


class _ThinkingThenTextGraph:
    async def astream_events(self, *args, **kwargs):
        blocks: tuple[list[str | dict], ...] = (
            [{"type": "thinking", "thinking": "progress note"}],
            [{"type": "text", "text": "answer"}],
        )
        for content in blocks:
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content=content)},
            }


@pytest.mark.anyio
@pytest.mark.parametrize("emit_reasoning", [True, False])
async def test_agent_stream_emits_thinking_only_when_requested(
    emit_reasoning: bool,
) -> None:
    events = [
        event
        async for event in agents.create_agent_token_generator(
            cast("TestAgentGraph", _ThinkingThenTextGraph()),
            [],
            emit_reasoning=emit_reasoning,
        )
    ]

    assert any(event.startswith("event: thinking") for event in events) is (
        emit_reasoning
    )
    assert any(
        "answer" in event for event in events if event.startswith("event: token")
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "reasoning", "expected"),
    [
        (Model.CLAUDE_SONNET_5_5, False, False),
        (Model.CLAUDE_SONNET_5_5, True, True),
        (Model.CLAUDE_HAIKU_5_5, False, False),
        (Model.CLAUDE_OPUS_5_5, False, True),
    ],
)
async def test_anthropic_agent_hides_thinking_when_reasoning_is_off(
    monkeypatch,
    model: Model,
    reasoning: bool,
    expected: bool,
) -> None:
    seen: dict[str, bool] = {}

    async def no_tools():
        return []

    async def empty_stream():
        yield ""

    def fake_generator(*args, emit_reasoning: bool = True, **kwargs):
        seen["emit_reasoning"] = emit_reasoning
        return empty_stream()

    monkeypatch.setattr(anthropic, "get_anthropic_llm", lambda *a, **k: object())
    monkeypatch.setattr(anthropic, "get_agent_tools", no_tools)
    monkeypatch.setattr(anthropic, "create_agent", lambda *a, **k: object())
    monkeypatch.setattr(anthropic, "create_agent_token_generator", fake_generator)

    await anthropic.stream_anthropic_agent_response(
        "question",
        model,
        system_prompt="system",
        temperature=0.2,
        top_p=1.0,
        max_tokens=1024,
        reasoning=reasoning,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    assert seen == {"emit_reasoning": expected}
