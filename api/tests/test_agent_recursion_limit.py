import json
from collections.abc import Iterator
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool

from app.llms.agents import create_agent_token_generator
from app.llms.tools import agent_middleware

_TOOL_ROUNDS = 8


class _ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003, ANN201, ARG002
        return self

    # The stock fake streams nothing for a message without text, so emit each turn,
    # tool calls included, as a single chunk.
    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:  # noqa: ARG002
        message = next(self.messages)
        assert isinstance(message, AIMessage)
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_call_chunks=[
                    {
                        "name": call["name"],
                        "args": json.dumps(call["args"]),
                        "id": call["id"],
                        "index": index,
                    }
                    for index, call in enumerate(message.tool_calls)
                ],
            ),
        )


@tool
def list_courses(query: str) -> str:
    """List courses."""
    return f"courses for {query}"


def _turns() -> Iterator[AIMessage]:
    for index in range(_TOOL_ROUNDS):
        yield AIMessage(
            content="",
            tool_calls=[
                {"name": "list_courses", "args": {"query": "q"}, "id": f"call{index}"},
            ],
        )
    yield AIMessage(content="Готово.")


def _events(chunks: list[str]) -> list[tuple[str, dict]]:
    events = []
    for chunk in chunks:
        lines = chunk.strip().splitlines()
        name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((name, data))
    return events


@pytest.mark.anyio
async def test_long_tool_runs_end_by_call_limits_not_graph_steps():
    # Eight tool rounds stay under the model and tool call limits but take more than
    # LangChain's default 25 graph steps.
    agent = create_agent(
        _ToolCallingModel(messages=_turns()),
        [list_courses],
        middleware=agent_middleware(),
    )

    chunks = [
        chunk
        async for chunk in create_agent_token_generator(
            agent,
            [HumanMessage(content="Кои предмети?")],
        )
    ]
    events = _events(chunks)

    assert not [data for name, data in events if name == "error"]
    assert sum(name == "tool_call" for name, _ in events) == _TOOL_ROUNDS
    assert "Готово." in "".join(
        data["text"] for name, data in events if name == "token"
    )
