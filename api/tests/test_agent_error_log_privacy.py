import logging
from collections.abc import AsyncGenerator
from typing import cast

import anyio
import pytest
from anyio.lowlevel import checkpoint
from langchain.agents.middleware.types import (
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.llms import agents as agents_module

_PRIVATE_PROVIDER_ERROR = (
    "provider rejected key=secret at https://private-provider.example/v1"
)

type TestAgentGraph = CompiledStateGraph[
    AgentState[object], None, InputAgentState, OutputAgentState[object]
]


class _FailingAgent:
    def __init__(self) -> None:
        self.agent_input: InputAgentState | None = None
        self.config: dict[str, object] | None = None
        self.version: str | None = None

    async def astream_events(
        self,
        agent_input: InputAgentState,
        config: dict[str, object],
        *,
        version: str,
    ) -> AsyncGenerator[dict[str, str]]:
        self.agent_input = agent_input
        self.config = config
        self.version = version
        await checkpoint()
        yield {"event": "on_chain_start"}
        raise RuntimeError(_PRIVATE_PROVIDER_ERROR)


def test_agent_stream_failure_omits_raw_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger=agents_module.__name__)
    fake_agent = _FailingAgent()

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in agents_module.create_agent_token_generator(
                cast("TestAgentGraph", fake_agent),
                [HumanMessage(content="private prompt")],
            )
        ]

    chunks = anyio.run(collect)

    assert fake_agent.agent_input is not None
    received_messages = fake_agent.agent_input["messages"]
    assert len(received_messages) == 1
    assert isinstance(received_messages[0], dict)
    assert received_messages[0]["type"] == "human"
    assert fake_agent.config == {"configurable": {"thread_id": "default"}}
    assert fake_agent.version == "v2"
    assert _PRIVATE_PROVIDER_ERROR not in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert any('"code": "agent_error"' in chunk for chunk in chunks)
