import logging

import pytest

from app.llms import mcp
from app.utils.settings import McpServerSettings, Settings


class SimulatedCancellationError(Exception):
    pass


@pytest.mark.anyio
async def test_get_mcp_tools_propagates_backend_cancellation(monkeypatch, caplog):
    captured_events = []

    class FakeClient:
        async def get_tools(self, *, server_name=None):
            raise SimulatedCancellationError

    def build_fake_client():
        return FakeClient()

    monkeypatch.setattr(mcp, "build_mcp_client", build_fake_client)
    monkeypatch.setattr(
        mcp,
        "get_cancelled_exc_class",
        lambda: SimulatedCancellationError,
    )
    monkeypatch.setattr(
        mcp,
        "capture",
        lambda distinct_id, event, props: captured_events.append(
            (distinct_id, event, props),
        ),
    )
    monkeypatch.setattr(
        mcp,
        "settings",
        Settings(
            MCP_SERVERS=[
                McpServerSettings(
                    name="cancelled",
                    url="https://cancelled-mcp:8808/mcp",
                    transport="streamable_http",
                ),
            ],
        ),
    )
    monkeypatch.setattr(mcp, "mcp_tools", None)
    monkeypatch.setattr(mcp, "mcp_tools_fetched_at", 0.0)

    with (
        caplog.at_level(logging.WARNING, logger=mcp.__name__),
        pytest.raises(SimulatedCancellationError),
    ):
        await mcp.get_mcp_tools()

    assert captured_events == []
    assert caplog.messages == [
        "MCP server tool loading cancelled; propagating cancellation: cancelled",
    ]
