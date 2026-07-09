from dataclasses import dataclass

import pytest

from app.llms import mcp
from app.utils.settings import McpServerSettings, Settings


@dataclass(frozen=True, slots=True)
class ToolStub:
    name: str


def server(name: str) -> McpServerSettings:
    return McpServerSettings(
        name=name,
        url=f"https://{name}-mcp:8808/mcp",
        transport="streamable_http",
    )


@pytest.mark.anyio
async def test_get_mcp_tools_retries_partial_refresh_instead_of_caching_it(
    monkeypatch,
):
    class FakeClient:
        def __init__(self) -> None:
            self.calls = {"search": 0, "records": 0}

        async def get_tools(self, *, server_name: str | None = None):
            if server_name is None:
                raise AssertionError("server_name is required")
            self.calls[server_name] += 1
            if server_name == "records" and self.calls[server_name] == 1:
                raise OSError("records MCP is temporarily unavailable")
            tool_name = "web_search" if server_name == "search" else "lookup"
            return [ToolStub(tool_name)]

    fake_client = FakeClient()
    monkeypatch.setattr(mcp, "build_mcp_client", lambda: fake_client)
    monkeypatch.setattr(mcp, "capture", lambda *args: None)
    monkeypatch.setattr(
        mcp,
        "settings",
        Settings(MCP_SERVERS=[server("search"), server("records")]),
    )
    monkeypatch.setattr(mcp, "mcp_tools", None)
    monkeypatch.setattr(mcp, "mcp_tools_fetched_at", -10_000.0)

    degraded_tools = await mcp.get_mcp_tools()
    refreshed_tools = await mcp.get_mcp_tools()

    assert [tool.name for tool in degraded_tools] == ["web_search"]
    assert [tool.name for tool in refreshed_tools] == ["web_search", "lookup"]
    assert fake_client.calls == {"search": 2, "records": 2}


@pytest.mark.anyio
async def test_get_mcp_tools_excludes_stale_tools_when_healthy_server_is_empty(
    monkeypatch,
):
    class FakeClient:
        async def get_tools(self, *, server_name: str | None = None):
            if server_name == "broken":
                raise OSError("broken MCP is unavailable")
            return []

    def build_fake_client() -> FakeClient:
        return FakeClient()

    stale_tools = [ToolStub("stale_broken_tool")]
    monkeypatch.setattr(mcp, "build_mcp_client", build_fake_client)
    monkeypatch.setattr(mcp, "capture", lambda *args: None)
    monkeypatch.setattr(
        mcp,
        "settings",
        Settings(MCP_SERVERS=[server("broken"), server("empty")]),
    )
    monkeypatch.setattr(mcp, "mcp_tools", stale_tools)
    monkeypatch.setattr(mcp, "mcp_tools_fetched_at", -10_000.0)

    tools = await mcp.get_mcp_tools()

    assert tools == []
    assert mcp.mcp_tools == stale_tools
    assert mcp.mcp_tools_fetched_at == -10_000.0


@pytest.mark.anyio
async def test_get_mcp_tools_caches_successful_empty_response(monkeypatch):
    class FakeClient:
        async def get_tools(self, *, server_name: str | None = None):
            return []

    def build_fake_client() -> FakeClient:
        return FakeClient()

    stale_tools = [ToolStub("stale_tool")]
    monkeypatch.setattr(mcp, "build_mcp_client", build_fake_client)
    monkeypatch.setattr(
        mcp,
        "settings",
        Settings(MCP_SERVERS=[server("empty")]),
    )
    monkeypatch.setattr(mcp, "mcp_tools", stale_tools)
    monkeypatch.setattr(mcp, "mcp_tools_fetched_at", -10_000.0)

    tools = await mcp.get_mcp_tools()

    assert tools == []
    assert mcp.mcp_tools == []
    assert mcp.mcp_tools_fetched_at > 0
