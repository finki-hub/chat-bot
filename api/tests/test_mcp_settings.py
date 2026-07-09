import json
from dataclasses import dataclass

import pytest
from anyio import lowlevel
from pydantic import ValidationError

from app.llms import mcp
from app.utils.settings import McpServerSettings, Settings


@dataclass(frozen=True, slots=True)
class ToolStub:
    name: str


def test_settings_parses_structured_mcp_servers(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps(
            [
                {
                    "name": "search",
                    "url": "https://search-mcp:8808/mcp",
                    "transport": "streamable_http",
                    "api_key": "search-key",
                    "allowed_tools": ["web_search"],
                    "blocked_tools": ["web_fetch"],
                },
                {
                    "name": "events",
                    "url": "https://events-mcp:8808/sse",
                    "transport": "sse",
                },
            ],
        ),
    )

    settings = Settings()

    assert [server.name for server in settings.mcp_server_configs()] == [
        "search",
        "events",
    ]
    assert settings.mcp_server_configs()[0].api_key == "search-key"
    assert settings.mcp_server_configs()[0].allowed_tools == ("web_search",)
    assert settings.mcp_server_configs()[0].blocked_tools == ("web_fetch",)


def test_settings_rejects_duplicate_mcp_server_names():
    servers = [
        McpServerSettings(
            name="search",
            url="https://search-mcp:8808/mcp",
            transport="streamable_http",
        ),
        McpServerSettings(
            name="search",
            url="https://other-mcp:8808/mcp",
            transport="streamable_http",
        ),
    ]

    with pytest.raises(ValidationError, match="unique names"):
        Settings(MCP_SERVERS=servers)


def test_mcp_server_settings_rejects_blank_identity():
    with pytest.raises(ValidationError):
        McpServerSettings(
            name="   ",
            url="https://search-mcp:8808/mcp",
            transport="streamable_http",
        )


def test_insecure_secret_names_include_default_structured_mcp_key():
    settings = Settings(
        API_KEY="custom-api-key",
        MCP_SERVERS=[
            McpServerSettings(
                name="local",
                url="https://local-mcp:8808/mcp",
                transport="streamable_http",
                api_key=" SystemPass ",
            ),
        ],
    )

    assert settings.insecure_secret_names() == ["MCP_SERVERS.local.api_key"]


def test_build_mcp_client_uses_per_server_headers(monkeypatch):
    captured_connections = {}

    class FakeClient:
        def __init__(self, *, connections):
            captured_connections.update(connections)

    monkeypatch.setattr(mcp, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(
        mcp,
        "settings",
        Settings(
            MCP_SERVERS=[
                McpServerSettings(
                    name="search",
                    url="https://search-mcp:8808/mcp",
                    transport="streamable_http",
                    api_key="search-key",
                ),
                McpServerSettings(
                    name="events",
                    url="https://events-mcp:8808/sse",
                    transport="sse",
                ),
            ],
        ),
    )
    monkeypatch.setattr(mcp, "mcp_client", None)

    mcp.build_mcp_client()

    assert captured_connections == {
        "search": {
            "url": "https://search-mcp:8808/mcp",
            "transport": "streamable_http",
            "headers": {"X-Api-Key": "search-key"},
        },
        "events": {
            "url": "https://events-mcp:8808/sse",
            "transport": "sse",
        },
    }


@pytest.mark.anyio
async def test_get_mcp_tools_filters_each_server(monkeypatch):
    class FakeClient:
        async def get_tools(self, *, server_name=None):
            if server_name is None:
                msg = "server_name is required for this test fake"
                raise AssertionError(msg)
            tools_by_server = {
                "search": [
                    ToolStub("web_search"),
                    ToolStub("web_fetch"),
                    ToolStub("local_search"),
                ],
                "records": [ToolStub("lookup"), ToolStub("delete_record")],
            }
            return tools_by_server[server_name]

    def build_fake_client():
        return FakeClient()

    monkeypatch.setattr(mcp, "build_mcp_client", build_fake_client)
    monkeypatch.setattr(
        mcp,
        "settings",
        Settings(
            MCP_SERVERS=[
                McpServerSettings(
                    name="search",
                    url="https://search-mcp:8808/mcp",
                    transport="streamable_http",
                    allowed_tools=("web_search", "web_fetch"),
                    blocked_tools=("web_fetch",),
                ),
                McpServerSettings(
                    name="records",
                    url="https://records-mcp:8808/mcp",
                    transport="streamable_http",
                    blocked_tools=("delete_record",),
                ),
            ],
        ),
    )
    monkeypatch.setattr(mcp, "mcp_tools", None)
    monkeypatch.setattr(mcp, "mcp_tools_fetched_at", 0.0)

    tools = await mcp.get_mcp_tools()

    assert [tool.name for tool in tools] == ["web_search", "lookup"]


@pytest.mark.anyio
async def test_get_mcp_tools_keeps_healthy_servers_when_one_server_fails(monkeypatch):
    captured = []

    class FakeClient:
        async def get_tools(self, *, server_name=None):
            await lowlevel.checkpoint()
            if server_name == "broken":
                raise OSError("TLS certificate verification failed")
            return [ToolStub("lookup")]

    def build_fake_client():
        return FakeClient()

    monkeypatch.setattr(mcp, "build_mcp_client", build_fake_client)
    monkeypatch.setattr(
        mcp,
        "capture",
        lambda distinct_id, event, props: captured.append((distinct_id, event, props)),
    )
    monkeypatch.setattr(
        mcp,
        "settings",
        Settings(
            MCP_SERVERS=[
                McpServerSettings(
                    name="local",
                    url="https://local-mcp:8808/mcp",
                    transport="streamable_http",
                ),
                McpServerSettings(
                    name="broken",
                    url="https://broken-mcp:8808/sse",
                    transport="sse",
                ),
            ],
        ),
    )
    monkeypatch.setattr(mcp, "mcp_tools", None)
    monkeypatch.setattr(mcp, "mcp_tools_fetched_at", 0.0)

    tools = await mcp.get_mcp_tools()

    assert [tool.name for tool in tools] == ["lookup"]
    assert captured == [
        (
            "server",
            "mcp_server_tool_loading_failed",
            {"server_name": "broken", "transport": "sse", "error_type": "OSError"},
        ),
    ]


@pytest.mark.anyio
async def test_get_mcp_tools_caches_empty_list_when_no_servers(monkeypatch):
    def fail_build_client():
        msg = "MCP client should not be built without configured servers"
        raise AssertionError(msg)

    monkeypatch.setattr(mcp, "build_mcp_client", fail_build_client)
    monkeypatch.setattr(mcp, "settings", Settings(MCP_SERVERS=[]))
    monkeypatch.setattr(mcp, "mcp_tools", None)
    monkeypatch.setattr(mcp, "mcp_tools_fetched_at", -10_000.0)

    tools = await mcp.get_mcp_tools()

    assert tools == []
    assert mcp.mcp_tools == []
    assert mcp.mcp_tools_fetched_at > 0


@pytest.mark.anyio
async def test_get_mcp_tools_caches_intentionally_empty_filter_result(monkeypatch):
    class FakeClient:
        async def get_tools(self, *, server_name=None):
            if server_name is None:
                msg = "server_name is required for this test fake"
                raise AssertionError(msg)
            return [ToolStub("blocked_tool")]

    def build_fake_client():
        return FakeClient()

    monkeypatch.setattr(mcp, "build_mcp_client", build_fake_client)
    monkeypatch.setattr(
        mcp,
        "settings",
        Settings(
            MCP_SERVERS=[
                McpServerSettings(
                    name="blocked",
                    url="https://blocked-mcp:8808/mcp",
                    transport="streamable_http",
                    blocked_tools=("blocked_tool",),
                ),
            ],
        ),
    )
    monkeypatch.setattr(mcp, "mcp_tools", [ToolStub("stale_tool")])
    monkeypatch.setattr(mcp, "mcp_tools_fetched_at", -10_000.0)

    tools = await mcp.get_mcp_tools()

    assert tools == []
