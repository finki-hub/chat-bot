import logging
import time
from collections.abc import Sequence
from typing import Protocol, assert_never

from anyio import get_cancelled_exc_class
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    Connection,
    SSEConnection,
    StreamableHttpConnection,
)

from app.utils.posthog_client import capture
from app.utils.settings import McpServerSettings, Settings
from app.utils.timing import timed

logger = logging.getLogger(__name__)

settings = Settings()

mcp_client: MultiServerMCPClient | None = None


class NamedTool(Protocol):
    name: str


mcp_tools: list[BaseTool] | None = None
mcp_tools_fetched_at: float = 0.0


def build_mcp_client() -> MultiServerMCPClient:
    """
    Return a singleton MultiServerMCPClient instance.
    If the client is not already created, it initializes a new one.
    The client is configured for all provided MCP servers.
    """
    global mcp_client  # noqa: PLW0603

    if mcp_client is not None:
        return mcp_client

    logger.info("Building MCP client...")

    connections: dict[str, Connection] = {}

    for server in settings.mcp_server_configs():
        logger.info(
            "Adding %s connection to MCP client: %s",
            server.transport,
            server.name,
        )
        connections[server.name] = _connection_for_server(server)

    logger.info(
        "Building MCP client with %d connections: %s",
        len(connections),
        list(connections.keys()),
    )

    mcp_client = MultiServerMCPClient(connections=connections)

    return mcp_client


def _connection_for_server(server: McpServerSettings) -> Connection:
    connection: Connection
    match server.transport:
        case "streamable_http":
            connection = StreamableHttpConnection(
                {"url": server.url, "transport": "streamable_http"},
            )
        case "sse":
            connection = SSEConnection(
                {"url": server.url, "transport": "sse"},
            )
        case unreachable:
            assert_never(unreachable)

    api_key = server.api_key.strip()
    if api_key:
        connection["headers"] = {"X-Api-Key": api_key}
    return connection


def _filter_tools[ToolT: NamedTool](
    tools: Sequence[ToolT],
    server: McpServerSettings,
) -> list[ToolT]:
    allowed_tools = set(server.allowed_tools)
    blocked_tools = set(server.blocked_tools)
    return [
        tool
        for tool in tools
        if (not allowed_tools or tool.name in allowed_tools)
        and tool.name not in blocked_tools
    ]


async def get_mcp_tools() -> list[BaseTool]:
    """
    Return a cached list of MCP tools, refreshing after MCP_TOOLS_TTL seconds.
    This avoids creating a new MCP session on every request while still picking
    up newly registered tools once the TTL expires.
    """
    global mcp_tools, mcp_tools_fetched_at  # noqa: PLW0603

    now = time.monotonic()
    ttl = settings.MCP_TOOLS_TTL

    if mcp_tools is not None and (now - mcp_tools_fetched_at) < ttl:
        return mcp_tools

    servers = settings.mcp_server_configs()
    if not servers:
        mcp_tools = []
        mcp_tools_fetched_at = now
        return mcp_tools

    client = build_mcp_client()
    # Timed only on an actual fetch (cache hits return above), so the span's presence on
    # a request flags a refresh — and a degraded MCP server, which refetches every time.
    failed_server_names: list[str] = []
    successful_server_count = 0
    with timed("agent.mcp_tools"):
        fetched: list[BaseTool] = []
        for server in servers:
            try:
                server_tools = await client.get_tools(server_name=server.name)
            except get_cancelled_exc_class():
                logger.warning(
                    "MCP server tool loading cancelled; propagating cancellation: %s",
                    server.name,
                )
                raise
            except Exception as exc:
                failed_server_names.append(server.name)
                logger.warning(
                    "MCP server tool loading failed; skipping server: %s",
                    server.name,
                    exc_info=True,
                )
                capture(
                    "server",
                    "mcp_server_tool_loading_failed",
                    {
                        "server_name": server.name,
                        "transport": server.transport,
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            successful_server_count += 1
            fetched.extend(_filter_tools(server_tools, server))

    if failed_server_names:
        if successful_server_count > 0:
            logger.warning(
                "Fetched %d MCP tools from %d/%d MCP servers; failed servers (%s) "
                "will be retried on the next request instead of caching a degraded list",
                len(fetched),
                successful_server_count,
                len(servers),
                ", ".join(failed_server_names),
            )
            return fetched

        logger.warning(
            "All MCP servers failed (%s); keeping previously cached tools (if any) "
            "and retrying on the next request instead of caching the failed refresh",
            ", ".join(failed_server_names),
        )
        return mcp_tools if mcp_tools is not None else []

    mcp_tools = fetched
    mcp_tools_fetched_at = now

    logger.info(
        "Fetched and cached %d MCP tools (TTL=%ds): %s",
        len(mcp_tools),
        ttl,
        ", ".join(tool.name for tool in mcp_tools),
    )

    return mcp_tools
