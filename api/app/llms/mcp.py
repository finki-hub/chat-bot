import logging
import time

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    Connection,
    SSEConnection,
    StreamableHttpConnection,
)

from app.utils.settings import Settings
from app.utils.timing import timed

logger = logging.getLogger(__name__)

settings = Settings()

mcp_client: MultiServerMCPClient | None = None
mcp_tools: list | None = None
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

    headers: dict[str, str] | None = (
        {"X-Api-Key": settings.MCP_API_KEY} if settings.MCP_API_KEY else None
    )

    for url in settings.mcp_http_url_list():
        logger.info(
            "Adding streamable HTTP connection to MCP client: %s",
            url,
        )

        streamable_connection: StreamableHttpConnection = {
            "url": url,
            "transport": "streamable_http",
        }
        if headers:
            streamable_connection["headers"] = headers
        connections[url] = streamable_connection

    for url in settings.mcp_sse_url_list():
        logger.info(
            "Adding SSE connection to MCP client: %s",
            url,
        )

        sse_connection: SSEConnection = {
            "url": url,
            "transport": "sse",
        }
        if headers:
            sse_connection["headers"] = headers
        connections[url] = sse_connection

    logger.info(
        "Building MCP client with %d connections: %s",
        len(connections),
        list(connections.keys()),
    )

    mcp_client = MultiServerMCPClient(connections=connections)

    return mcp_client


async def get_mcp_tools() -> list:
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

    client = build_mcp_client()
    # Timed only on an actual fetch (cache hits return above), so the span's presence on
    # a request flags a refresh — and a degraded MCP server, which refetches every time.
    with timed("agent.mcp_tools"):
        fetched = await client.get_tools()

    if not fetched:
        # Empty usually means the MCP server is unreachable/degraded. Keep the last-good
        # tools and leave the timestamp stale so the next request retries, instead of
        # caching a tool-less agent for the whole TTL.
        logger.warning(
            "MCP returned an empty tool list; keeping previously cached tools (if any) "
            "and retrying on the next request instead of caching the empty result",
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
