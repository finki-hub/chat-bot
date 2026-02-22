import logging

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    Connection,
    SSEConnection,
    StreamableHttpConnection,
)

from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

mcp_client: MultiServerMCPClient | None = None


def build_mcp_client() -> MultiServerMCPClient:
    """
    Return a singleton MultiServerMCPClient instance.
    If the client is not already created, it initializes a new one.
    The client is configured for all provided MCP servers.
    """
    global mcp_client  # noqa: PLW0603

    logger.info("Building MCP client...")

    if mcp_client is not None:
        return mcp_client

    connections: dict[str, Connection] = {}

    for url in settings.mcp_http_url_list():
        logger.info(
            "Adding streamable HTTP connection to MCP client: %s",
            url,
        )

        streamable_connection: StreamableHttpConnection = {
            "url": url,
            "transport": "streamable_http",
        }
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
        connections[url] = sse_connection

    logger.info(
        "Building MCP client with %d connections: %s",
        len(connections),
        list(connections.keys()),
    )

    mcp_client = MultiServerMCPClient(connections=connections)

    return mcp_client
