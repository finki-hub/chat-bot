import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.tools import BaseTool

from app.llms.mcp import get_mcp_tools

logger = logging.getLogger(__name__)

_request_tools: ContextVar[tuple[BaseTool, ...]] = ContextVar(
    "request_tools",
    default=(),
)


@contextmanager
def agent_request_tools(tools: Sequence[BaseTool]) -> Iterator[None]:
    token = _request_tools.set(tuple(tools))
    try:
        yield
    finally:
        _request_tools.reset(token)


async def get_agent_tools() -> list[BaseTool]:
    request_tools = _request_tools.get()
    try:
        mcp_tools = await get_mcp_tools()
    except Exception as exc:
        if not request_tools:
            raise
        logger.warning(
            "MCP tool loading failed; using request tools error_type=%s",
            type(exc).__name__,
        )
        return list(request_tools)
    return [*mcp_tools, *request_tools]
