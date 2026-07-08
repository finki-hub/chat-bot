from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.tools import BaseTool

from app.llms.mcp import get_mcp_tools

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
    return [*await get_mcp_tools(), *_request_tools.get()]
