import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.tools import BaseTool

from app.llms.mcp import get_mcp_tools

logger = logging.getLogger(__name__)

# Per answer: searches beyond 4 stopped adding sources in testing, and one prod answer
# made 97 course-tool calls. Over-limit calls are blocked before they run and the model
# is told to answer with what it has.
SEARCH_TOOL_NAME = "search_knowledge_base"
MAX_SEARCHES_PER_ANSWER = 4
MAX_TOOL_CALLS_PER_ANSWER = 15

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


def agent_middleware() -> list[ToolCallLimitMiddleware]:
    return [
        ToolCallLimitMiddleware(
            tool_name=SEARCH_TOOL_NAME,
            run_limit=MAX_SEARCHES_PER_ANSWER,
        ),
        ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS_PER_ANSWER),
    ]
