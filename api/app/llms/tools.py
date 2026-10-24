import logging
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.tools import BaseTool

from app.llms.mcp import get_mcp_tools

logger = logging.getLogger(__name__)

# Per answer: searches beyond 4 stopped adding sources in testing, and one prod answer
# made 97 course-tool calls. Over-limit tool calls are blocked before they run and the
# model is told to answer with what it has; the model-call limit ends a run that keeps
# calling tools anyway.
SEARCH_TOOL_NAME = "search_knowledge_base"
MAX_SEARCHES_PER_ANSWER = 4
MAX_TOOL_CALLS_PER_ANSWER = 15
MAX_MODEL_CALLS_PER_ANSWER = 10

type FallbackPrompts = Callable[[], Awaitable[tuple[str, str]]]

_request_tools: ContextVar[tuple[BaseTool, ...]] = ContextVar(
    "request_tools",
    default=(),
)
_fallback_prompts: ContextVar[FallbackPrompts | None] = ContextVar(
    "fallback_prompts",
    default=None,
)


@contextmanager
def agent_request_tools(
    tools: Sequence[BaseTool],
    fallback: FallbackPrompts | None = None,
) -> Iterator[None]:
    tools_token = _request_tools.set(tuple(tools))
    fallback_token = _fallback_prompts.set(fallback)
    try:
        yield
    finally:
        _fallback_prompts.reset(fallback_token)
        _request_tools.reset(tools_token)


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


async def fallback_prompts(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """Prompts for the tool-less fallback answer: the request's own search-based
    prompts when registered, so the fallback still answers from retrieved sources."""
    build = _fallback_prompts.get()
    if build is None:
        return system_prompt, user_prompt
    try:
        return await build()
    # ruff: ignore[BLE001] -- the fallback answer must not fail on its own search
    except Exception as exc:
        logger.warning(
            "Fallback prompt search failed error_type=%s",
            type(exc).__name__,
        )
        return system_prompt, user_prompt


def agent_middleware() -> list[AgentMiddleware[Any, Any, Any]]:
    return [
        ToolCallLimitMiddleware(
            tool_name=SEARCH_TOOL_NAME,
            run_limit=MAX_SEARCHES_PER_ANSWER,
        ),
        ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS_PER_ANSWER),
        ModelCallLimitMiddleware(
            run_limit=MAX_MODEL_CALLS_PER_ANSWER,
            exit_behavior="end",
        ),
    ]
