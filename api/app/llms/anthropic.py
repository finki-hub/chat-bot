import logging
from collections.abc import Generator

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from app.llms.agents import (
    content_to_text,
    create_agent_token_generator,
    stream_sync_gen_as_sse,
    thinking_budget,
)
from app.llms.mcp import get_mcp_tools
from app.llms.models import ANTHROPIC_NO_SAMPLING_MODELS, Model
from app.llms.prompts import build_agent_messages
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Model, temperature, top_p, max_tokens, reasoning -> LLM
anthropic_llm_clients: dict[tuple[str, float, float, int, bool], ChatAnthropic] = {}

# Opus 4.7/4.8 reject `budget_tokens`; they use summarized adaptive thinking instead.
_ADAPTIVE_THINKING_MODELS: frozenset[Model] = ANTHROPIC_NO_SAMPLING_MODELS

# Anthropic rejects an enabled-thinking `budget_tokens` below 1024 and requires
# `max_tokens` to exceed it. `thinking_budget` can fall below 1024 for small `max_tokens`
# (e.g. max_tokens=2000 -> 1000), which 400s sonnet-4-6 / haiku-4-5, so clamp to the floor.
_MIN_ANTHROPIC_THINKING_BUDGET = 1024


def _thinking_config(
    model: Model,
    max_tokens: int,
) -> tuple[dict[str, object], int]:
    """The Anthropic ``thinking`` payload and effective ``max_tokens`` (adaptive for Opus
    4.7/4.8, else an explicit budget clamped to Anthropic's 1024-token floor)."""
    if model in _ADAPTIVE_THINKING_MODELS:
        return {"type": "adaptive", "display": "summarized"}, max_tokens

    budget, effective_max = thinking_budget(max_tokens)
    budget = max(budget, _MIN_ANTHROPIC_THINKING_BUDGET)
    # max_tokens must exceed budget_tokens; always keep at least a budget's worth of room
    # for the answer so a small max_tokens isn't swallowed whole by the thinking budget
    # (without this, max_tokens just above 1024 leaves a near-empty answer).
    effective_max = max(effective_max, budget + _MIN_ANTHROPIC_THINKING_BUDGET)
    return {"type": "enabled", "budget_tokens": budget}, effective_max


def get_anthropic_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
    *,
    reasoning: bool = False,
) -> ChatAnthropic:
    """
    Return a singleton ChatAnthropic instance for the specified model and parameters.

    Anthropic's API is stricter about sampling parameters than the other providers:
    Claude Opus 4.7 / 4.8 reject `temperature`, `top_p`, and `top_k` outright (HTTP 400),
    and every Claude 4+ model rejects requests that set both `temperature` and `top_p`.
    To stay within those limits we never forward `top_p`, and forward `temperature` only
    for models that accept it (see `ANTHROPIC_NO_SAMPLING_MODELS`).

    When `reasoning` is on, extended thinking additionally requires `temperature` to be
    unset (the API uses its default of 1), so we send `None` for every reasoning request.
    """
    key = (model.value, temperature, top_p, max_tokens, reasoning)

    if key not in anthropic_llm_clients:
        thinking, effective_max = (
            _thinking_config(model, max_tokens) if reasoning else (None, max_tokens)
        )
        temperature_arg = (
            None
            if (reasoning or model in ANTHROPIC_NO_SAMPLING_MODELS)
            else temperature
        )
        anthropic_llm_clients[key] = ChatAnthropic(
            model=model.value,  # type: ignore[call-arg]
            api_key=SecretStr(settings.ANTHROPIC_API_KEY),
            base_url=settings.ANTHROPIC_BASE_URL or None,
            temperature=temperature_arg,
            max_tokens=effective_max,  # type: ignore[call-arg]
            thinking=thinking,  # type: ignore[call-arg]
        )

    return anthropic_llm_clients[key]


def stream_anthropic_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    history: list[BaseMessage] | None = None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool = False,
) -> StreamingResponse:
    """
    Stream a response from the specified Anthropic model using the provided prompts.
    The response is formatted as Server-Sent Events (SSE).
    This function is a direct parallel to stream_openai_response.
    """
    logger.info(
        "Streaming Anthropic response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    llm = get_anthropic_llm(model, temperature, top_p, max_tokens, reasoning=reasoning)
    prompt_messages = build_agent_messages(system_prompt, history or [], user_prompt)

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(prompt_messages):
            yield content_to_text(chunk.content)

    return stream_sync_gen_as_sse(sync_token_gen())


async def transform_query_with_anthropic(
    query: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """
    Transform a query using the specified Anthropic model and system prompt.
    If the transformation fails, return the original query.
    """

    logger.info(
        "Transforming query: '%s' with model: %s",
        query,
        model,
    )

    try:
        llm = get_anthropic_llm(model, temperature, top_p, max_tokens)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        response = await llm.ainvoke(messages)
        return content_to_text(response.content).strip()
    except Exception:
        logger.exception("Query transformation failed: %s. Using original query.")

        return query


async def stream_anthropic_agent_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    history: list[BaseMessage] | None = None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool = False,
) -> StreamingResponse:
    """
    Stream a response from an Anthropic agent with MCP tools.
    Falls back to regular response if MCP unavailable.
    """
    logger.info(
        "Streaming Anthropic agent response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    try:
        llm = get_anthropic_llm(
            model,
            temperature,
            top_p,
            max_tokens,
            reasoning=reasoning,
        )

        tools = await get_mcp_tools()

        logger.info(
            "Available tools: %s",
            ", ".join(tool.name for tool in tools) if tools else "None",
        )

        agent = create_agent(llm, tools)

        messages = build_agent_messages(system_prompt, history or [], user_prompt)

        return StreamingResponse(
            create_agent_token_generator(agent, messages),
            media_type="text/event-stream",
        )

    except Exception:
        logger.exception(
            "Failed to stream Anthropic agent response. Falling back to regular response",
        )

        return stream_anthropic_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
