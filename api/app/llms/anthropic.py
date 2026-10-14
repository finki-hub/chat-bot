import logging
from collections.abc import Generator

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from app.llms.agents import (
    StreamObservation,
    capture_model_fallback,
    content_to_text,
    create_agent_token_generator,
    stream_sync_gen_as_sse,
)
from app.llms.models import (
    ANTHROPIC_MANDATORY_REASONING_MODELS,
    ANTHROPIC_NO_SAMPLING_MODELS,
    Model,
)
from app.llms.prompts import build_agent_messages
from app.llms.provider_credentials import require_provider_credential
from app.llms.tools import get_agent_tools
from app.schemas.chat_credentials import ChatCredentialSecret

logger = logging.getLogger(__name__)

# Explicit, so a user key never inherits the deployment's ANTHROPIC_BASE_URL gateway.
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"

# Active Claude models use summarized adaptive thinking when reasoning is enabled.
_ADAPTIVE_THINKING_MODELS: frozenset[Model] = ANTHROPIC_NO_SAMPLING_MODELS


def _thinking_config(
    model: Model,
    max_tokens: int,
) -> tuple[dict[str, object], int]:
    if model in _ADAPTIVE_THINKING_MODELS:
        return {"type": "adaptive", "display": "summarized"}, max_tokens
    msg = f"Model {model.value} is missing its adaptive thinking policy"
    raise RuntimeError(msg)


def get_anthropic_llm(
    model: Model,
    temperature: float,
    top_p: float,  # ruff: ignore[ARG001] -- provider factories share one call signature
    max_tokens: int,
    *,
    reasoning: bool = False,
    credential: ChatCredentialSecret | None = None,
    upstream_model: str | None = None,
) -> ChatAnthropic:
    """
    Return a user-scoped ChatAnthropic instance for the specified model and parameters.

    Anthropic's API is stricter about sampling parameters than the other providers:
    Some models reject sampling parameters outright (HTTP 400), and every Claude 4+ model
    rejects requests that set both `temperature` and `top_p`.
    To stay within those limits we never forward `top_p`, and forward `temperature` only
    for models that accept it (see `ANTHROPIC_NO_SAMPLING_MODELS`).

    Active Claude models use adaptive thinking when reasoning is enabled. Sonnet 5.5 uses
    `between_tools` when reasoning is off; Haiku 5.5 disables thinking at medium effort;
    Fable 5.1 and Opus 5.5 always keep reasoning on.
    """
    reasoning_requested = reasoning
    reasoning = reasoning or model in ANTHROPIC_MANDATORY_REASONING_MODELS
    thinking, effective_max = (None, max_tokens)
    if reasoning:
        thinking, effective_max = _thinking_config(model, max_tokens)
    elif model == Model.CLAUDE_SONNET_5_5:
        thinking = {"type": "between_tools"}
    elif model == Model.CLAUDE_HAIKU_5_5:
        thinking = {"type": "disabled"}
    temperature_arg = (
        None if (reasoning or model in ANTHROPIC_NO_SAMPLING_MODELS) else temperature
    )
    credential = require_provider_credential("anthropic", credential)
    client_kwargs: dict[str, object] = {
        "model": model.value if upstream_model is None else upstream_model,
        "api_key": SecretStr(credential.api_key),
        "base_url": credential.base_url or ANTHROPIC_DEFAULT_BASE_URL,
        "temperature": temperature_arg,
        "max_tokens": effective_max,
        "thinking": thinking,
    }
    # The installed client translates the effort alias to Anthropic's
    # output_config.effort. At medium effort adaptive thinking usually skips
    # thinking, so an explicit reasoning request raises it to high.
    if model in ANTHROPIC_NO_SAMPLING_MODELS:
        client_kwargs["effort"] = "high" if reasoning_requested else "medium"
    # LangChain's generated Pydantic signature omits documented constructor aliases.
    return ChatAnthropic(**client_kwargs)  # type: ignore[arg-type]


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
    credential: ChatCredentialSecret | None = None,
    upstream_model: str | None = None,
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

    llm = get_anthropic_llm(
        model,
        temperature,
        top_p,
        max_tokens,
        reasoning=reasoning,
        credential=credential,
        upstream_model=upstream_model,
    )
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
    credential: ChatCredentialSecret | None = None,
) -> str:
    """
    Transform a query using the specified Anthropic model and system prompt.
    If the transformation fails, return the original query.
    """

    logger.info(
        "Transforming query with model=%s query_len=%d",
        model.value,
        len(query),
    )

    try:
        llm = get_anthropic_llm(
            model,
            temperature,
            top_p,
            max_tokens,
            credential=credential,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        response = await llm.ainvoke(messages)
        return content_to_text(response.content).strip()
    # ruff: ignore[BLE001] -- query transformation is an optional upstream fallback
    except Exception as exc:
        logger.warning(
            "Anthropic query transformation failed; using original query model=%s error_type=%s",
            model.value,
            type(exc).__name__,
        )

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
    observation: StreamObservation | None = None,
    credential: ChatCredentialSecret | None = None,
    upstream_model: str | None = None,
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
            credential=credential,
            upstream_model=upstream_model,
        )

        tools = await get_agent_tools()

        logger.info(
            "Available tools: %s",
            ", ".join(tool.name for tool in tools) if tools else "None",
        )

        agent = create_agent(llm, tools)

        messages = build_agent_messages(system_prompt, history or [], user_prompt)

        # Sonnet 5.5's `between_tools` mode returns progress notes as thinking blocks;
        # keep them out of the stream when the user turned reasoning off.
        return StreamingResponse(
            create_agent_token_generator(
                agent,
                messages,
                observation,
                emit_reasoning=reasoning,
            ),
            media_type="text/event-stream",
        )

    # ruff: ignore[BLE001] -- agent setup falls back to the regular provider stream
    except Exception as exc:
        logger.warning(
            "Anthropic agent setup failed; using regular response model=%s error_type=%s",
            model.value,
            type(exc).__name__,
        )
        capture_model_fallback(
            observation,
            from_model=model.value,
            to_model=model.value,
            reason="agent_setup_failed",
        )

        return stream_anthropic_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning=reasoning,
            credential=credential,
            upstream_model=upstream_model,
        )
