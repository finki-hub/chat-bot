import asyncio
import logging
from collections.abc import Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from app.llms.agents import (
    StreamObservation,
    capture_model_fallback,
    content_to_text,
    create_agent_token_generator,
    stream_sync_gen_as_sse,
)
from app.llms.models import (
    OPENAI_MINIMAL_EFFORT_MODELS,
    REASONING_CAPABLE_MODELS,
    Model,
)
from app.llms.prompts import build_agent_messages
from app.llms.provider_credentials import require_provider_credential
from app.llms.tools import get_agent_tools
from app.schemas.chat_credentials import ChatCredentialSecret

logger = logging.getLogger(__name__)


def get_openai_embedder(
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> OpenAIEmbeddings:
    credential = require_provider_credential("openai", credential)
    return OpenAIEmbeddings(
        model=model.value,
        api_key=SecretStr(credential.api_key),  # type: ignore[call-arg]
        base_url=credential.base_url or None,
    )


def get_openai_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
    *,
    reasoning: bool = False,
    credential: ChatCredentialSecret | None = None,
    upstream_model: str | None = None,
) -> ChatOpenAI:
    """
    Return a user-scoped ChatOpenAI instance for the specified model and parameters.

    All OpenAI requests use the Responses API (`use_responses_api=True`); reasoning content
    is only surfaced there. When `reasoning` is on, a `reasoning` config requests a
    summarized reasoning trace (GPT-5 reasoning models ignore `temperature`, which the
    wrapper strips automatically). When `reasoning` is off, models in
    `OPENAI_MINIMAL_EFFORT_MODELS` are pinned to `effort: "minimal"` so they don't burn the
    whole token budget on hidden reasoning and return an empty answer.

    GPT-5 reasoning models reject `top_p` on the Responses API (HTTP 400 once a reasoning
    request is made); the wrapper strips `temperature` for them but not `top_p`, so `top_p`
    is forwarded only for the non-reasoning chat models. This mirrors the Anthropic client.
    """
    # top_p is dropped for reasoning-capable (GPT-5) models, so it must not split the cache
    # for them — fold it out of the key when it isn't forwarded.
    forwards_top_p = model not in REASONING_CAPABLE_MODELS
    client_kwargs: dict[str, object] = {"use_responses_api": True}
    if reasoning:
        client_kwargs["reasoning"] = {"effort": "medium", "summary": "auto"}
    elif model in OPENAI_MINIMAL_EFFORT_MODELS:
        client_kwargs["reasoning"] = {"effort": "minimal"}
    if forwards_top_p:
        client_kwargs["top_p"] = top_p

    credential = require_provider_credential("openai", credential)
    return ChatOpenAI(
        model=model.value if upstream_model is None else upstream_model,
        api_key=SecretStr(credential.api_key),
        base_url=credential.base_url or None,
        temperature=temperature,
        streaming=True,
        stream_usage=True,
        max_tokens=max_tokens,  # type: ignore[call-arg]
        **client_kwargs,
    )


@overload
async def generate_openai_embeddings(
    text: str,
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> list[float]: ...


@overload
async def generate_openai_embeddings(
    text: list[str],
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> list[list[float]]: ...


async def generate_openai_embeddings(
    text: str | list[str],
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings for the given text using the specified OpenAI model.
    This function runs the embedding generation in a separate thread to avoid blocking the event loop.
    """
    logger.info(
        "Generating OpenAI embeddings for text with length '%s' with model: %s",
        len(text) if isinstance(text, str) else sum(len(t) for t in text),
        model.value,
    )

    emb = get_openai_embedder(model, credential)

    if isinstance(text, str):
        return await asyncio.to_thread(emb.embed_query, text)

    return await asyncio.to_thread(emb.embed_documents, text)


def stream_openai_response(
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
    Stream a response from the specified OpenAI model using the provided user prompt and system prompt.
    The response is formatted as Server-Sent Events (SSE) for real-time updates.
    """
    logger.info(
        "Streaming OpenAI response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    llm = get_openai_llm(
        model,
        temperature,
        top_p,
        max_tokens,
        reasoning=reasoning,
        credential=credential,
        upstream_model=upstream_model,
    )

    messages = build_agent_messages(system_prompt, history or [], user_prompt)

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(messages):
            yield content_to_text(chunk.content)

    return stream_sync_gen_as_sse(sync_token_gen())


async def stream_openai_agent_response(
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
    Stream a response from an OpenAI agent with MCP tools.
    Falls back to regular response if MCP unavailable.
    """
    logger.info(
        "Streaming OpenAI agent response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    try:
        llm = get_openai_llm(
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

        return StreamingResponse(
            create_agent_token_generator(agent, messages, observation),
            media_type="text/event-stream",
        )

    except Exception:
        logger.exception(
            "Failed to stream OpenAI agent response. Falling back to regular response",
        )
        capture_model_fallback(
            observation,
            from_model=model.value,
            to_model=model.value,
            reason="agent_setup_failed",
        )

        return stream_openai_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            credential=credential,
            upstream_model=upstream_model,
        )


async def transform_query_with_openai(
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
    Transform a query using the specified OpenAI model and system prompt.
    If the transformation fails, return the original query.
    """

    logger.info(
        "Transforming query with model=%s query_len=%d",
        model.value,
        len(query),
    )

    try:
        llm = get_openai_llm(
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
    except Exception:
        logger.exception("Query transformation failed; using the original query.")

        return query
