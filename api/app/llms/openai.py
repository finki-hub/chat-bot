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
    OPENAI_NO_SAMPLING_MODELS,
    REASONING_CAPABLE_MODELS,
    Model,
)
from app.llms.prompts import build_agent_messages
from app.llms.provider_credentials import require_provider_credential
from app.llms.tools import agent_middleware, get_agent_tools
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import is_opencode_go_base_url

logger = logging.getLogger(__name__)

# Explicit, so a user key never inherits the deployment's OPENAI_BASE_URL gateway.
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Models whose Responses API accepts `effort: none`; omitting reasoning keeps the
# model default (medium), which adds seconds to short auxiliary calls.
_REASONING_NONE_MODELS: frozenset[Model] = frozenset({Model.GPT_6_LUNA})


def get_openai_embedder(
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> OpenAIEmbeddings:
    credential = require_provider_credential("openai", credential)
    # LangChain's generated Pydantic signature omits documented constructor aliases.
    return OpenAIEmbeddings(  # type: ignore[call-arg]
        model=model.value,
        api_key=SecretStr(credential.api_key),
        base_url=credential.base_url or OPENAI_DEFAULT_BASE_URL,
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
    conversation_id: str | None = None,
    disable_reasoning: bool = False,
) -> ChatOpenAI:
    """
    Return a user-scoped ChatOpenAI instance for the specified model and parameters.

    OpenAI requests use the Responses API (`use_responses_api=True`) except for Astra,
    whose documented Chat Completions surface is used for `reasoning_effort`.
    When `reasoning` is on, a `reasoning` config requests a summarized reasoning trace.

    Reasoning models reject sampling controls on the Responses API, so `top_p` is
    forwarded only for the non-reasoning chat models and temperature is explicitly omitted
    for `OPENAI_NO_SAMPLING_MODELS`.
    """
    # top_p is dropped for reasoning-capable (GPT-5) models, so it must not split the cache
    # for them — fold it out of the key when it isn't forwarded.
    forwards_top_p = model not in REASONING_CAPABLE_MODELS
    client_kwargs: dict[str, object] = {"use_responses_api": True}
    if model == Model.GPT_6_ASTRA:
        # Astra's documented Chat Completions surface uses reasoning_effort.  Keep this
        # separate from the Responses-only reasoning summary shape used by older models.
        client_kwargs["use_responses_api"] = False
        if reasoning:
            client_kwargs["reasoning_effort"] = "medium"
    elif reasoning:
        client_kwargs["reasoning"] = {"effort": "medium", "summary": "auto"}
    elif disable_reasoning and model in _REASONING_NONE_MODELS:
        client_kwargs["reasoning"] = {"effort": "none"}
    if forwards_top_p:
        client_kwargs["top_p"] = top_p

    credential = require_provider_credential("openai", credential)
    if conversation_id is not None and is_opencode_go_base_url(credential.base_url):
        client_kwargs["default_headers"] = {
            "User-Agent": "finki-hub/1.0",
            "x-opencode-session": conversation_id,
        }
    # LangChain's generated Pydantic signature omits documented constructor aliases.
    return ChatOpenAI(  # type: ignore[call-arg]
        model=model.value if upstream_model is None else upstream_model,
        api_key=SecretStr(credential.api_key),
        base_url=credential.base_url or OPENAI_DEFAULT_BASE_URL,
        # GPT-6 and GPT-5 reasoning models reject or ignore sampling controls;
        # omit them explicitly rather than relying on provider-specific SDK cleanup.
        temperature=(None if model in OPENAI_NO_SAMPLING_MODELS else temperature),
        streaming=True,
        stream_usage=True,
        max_tokens=max_tokens,
        **client_kwargs,  # type: ignore[arg-type]
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
    conversation_id: str | None = None,
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
        conversation_id=conversation_id,
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
    conversation_id: str | None = None,
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
            conversation_id=conversation_id,
        )

        tools = await get_agent_tools()

        logger.info(
            "Available tools: %s",
            ", ".join(tool.name for tool in tools) if tools else "None",
        )

        agent = create_agent(llm, tools, middleware=agent_middleware())

        messages = build_agent_messages(system_prompt, history or [], user_prompt)

        return StreamingResponse(
            create_agent_token_generator(agent, messages, observation),
            media_type="text/event-stream",
        )

    # ruff: ignore[BLE001] -- agent setup falls back to the regular provider stream
    except Exception as exc:
        logger.warning(
            "OpenAI agent setup failed; using regular response model=%s error_type=%s",
            model.value,
            type(exc).__name__,
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
            reasoning=reasoning,
            credential=credential,
            upstream_model=upstream_model,
            conversation_id=conversation_id,
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
    conversation_id: str | None = None,
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
            conversation_id=conversation_id,
            disable_reasoning=True,
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
            "OpenAI query transformation failed; using original query model=%s error_type=%s",
            model.value,
            type(exc).__name__,
        )

        return query
