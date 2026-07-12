import asyncio
import logging
from collections.abc import Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.llms.agents import (
    StreamObservation,
    capture_model_fallback,
    create_agent_token_generator,
    stream_sync_gen_as_sse,
)
from app.llms.models import ChatModel, Model, model_id
from app.llms.prompts import build_agent_messages, stitch_conversation
from app.llms.provider_credentials import require_provider_credential
from app.llms.tools import get_agent_tools
from app.schemas.chat_credentials import OLLAMA_DEFAULT_BASE_URL, ChatCredentialSecret

from .ollama_catalog import fetch_ollama_catalog

logger = logging.getLogger(__name__)

type _OllamaClientKwargs = dict[str, dict[str, str]]

__all__ = ("fetch_ollama_catalog",)


def _connection(
    credential: ChatCredentialSecret | None,
) -> tuple[str, _OllamaClientKwargs]:
    credential = require_provider_credential("ollama", credential)
    return (
        credential.base_url or OLLAMA_DEFAULT_BASE_URL,
        {"headers": {"Authorization": f"Bearer {credential.api_key}"}},
    )


def get_embedder(
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> OllamaEmbeddings:
    """
    Return a user-scoped OllamaEmbeddings instance for the specified model.
    """
    base_url, client_kwargs = _connection(credential)
    return OllamaEmbeddings(
        model=model.value,
        base_url=base_url,
        client_kwargs=client_kwargs,
    )


def get_llm(
    model: ChatModel,
    temperature: float,
    top_p: float,
    max_tokens: int,
    *,
    reasoning: bool = False,
    credential: ChatCredentialSecret | None = None,
) -> ChatOllama:
    base_url, client_kwargs = _connection(credential)
    return ChatOllama(
        model=model_id(model),
        base_url=base_url,
        client_kwargs=client_kwargs,
        temperature=temperature,
        top_p=top_p,
        num_predict=max_tokens,
        reasoning=reasoning,
    )


@overload
async def generate_ollama_embeddings(
    text: str,
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> list[float]: ...


@overload
async def generate_ollama_embeddings(
    text: list[str],
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> list[list[float]]: ...


async def generate_ollama_embeddings(
    text: str | list[str],
    model: Model,
    credential: ChatCredentialSecret | None = None,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings for the given text using the specified Ollama model.
    This function runs the embedding generation in a separate thread to avoid blocking the event loop.
    """
    logger.info(
        "Generating Ollama embeddings for text with length '%s' with model: %s",
        len(text) if isinstance(text, str) else sum(len(t) for t in text),
        model.value,
    )

    emb = get_embedder(model, credential)

    if isinstance(text, str):
        return await asyncio.to_thread(emb.embed_query, text)

    return await asyncio.to_thread(emb.embed_documents, text)


def stream_ollama_response(
    user_prompt: str,
    model: ChatModel,
    *,
    system_prompt: str,
    history: list[BaseMessage] | None = None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool = False,
    credential: ChatCredentialSecret | None = None,
) -> StreamingResponse:
    """
    Stream a response from the specified Ollama model using the provided user prompt and system prompt.
    This function constructs the full prompt by stitching the system prompt, prior
    conversation turns and the user prompt together, initializes the LLM client,
    and streams the response as an async generator.
    The response is formatted as Server-Sent Events (SSE) for real-time updates.
    """
    logger.info(
        "Streaming Ollama response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model_id(model),
    )

    llm = get_llm(
        model,
        temperature,
        top_p,
        max_tokens,
        reasoning=reasoning,
        credential=credential,
    )
    full_prompt = stitch_conversation(system_prompt, history or [], user_prompt)

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(full_prompt):
            yield str(chunk.content)

    return stream_sync_gen_as_sse(sync_token_gen())


async def transform_query_with_ollama(
    query: str,
    model: ChatModel,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    credential: ChatCredentialSecret | None = None,
) -> str:
    """
    Transform a query using the specified Ollama model and system prompt.
    If the transformation fails, return the original query.
    """

    logger.info(
        "Transforming query with model=%s query_len=%d",
        model_id(model),
        len(query),
    )

    try:
        llm = get_llm(
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
        return str(response.content).strip()
    except Exception:
        logger.exception("Query transformation failed: %s. Using original query.")

        return query


async def stream_ollama_agent_response(
    user_prompt: str,
    model: ChatModel,
    *,
    system_prompt: str,
    history: list[BaseMessage] | None = None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool = False,
    observation: StreamObservation | None = None,
    credential: ChatCredentialSecret | None = None,
) -> StreamingResponse:
    """
    Stream a response from an Ollama agent with MCP tools.
    Falls back to regular response if MCP unavailable.
    """
    logger.info(
        "Streaming Ollama agent response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model_id(model),
    )

    try:
        llm = get_llm(
            model,
            temperature,
            top_p,
            max_tokens,
            reasoning=reasoning,
            credential=credential,
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
            "Failed to stream Ollama agent response. Falling back to regular response",
        )
        capture_model_fallback(
            observation,
            from_model=model_id(model),
            to_model=model_id(model),
            reason="agent_setup_failed",
        )

        return stream_ollama_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            credential=credential,
        )
