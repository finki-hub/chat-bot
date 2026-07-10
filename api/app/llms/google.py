import asyncio
import logging
from collections.abc import Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from app.llms.agents import (
    StreamObservation,
    capture_model_fallback,
    content_to_text,
    create_agent_token_generator,
    stream_sync_gen_as_sse,
    thinking_budget,
)
from app.llms.models import Model
from app.llms.prompts import build_agent_messages
from app.llms.provider_credentials import require_provider_credential
from app.llms.tools import get_agent_tools
from app.schemas.chat_credentials import ChatCredentialSecret

logger = logging.getLogger(__name__)


def get_google_embedder(
    model: Model,
    *,
    is_document: bool = False,
    credential: ChatCredentialSecret | None = None,
) -> GoogleGenerativeAIEmbeddings:
    task_type = "RETRIEVAL_DOCUMENT" if is_document else "RETRIEVAL_QUERY"
    credential = require_provider_credential("google", credential)
    return GoogleGenerativeAIEmbeddings(
        model=model.value,
        api_key=SecretStr(credential.api_key),
        task_type=task_type,
        base_url=credential.base_url or None,
    )


def get_google_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
    *,
    reasoning: bool = False,
    credential: ChatCredentialSecret | None = None,
) -> ChatGoogleGenerativeAI:
    """
    Return a user-scoped ChatGoogleGenerativeAI instance for the specified model.

    When `reasoning` is on, `include_thoughts=True` returns the model's thoughts; Gemini 3
    uses `thinking_level`, while Gemini 2.5 uses a `thinking_budget` token cap.
    """
    client_kwargs: dict[str, object] = {}
    max_output = max_tokens
    if reasoning:
        client_kwargs["include_thoughts"] = True
        budget, max_output = thinking_budget(max_tokens)
        if model == Model.GEMINI_3_FLASH_PREVIEW:
            client_kwargs["thinking_level"] = "medium"
        else:
            client_kwargs["thinking_budget"] = budget
    credential = require_provider_credential("google", credential)
    return ChatGoogleGenerativeAI(
        model=model.value,
        google_api_key=credential.api_key,
        base_url=credential.base_url or None,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output,
        **client_kwargs,
    )


@overload
async def generate_google_embeddings(
    text: str,
    model: Model,
    *,
    is_document: bool = ...,
    credential: ChatCredentialSecret | None = None,
) -> list[float]: ...


@overload
async def generate_google_embeddings(
    text: list[str],
    model: Model,
    *,
    is_document: bool = ...,
    credential: ChatCredentialSecret | None = None,
) -> list[list[float]]: ...


async def generate_google_embeddings(
    text: str | list[str],
    model: Model,
    *,
    is_document: bool = False,
    credential: ChatCredentialSecret | None = None,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings for the given text using the specified Google model.
    This function runs the embedding generation in a separate thread to avoid
    blocking the event loop, mirroring the OpenAI implementation.
    """
    logger.info(
        "Generating Google embeddings for text with length '%s' with model: %s",
        len(text) if isinstance(text, str) else sum(len(t) for t in text),
        model.value,
    )

    emb = get_google_embedder(model, is_document=is_document, credential=credential)

    if isinstance(text, str):
        return await asyncio.to_thread(emb.embed_query, text)

    return await asyncio.to_thread(emb.embed_documents, text)


def stream_google_response(
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
) -> StreamingResponse:
    """
    Stream a response from the specified Google model using the provided prompts.
    The response is formatted as Server-Sent Events (SSE).
    This function is a direct parallel to stream_openai_response.
    """
    logger.info(
        "Streaming Google response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    llm = get_google_llm(
        model,
        temperature,
        top_p,
        max_tokens,
        reasoning=reasoning,
        credential=credential,
    )
    prompt_messages = build_agent_messages(system_prompt, history or [], user_prompt)

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(prompt_messages):
            yield content_to_text(chunk.content)

    return stream_sync_gen_as_sse(sync_token_gen())


async def transform_query_with_google(
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
    Transform a query using the specified Google model and system prompt.
    If the transformation fails, return the original query.
    """

    logger.info(
        "Transforming query with model=%s query_len=%d",
        model.value,
        len(query),
    )

    try:
        llm = get_google_llm(
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
        logger.exception("Query transformation failed: %s. Using original query.")

        return query


async def stream_google_agent_response(
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
) -> StreamingResponse:
    """
    Stream a response from a Google agent with MCP tools.
    Falls back to regular response if MCP unavailable.
    """
    logger.info(
        "Streaming Google agent response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    try:
        llm = get_google_llm(
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
            "Failed to stream Google agent response. Falling back to regular response",
        )
        capture_model_fallback(
            observation,
            from_model=model.value,
            to_model=model.value,
            reason="agent_setup_failed",
        )

        return stream_google_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            credential=credential,
        )
