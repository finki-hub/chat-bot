import asyncio
import logging
from collections.abc import Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.llms.agents import create_agent_token_generator, stream_sync_gen_as_sse
from app.llms.mcp import get_mcp_tools
from app.llms.models import Model
from app.llms.prompts import build_agent_messages, stitch_conversation
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Model, temperature, top_p, max_tokens, reasoning -> LLM
ollama_chat_clients: dict[tuple[str, float, float, int, bool], ChatOllama] = {}
ollama_embedders: dict[Model, OllamaEmbeddings] = {}


def get_embedder(model: Model) -> OllamaEmbeddings:
    """
    Return a singleton OllamaEmbeddings instance for the specified model.
    If the model is not already in the cache, create a new instance.
    """
    if model not in ollama_embedders:
        ollama_embedders[model] = OllamaEmbeddings(
            model=model.value,
            base_url=settings.OLLAMA_URL,
        )

    return ollama_embedders[model]


def get_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
    *,
    reasoning: bool = False,
) -> ChatOllama:
    """
    Return a singleton ChatOllama instance for the specified model and parameters.
    If the model and parameters are not already in the cache, create a new instance.

    `reasoning` maps to Ollama's `think` mode: True makes reasoning-capable models (e.g.
    deepseek-r1) expose thoughts in `additional_kwargs["reasoning_content"]`; False disables
    it so a model that would otherwise emit raw inline `<think>` tags stays clean.
    """
    key = (model.value, temperature, top_p, max_tokens, reasoning)

    if key not in ollama_chat_clients:
        ollama_chat_clients[key] = ChatOllama(
            model=model.value,
            base_url=settings.OLLAMA_URL,
            temperature=temperature,
            top_p=top_p,
            num_predict=max_tokens,
            reasoning=reasoning,
        )

    return ollama_chat_clients[key]


@overload
async def generate_ollama_embeddings(
    text: str,
    model: Model,
) -> list[float]: ...


@overload
async def generate_ollama_embeddings(
    text: list[str],
    model: Model,
) -> list[list[float]]: ...


async def generate_ollama_embeddings(
    text: str | list[str],
    model: Model,
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

    emb = get_embedder(model)

    if isinstance(text, str):
        return await asyncio.to_thread(emb.embed_query, text)

    return await asyncio.to_thread(emb.embed_documents, text)


def stream_ollama_response(
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
    Stream a response from the specified Ollama model using the provided user prompt and system prompt.
    This function constructs the full prompt by stitching the system prompt, prior
    conversation turns and the user prompt together, initializes the LLM client,
    and streams the response as an async generator.
    The response is formatted as Server-Sent Events (SSE) for real-time updates.
    """
    logger.info(
        "Streaming Ollama response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    llm = get_llm(model, temperature, top_p, max_tokens, reasoning=reasoning)
    full_prompt = stitch_conversation(system_prompt, history or [], user_prompt)

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(full_prompt):
            yield str(chunk.content)

    return stream_sync_gen_as_sse(sync_token_gen())


async def transform_query_with_ollama(
    query: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """
    Transform a query using the specified Ollama model and system prompt.
    If the transformation fails, return the original query.
    """

    logger.info(
        "Transforming query: '%s' with model: %s",
        query,
        model,
    )

    try:
        llm = get_llm(model, temperature, top_p, max_tokens)

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
    Stream a response from an Ollama agent with MCP tools.
    Falls back to regular response if MCP unavailable.
    """
    logger.info(
        "Streaming Ollama agent response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    try:
        llm = get_llm(model, temperature, top_p, max_tokens, reasoning=reasoning)

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
            "Failed to stream Ollama agent response. Falling back to regular response",
        )

        # The non-agent fallback streams plain text (stream_sync_gen_as_sse has no
        # `thinking` channel), so it runs WITHOUT reasoning rather than enabling Ollama
        # think mode whose reasoning_content would be silently dropped.
        return stream_ollama_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
