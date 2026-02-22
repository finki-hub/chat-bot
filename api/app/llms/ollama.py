import asyncio
import logging
from collections.abc import Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.llms.agents import create_agent_token_generator, stream_sync_gen_as_sse
from app.llms.mcp import get_mcp_tools
from app.llms.models import Model
from app.llms.prompts import stitch_system_user
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Model, temperature, top_p, max_tokens -> LLM
ollama_chat_clients: dict[tuple[str, float, float, int], ChatOllama] = {}
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
) -> ChatOllama:
    """
    Return a singleton ChatOllama instance for the specified model and sampling parameters.
    If the model and parameters are not already in the cache, create a new instance.
    """
    key = (model.value, temperature, top_p, max_tokens)

    if key not in ollama_chat_clients:
        ollama_chat_clients[key] = ChatOllama(
            model=model.value,
            base_url=settings.OLLAMA_URL,
            temperature=temperature,
            top_p=top_p,
            num_predict=max_tokens,
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
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from the specified Ollama model using the provided user prompt and system prompt.
    This function constructs the full prompt by stitching the system and user prompts together,
    initializes the LLM client, and streams the response as an async generator.
    The response is formatted as Server-Sent Events (SSE) for real-time updates.
    """
    logger.info(
        "Streaming Ollama response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    llm = get_llm(model, temperature, top_p, max_tokens)
    full_prompt = stitch_system_user(system_prompt, user_prompt)

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(full_prompt):
            yield str(chunk.content)

    return stream_sync_gen_as_sse(sync_token_gen())


async def stream_ollama_agent_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
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
        llm = get_llm(model, temperature, top_p, max_tokens)

        tools = await get_mcp_tools()

        logger.info(
            "Available tools: %s",
            ", ".join(tool.name for tool in tools) if tools else "None",
        )

        agent = create_agent(llm, tools)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        return StreamingResponse(
            create_agent_token_generator(agent, messages),
            media_type="text/event-stream",
        )

    except Exception:
        logger.exception(
            "Failed to stream Ollama agent response. Falling back to regular response",
        )

        return stream_ollama_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
