import asyncio
import logging
from collections.abc import AsyncGenerator, Generator

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, BaseMessage
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_SENTINEL = object()


def stream_sync_gen_as_sse(gen: Generator[str]) -> StreamingResponse:
    """Wrap a synchronous token generator as a Server-Sent Events StreamingResponse."""

    async def async_token_gen() -> AsyncGenerator[str]:
        emitted = False
        try:
            while True:
                chunk = await asyncio.to_thread(next, gen, _SENTINEL)
                if chunk is _SENTINEL:
                    break
                preserved_chunk = str(chunk).replace("\n", "\\n")
                emitted = True
                yield f"data: {preserved_chunk}\n\n"
        except Exception:
            logger.exception("Error while streaming the sync token generator")
            if emitted:
                yield "data: (Прекин: одговорот не може да се доврши поради грешка.)\n\n"
            else:
                yield "data: Се случи грешка при обработката на барањето. Обидете се повторно.\n\n"

    return StreamingResponse(
        async_token_gen(),
        media_type="text/event-stream",
    )


async def create_agent_token_generator(
    agent: CompiledStateGraph,
    messages: list[BaseMessage],
) -> AsyncGenerator[str]:
    """Generate SSE tokens from an agent stream."""
    emitted = False
    try:
        async for message, _metadata in agent.astream(
            {"messages": messages},
            {"configurable": {"thread_id": "default"}},
            stream_mode="messages",
        ):
            if not isinstance(message, AIMessageChunk):
                continue
            raw = message.content
            if isinstance(raw, list):
                text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw
                )
            else:
                text = str(raw)
            if not text:
                continue
            preserved = text.replace("\n", "\\n")
            emitted = True
            yield f"data: {preserved}\n\n"

    except Exception:
        logger.exception("Agent error occurred during streaming")
        # Tokens already streamed: "try again" would contradict the partial answer.
        if emitted:
            yield "data: (Прекин: одговорот не може да се доврши поради грешка.)\n\n"
        else:
            yield "data: Се случи грешка при обработката на барањето. Обидете се повторно.\n\n"
