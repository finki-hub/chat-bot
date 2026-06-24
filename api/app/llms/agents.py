import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Generator

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, BaseMessage
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_SENTINEL = object()

_STATUS_LABEL = "🔍 Пребарувам…"
_NO_ANSWER_MSG = "Не успеав да состават одговор. Обидете се повторно."
_INTERRUPTED_MSG = "Одговорот не може да се доврши поради грешка."
_ERROR_MSG = "Се случи грешка при обработката на барањето. Обидете се повторно."


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def token_event(text: str) -> str:
    return _sse("token", {"text": text})


def status_event(tool: str | None = None) -> str:
    payload: dict[str, object] = {"state": "tool_call", "label": _STATUS_LABEL}
    if tool:
        payload["tool"] = tool
    return _sse("status", payload)


def error_event(code: str, message: str) -> str:
    return _sse("error", {"code": code, "message": message})


RESET_EVENT = _sse("reset", {})
DONE_EVENT = _sse("done", {})


def stream_sync_gen_as_sse(gen: Generator[str]) -> StreamingResponse:
    """Wrap a synchronous token generator as a chat-SSE StreamingResponse."""

    async def async_token_gen() -> AsyncGenerator[str]:
        streamed = False
        try:
            while True:
                chunk = await asyncio.to_thread(next, gen, _SENTINEL)
                if chunk is _SENTINEL:
                    break
                text = str(chunk)
                if not text:
                    continue
                streamed = True
                yield token_event(text)
            if not streamed:
                yield error_event("no_answer", _NO_ANSWER_MSG)
            yield DONE_EVENT
        except Exception:
            logger.exception("Error while streaming the sync token generator")
            yield (
                error_event("interrupted", _INTERRUPTED_MSG)
                if streamed
                else error_event("agent_error", _ERROR_MSG)
            )
            yield DONE_EVENT

    return StreamingResponse(async_token_gen(), media_type="text/event-stream")


def _chunk_text(message: AIMessageChunk) -> str:
    """The plain text of an AIMessageChunk; '' for tool-call/non-text content blocks."""
    raw = message.content
    if isinstance(raw, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    return str(raw)


async def create_agent_token_generator(
    agent: CompiledStateGraph,
    messages: list[BaseMessage],
) -> AsyncGenerator[str]:
    """Stream an agent run as SSE, mapping `astream_events` onto the protocol: each
    `on_tool_start` becomes a `status`, and a `reset` precedes the answer so any
    pre-tool preamble is dropped."""
    streamed_text = False
    pending_reset = False
    try:
        async for event in agent.astream_events(
            {"messages": messages},
            {"configurable": {"thread_id": "default"}},
            version="v2",
        ):
            if event["event"] == "on_tool_start":
                yield status_event(event["name"])
                pending_reset = True
                continue

            if event["event"] != "on_chat_model_stream":
                continue

            chunk = event["data"].get("chunk")
            text = _chunk_text(chunk) if isinstance(chunk, AIMessageChunk) else ""
            if not text:
                continue

            if pending_reset:
                pending_reset = False
                yield RESET_EVENT

            streamed_text = True
            yield token_event(text)

        if not streamed_text:
            yield RESET_EVENT
            yield error_event("no_answer", _NO_ANSWER_MSG)
        yield DONE_EVENT

    except Exception:
        logger.exception("Agent error occurred during streaming")
        # Tokens already streamed: a fresh "try again" would contradict the partial answer.
        if streamed_text:
            yield error_event("interrupted", _INTERRUPTED_MSG)
        else:
            yield RESET_EVENT
            yield error_event("agent_error", _ERROR_MSG)
        yield DONE_EVENT
