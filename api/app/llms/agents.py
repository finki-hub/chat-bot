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
_NO_ANSWER_MSG = "Не успеав да составам одговор. Обидете се повторно."
_INTERRUPTED_MSG = "Одговорот не може да се доврши поради грешка."
_ERROR_MSG = "Се случи грешка при обработката на барањето. Обидете се повторно."


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def token_event(text: str) -> str:
    return _sse("token", {"text": text})


def thinking_event(text: str) -> str:
    return _sse("thinking", {"text": text})


def status_event(tool: str | None = None) -> str:
    payload: dict[str, object] = {"state": "tool_call", "label": _STATUS_LABEL}
    if tool:
        payload["tool"] = tool
    return _sse("status", payload)


def error_event(code: str, message: str) -> str:
    return _sse("error", {"code": code, "message": message})


def meta_event(payload: dict[str, object]) -> str:
    return _sse("meta", payload)


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


def content_to_text(content: object) -> str:
    """Plain text of a message's content; non-text blocks (tool calls, reasoning) yield ''."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            parts.append(str(part.get("text", "")))
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)


_MAX_THINKING_BUDGET = 2048


def thinking_budget(max_tokens: int) -> tuple[int, int]:
    """A thinking budget (at most half of ``max_tokens``) and that cap returned unchanged."""
    budget = min(_MAX_THINKING_BUDGET, max_tokens // 2)
    return budget, max_tokens


def _chunk_text(message: AIMessageChunk) -> str:
    """The plain text of an AIMessageChunk; '' for tool-call/non-text content blocks."""
    return content_to_text(message.content)


def _chunk_reasoning(message: AIMessageChunk) -> str:
    """Reasoning text in an AIMessageChunk; '' if none."""
    parts: list[str] = []

    raw = message.content
    if isinstance(raw, list):
        for part in raw:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "thinking":
                parts.append(str(part.get("thinking", "")))
            elif kind == "reasoning":
                parts.extend(
                    str(summary.get("text", ""))
                    for summary in part.get("summary", []) or []
                    if isinstance(summary, dict)
                )

    extra = message.additional_kwargs.get("reasoning_content")
    if isinstance(extra, str):
        parts.append(extra)

    return "".join(parts)


def _accumulate_usage(usage: dict[str, int], output: object) -> None:
    """Fold one ``on_chat_model_end`` message's token counts into ``usage``.

    Accumulates across the answer and any tool-loop turns; providers that omit
    ``usage_metadata`` (e.g. the GPU-API path) add nothing.
    """
    metadata = getattr(output, "usage_metadata", None)
    if not metadata:
        return
    usage["input"] += metadata.get("input_tokens") or 0
    usage["output"] += metadata.get("output_tokens") or 0
    # Derive total from the parts so the shown counts always reconcile.
    usage["total"] = usage["input"] + usage["output"]


async def create_agent_token_generator(
    agent: CompiledStateGraph,
    messages: list[BaseMessage],
) -> AsyncGenerator[str]:
    """Stream an agent run as SSE, mapping `astream_events` onto the protocol: each
    `on_tool_start` becomes a `status`, and a `reset` precedes the answer so any
    pre-tool preamble is dropped."""
    streamed_text = False
    pending_reset = False
    usage = {"input": 0, "output": 0, "total": 0}
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

            if event["event"] == "on_chat_model_end":
                _accumulate_usage(usage, event["data"].get("output"))
                continue

            if event["event"] != "on_chat_model_stream":
                continue

            chunk = event["data"].get("chunk")
            if not isinstance(chunk, AIMessageChunk):
                continue

            reasoning = _chunk_reasoning(chunk)
            if reasoning:
                yield thinking_event(reasoning)

            text = _chunk_text(chunk)
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
        if any(usage.values()):
            yield meta_event({"tokens": usage})
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
