import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from datetime import datetime
from typing import assert_never

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, BaseMessage
from langgraph.graph.state import CompiledStateGraph

from app.llms.retrieval_result import RetrievalSourcePayload
from app.schemas.sponsored_access import SafeErrorDetails, SponsoredErrorCode
from app.utils.posthog_client import capture

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


def status_event(*, stage: str, tool: str | None = None) -> str:
    payload: dict[str, object] = {
        "state": "tool_call",
        "stage": stage,
        "label": _STATUS_LABEL,
    }
    if tool:
        payload["tool"] = tool
    return _sse("status", payload)


def error_event(code: str, message: str, **details: str) -> str:
    return _sse("error", {"code": code, "message": message, **details})


def sponsored_error_event(
    code: SponsoredErrorCode,
    message: str,
    *,
    resets_at: datetime | None = None,
) -> str:
    """Emit a sponsored error while exposing only its approved reset metadata."""
    match code:
        case "free_quota_exhausted":
            reset = SafeErrorDetails(resets_at=resets_at).sse_reset()
            if reset is None:
                return error_event(code, message)
            return error_event(code, message, resets_at=reset)
        case "free_tier_unavailable" | "sponsored_request_in_progress":
            return error_event(code, message)
        case unreachable:
            assert_never(unreachable)


def meta_event(payload: dict[str, object]) -> str:
    return _sse("meta", payload)


def sources_event(sources: list[RetrievalSourcePayload]) -> str:
    return _sse("sources", {"sources": sources})


RESET_EVENT = _sse("reset", {})
DONE_EVENT = _sse("done", {})


@dataclass
class StreamObservation:
    distinct_id: str
    response_id: str
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "input": 0,
            "output": 0,
            "total": 0,
            "cache_read": 0,
            "reasoning": 0,
        },
    )
    answer_chars: int = 0
    tool_call_count: int = 0
    tool_names: set[str] = field(default_factory=set)
    finish_reason: str = ""
    context_chars: int = 0


def _error_status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _content_len(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return len(content)
    return len(str(value))


def _tool_succeeded(output: object) -> bool:
    return getattr(output, "status", None) != "error"


def capture_tool_called(
    observation: StreamObservation | None,
    *,
    tool: str,
    latency_ms: float,
    success: bool,
    arg_len: int,
    result_len: int,
) -> None:
    if observation is None:
        return
    capture(
        observation.distinct_id,
        "tool_called",
        {
            "response_id": observation.response_id,
            "model": observation.model,
            "provider": observation.provider,
            "tool": tool,
            "latency_ms": round(latency_ms, 1),
            "success": success,
            "arg_len": arg_len,
            "result_len": result_len,
        },
    )


def capture_model_error(
    observation: StreamObservation | None,
    *,
    error_type: str,
    status_code: int | None = None,
) -> None:
    if observation is None:
        return
    capture(
        observation.distinct_id,
        "model_error",
        {
            "response_id": observation.response_id,
            "model": observation.model,
            "provider": observation.provider,
            "error_type": error_type,
            "status_code": status_code,
        },
    )


def capture_model_fallback(
    observation: StreamObservation | None,
    *,
    from_model: str,
    to_model: str,
    reason: str,
) -> None:
    if observation is None:
        return
    capture(
        observation.distinct_id,
        "model_fallback",
        {
            "response_id": observation.response_id,
            "from_model": from_model,
            "to_model": to_model,
            "reason": reason,
        },
    )


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
    input_details = metadata.get("input_token_details") or {}
    output_details = metadata.get("output_token_details") or {}
    usage["cache_read"] = usage.get("cache_read", 0) + (
        input_details.get("cache_read") or 0
    )
    usage["reasoning"] = usage.get("reasoning", 0) + (
        output_details.get("reasoning") or 0
    )


def _finish_reason(output: object) -> str:
    """A normalised finish/stop reason from a chat-model end message, or "" if absent."""
    metadata = getattr(output, "response_metadata", None)
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("finish_reason") or metadata.get("stop_reason") or ""
    mapping = {
        "length": "length",
        "max_tokens": "length",
        "model_length": "length",
        "content_filter": "content_filter",
        "stop": "stop",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_calls": "tool_calls",
        "tool_use": "tool_calls",
    }
    return mapping.get(str(raw), str(raw))


async def create_agent_token_generator(
    agent: CompiledStateGraph,
    messages: list[BaseMessage],
    observation: StreamObservation | None = None,
) -> AsyncGenerator[str]:
    """Stream an agent run as SSE, mapping `astream_events` onto the protocol: each
    `on_tool_start` becomes a `status`, and a `reset` precedes the answer so any
    pre-tool preamble is dropped."""
    streamed_text = False
    pending_reset = False
    usage = (
        observation.usage
        if observation is not None
        else {"input": 0, "output": 0, "total": 0}
    )
    tool_runs: dict[str, tuple[str, float, int]] = {}
    try:
        async for event in agent.astream_events(
            {"messages": messages},
            {"configurable": {"thread_id": "default"}},
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_tool_start":
                tool_runs[event["run_id"]] = (
                    event["name"],
                    time.perf_counter(),
                    _content_len(event["data"].get("input")),
                )
                yield status_event(stage="retrieve", tool=event["name"])
                pending_reset = True
                continue

            if kind in ("on_tool_end", "on_tool_error"):
                started = tool_runs.pop(event["run_id"], None)
                if started is not None:
                    name, start, arg_len = started
                    is_error = kind == "on_tool_error"
                    output = None if is_error else event["data"].get("output")
                    capture_tool_called(
                        observation,
                        tool=name,
                        latency_ms=(time.perf_counter() - start) * 1000.0,
                        success=not is_error and _tool_succeeded(output),
                        arg_len=arg_len,
                        result_len=_content_len(output),
                    )
                    if observation is not None:
                        observation.tool_call_count += 1
                        observation.tool_names.add(name)
                continue

            if kind == "on_chat_model_end":
                output = event["data"].get("output")
                _accumulate_usage(usage, output)
                if observation is not None:
                    reason = _finish_reason(output)
                    if reason:
                        observation.finish_reason = reason
                continue

            if kind != "on_chat_model_stream":
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
                # The client clears any pre-tool preamble on reset; mirror that so the
                # length reflects the final answer only.
                if observation is not None:
                    observation.answer_chars = 0
                yield RESET_EVENT

            streamed_text = True
            if observation is not None:
                observation.answer_chars += len(text)
            yield token_event(text)

        if not streamed_text:
            yield RESET_EVENT
            yield error_event("no_answer", _NO_ANSWER_MSG)
        if any(usage.values()):
            yield meta_event({"tokens": usage})
        yield DONE_EVENT

    except Exception as exc:
        logger.exception("Agent error occurred during streaming")
        capture_model_error(
            observation,
            error_type=type(exc).__name__,
            status_code=_error_status_code(exc),
        )
        # Tokens already streamed: a fresh "try again" would contradict the partial answer.
        if streamed_text:
            yield error_event("interrupted", _INTERRUPTED_MSG)
        else:
            yield RESET_EVENT
            yield error_event("agent_error", _ERROR_MSG)
        yield DONE_EVENT
