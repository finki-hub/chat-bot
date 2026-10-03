import asyncio
import os
import threading
from collections.abc import AsyncGenerator, Generator
from typing import cast

import anyio
import pytest
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk

from app.llms import openrouter
from app.llms.models import Model
from app.schemas.chat_credentials import ChatCredentialSecret


def test_openrouter_byok_client_does_not_inherit_deployment_base_url(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://deployment.example/v1")

    llm = openrouter.get_openrouter_llm(
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openrouter", api_key="user-key"),
    )

    assert llm.openrouter_api_base is None
    assert os.environ["OPENROUTER_API_BASE"] == "https://deployment.example/v1"


def test_openrouter_regular_stream_preserves_reasoning_events(monkeypatch) -> None:
    class FakeOpenRouter:
        def stream(self, _messages) -> Generator[AIMessageChunk]:
            yield AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "thinking"},
                    ],
                },
            )
            yield AIMessageChunk(content="answer")

    monkeypatch.setattr(
        openrouter,
        "get_openrouter_llm",
        lambda *_args, **_kwargs: FakeOpenRouter(),
    )
    response = openrouter.stream_openrouter_response(
        "question",
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        system_prompt="system",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        reasoning=True,
        credential=ChatCredentialSecret(provider="openrouter", api_key="user-key"),
    )

    async def collect() -> str:
        chunks = [
            chunk if isinstance(chunk, str) else bytes(chunk).decode()
            async for chunk in response.body_iterator
        ]
        return "".join(chunks)

    body = anyio.run(collect)

    assert 'event: thinking\ndata: {"text": "thinking"}' in body
    assert 'event: token\ndata: {"text": "answer"}' in body


def test_openrouter_regular_stream_closes_sync_source_on_body_close(
    monkeypatch,
) -> None:
    closed = []

    def sync_stream() -> Generator[str]:
        try:
            yield "answer"
            raise AssertionError("Closing must not advance the provider")
        finally:
            closed.append(True)

    class FakeOpenRouter:
        def stream(self, _messages) -> Generator[str]:
            return sync_stream()

    monkeypatch.setattr(
        openrouter,
        "get_openrouter_llm",
        lambda *_args, **_kwargs: FakeOpenRouter(),
    )
    response = openrouter.stream_openrouter_response(
        "question",
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        system_prompt="system",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openrouter", api_key="user-key"),
    )

    async def close_after_first_token() -> None:
        body = cast("AsyncGenerator[str]", response.body_iterator)
        assert "answer" in str(await anext(body))
        await body.aclose()

    anyio.run(close_after_first_token)
    assert closed == [True]


def test_openrouter_sync_source_settles_blocked_next_before_close(monkeypatch) -> None:
    next_started = threading.Event()
    release_next = threading.Event()
    closed = []

    def sync_stream() -> Generator[str]:
        try:
            yield "answer"
            next_started.set()
            if not release_next.wait(timeout=2):
                raise AssertionError("blocked provider advance was not released")
            yield "discarded after cancellation"
        finally:
            closed.append(True)

    class FakeOpenRouter:
        def stream(self, _messages) -> Generator[str]:
            return sync_stream()

    monkeypatch.setattr(
        openrouter,
        "get_openrouter_llm",
        lambda *_args, **_kwargs: FakeOpenRouter(),
    )
    response = openrouter.stream_openrouter_response(
        "question",
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        system_prompt="system",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openrouter", api_key="user-key"),
    )

    async def cancel_during_blocked_next() -> None:
        body = cast("AsyncGenerator[str]", response.body_iterator)
        assert "answer" in str(await anext(body))
        pending = asyncio.create_task(anext(body))
        try:
            assert await asyncio.to_thread(next_started.wait, 2)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            release_next.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            release_next.set()
            if not pending.done():
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending

    anyio.run(cancel_during_blocked_next)
    assert closed == [True]


@pytest.mark.parametrize("close_via", ["cancel", "generator_exit", "normal"])
def test_openrouter_sync_cleanup_failure_keeps_primary_or_reports_without_one(
    monkeypatch, close_via
) -> None:
    cleanup_failure = RuntimeError("sync cleanup failed")
    closed = []

    def sync_stream() -> Generator[str]:
        try:
            yield "answer"
        finally:
            closed.append(True)
            raise cleanup_failure

    class FakeOpenRouter:
        def stream(self, _messages) -> Generator[str]:
            return sync_stream()

    monkeypatch.setattr(
        openrouter,
        "get_openrouter_llm",
        lambda *_args, **_kwargs: FakeOpenRouter(),
    )
    response = openrouter.stream_openrouter_response(
        "question",
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        system_prompt="system",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openrouter", api_key="user-key"),
    )

    async def consume() -> str:
        body = cast("AsyncGenerator[str]", response.body_iterator)
        assert "answer" in str(await anext(body))
        if close_via == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await body.athrow(asyncio.CancelledError())
            return ""
        if close_via == "generator_exit":
            await body.aclose()
            return ""
        return "".join([str(chunk) async for chunk in body])

    body = anyio.run(consume)
    assert closed == [True]
    if close_via == "normal":
        assert '"code": "interrupted"' in body


def test_openrouter_agent_fallback_preserves_upstream_model(monkeypatch) -> None:
    captured: list[str | None] = []

    async def fail_get_agent_tools() -> list[object]:
        msg = "agent tools unavailable"
        raise RuntimeError(msg)

    def fake_stream_openrouter_response(
        user_prompt: str,
        routed_model: Model,
        *,
        system_prompt: str,
        history=None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        reasoning: bool = False,
        credential: ChatCredentialSecret | None = None,
        upstream_model: str | None = None,
    ) -> StreamingResponse:
        captured.append(upstream_model)
        return StreamingResponse(iter(()), media_type="text/event-stream")

    monkeypatch.setattr(
        openrouter, "get_openrouter_llm", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(openrouter, "get_agent_tools", fail_get_agent_tools)
    monkeypatch.setattr(
        openrouter,
        "stream_openrouter_response",
        fake_stream_openrouter_response,
    )

    async def route() -> StreamingResponse:
        return await openrouter.stream_openrouter_agent_response(
            "question",
            Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
            system_prompt="system",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            credential=ChatCredentialSecret(
                provider="openrouter",
                api_key="sponsored-key",
            ),
            upstream_model="sponsored/upstream-model",
        )

    anyio.run(route)

    assert captured == ["sponsored/upstream-model"]
