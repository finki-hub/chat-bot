from collections.abc import Callable

import pytest
import torch
from fastapi import FastAPI

from app import main


async def _run_in_current_task[ResultT](
    function: Callable[..., ResultT],
    *args: object,
) -> ResultT:
    return function(*args)


def _patch_lifespan_dependencies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main, "capture", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "init_analytics", lambda _settings: events.append("init"))
    monkeypatch.setattr(main, "shutdown_analytics", lambda: events.append("shutdown"))
    monkeypatch.setattr(main, "to_thread", _run_in_current_task)
    return events


@pytest.mark.anyio
async def test_gpu_lifespan_shuts_down_analytics_when_model_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _patch_lifespan_dependencies(monkeypatch)

    def fail_reranker(_model: str) -> None:
        raise RuntimeError("model startup failed")

    monkeypatch.setattr(main, "init_reranker", fail_reranker)
    monkeypatch.setattr(main, "init_bge_m3_embedder", lambda: None)

    with pytest.raises(RuntimeError, match="model startup failed"):
        async with main.lifespan(FastAPI()):
            pass

    assert events == ["init", "shutdown"]


@pytest.mark.anyio
async def test_gpu_lifespan_shuts_down_analytics_after_body_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setattr(main, "init_reranker", lambda _model: None)
    monkeypatch.setattr(main, "init_bge_m3_embedder", lambda: None)

    with pytest.raises(RuntimeError, match="body failed"):
        async with main.lifespan(FastAPI()):
            raise RuntimeError("body failed")

    assert events == ["init", "shutdown"]
