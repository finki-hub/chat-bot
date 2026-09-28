import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from posthog import Posthog

from app.utils.settings import Settings


@pytest.fixture
def analytics(monkeypatch):
    # These tests exercise telemetry only; no torch import or model initialization.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    spec = importlib.util.spec_from_file_location(
        "gpu_analytics_under_test", Path(__file__).parents[1] / "app/utils/analytics.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module._state, "client", create_autospec(Posthog, instance=True)
    )
    monkeypatch.setattr(module._state, "revisions", {})
    return module


@pytest.mark.parametrize("revision", ["", "A" * 40, "a" * 39, "a" * 41, "invalid"])
def test_invalid_or_spoofed_revision_is_omitted(analytics, revision):
    analytics.init_analytics(Settings(APP_REVISION=revision, POSTHOG_KEY=""))
    analytics.capture(
        "gpu",
        "event",
        {
            "app_revision": "b" * 40,
            "corpus_revision": "c" * 40,
            "document_corpus_revision": "d" * 40,
        },
    )
    assert analytics._state.client.capture.call_args.kwargs["properties"] == {
        "service": "chat-bot-gpu-api"
    }


def test_exception_is_redacted_and_gpu_revision_is_trusted(analytics):
    analytics.init_analytics(Settings(APP_REVISION="a" * 40, POSTHOG_KEY=""))
    sentinel = "sk-secret person@example.test prompt answer https://private?key=secret"
    request = Request(
        {"type": "http", "method": "POST", "path": sentinel, "headers": []}
    )
    analytics.capture_request_exception(request, RuntimeError(sentinel))
    client = analytics._state.client
    client.capture_exception.assert_not_called()
    call = client.capture.call_args.kwargs
    assert call["event"] == "$exception"
    assert call["properties"]["path"] == "unmatched"
    assert call["properties"]["app_revision"] == "a" * 40
    assert call["properties"]["$exception_list"] == [
        {"type": "RuntimeError", "value": "Redacted"}
    ]
    assert sentinel not in repr(call)


def test_inference_trace_is_preserved_and_failures_are_nonblocking(analytics):
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/embed",
            "headers": [(b"x-response-id", b"opaque-response")],
        }
    )
    analytics.capture_chat_inference(
        request, stage="embed", ms=20, props={"model": "embedding-model", "count": 2}
    )
    client = analytics._state.client
    call = client.capture.call_args.kwargs
    assert call["event"] == "$ai_embedding"
    assert call["properties"]["$ai_trace_id"] == "opaque-response"
    client.capture.side_effect = RuntimeError("offline")
    analytics.capture_exception(RuntimeError("private"))
    analytics.shutdown_analytics()
    client.flush.assert_called_once()
    client.shutdown.assert_called_once()


def test_request_tracking_uses_templates_and_bounds_unknown_routes(analytics):
    app = FastAPI()
    analytics.register_request_middleware(app)

    @app.get("/items/{item_id}")
    def item(item_id: str):
        return {}

    http = TestClient(app)
    client = analytics._state.client
    http.get("/items/private@example.test?key=sk-secret")
    assert client.capture.call_args.kwargs["properties"]["route"] == "/items/{item_id}"
    http.get("/unknown/private@example.test?key=sk-secret")
    assert client.capture.call_args.kwargs["properties"]["route"] == "unmatched"
    assert "private@example.test" not in repr(client.capture.call_args_list)
    assert "sk-secret" not in repr(client.capture.call_args_list)
