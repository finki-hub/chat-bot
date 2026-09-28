from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from posthog import Posthog

from app.utils import posthog_client
from app.utils.settings import Settings


@pytest.fixture
def client(monkeypatch):
    client = create_autospec(Posthog, instance=True)
    monkeypatch.setattr(posthog_client._state, "client", client)
    monkeypatch.setattr(posthog_client._state, "revisions", {})
    return client


def test_exception_payload_is_metadata_only(client):
    sentinel = "sk-secret person@example.test private prompt answer https://host/path?key=secret"
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": sentinel,
            "headers": [],
            "route": SimpleNamespace(path="/chat/{id}"),
        }
    )

    def fail():
        raise ValueError(sentinel)

    try:
        fail()
    except ValueError as cause:
        exception = RuntimeError(sentinel)
        exception.__cause__ = cause
        posthog_client.capture_request_exception(request, exception)
    client.capture_exception.assert_not_called()
    call = client.capture.call_args.kwargs
    assert call["event"] == "$exception"
    assert call["properties"]["$exception_list"] == [
        {"type": "RuntimeError", "value": "Redacted"}
    ]
    assert call["properties"]["path"] == "/chat/{id}"
    assert sentinel not in repr(call)
    assert "stacktrace" not in repr(call)


def test_unmatched_exception_route_never_uses_raw_path(client):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/private@example.test",
            "headers": [],
        }
    )
    posthog_client.capture_request_exception(request, Exception("secret"))
    assert client.capture.call_args.kwargs["properties"]["path"] == "unmatched"


def test_request_tracking_uses_templates_and_bounds_unknown_routes(client):
    app = FastAPI()
    posthog_client.register_request_middleware(app)

    @app.get("/items/{item_id}")
    def item(item_id: str):
        return {}

    http = TestClient(app)
    http.get("/items/private@example.test?key=sk-secret")
    assert client.capture.call_args.kwargs["properties"]["route"] == "/items/{item_id}"
    http.get("/unknown/private@example.test?key=sk-secret")
    assert client.capture.call_args.kwargs["properties"]["route"] == "unmatched"
    assert "private@example.test" not in repr(client.capture.call_args_list)
    assert "sk-secret" not in repr(client.capture.call_args_list)


@pytest.mark.parametrize(
    "revision", ["", "A" * 40, "a" * 39, "a" * 41, "not-a-sha", " a" * 20]
)
def test_invalid_revisions_and_caller_spoofs_are_omitted(client, revision):
    posthog_client.init_posthog(
        Settings(
            APP_REVISION=revision,
            RAG_SYNC_EXPECTED_SOURCE_COMMIT=revision,
            POSTHOG_KEY="",
        )
    )
    posthog_client.capture(
        "test",
        "event",
        {
            "app_revision": "b" * 40,
            "document_corpus_revision": "c" * 40,
            "corpus_revision": "d" * 40,
        },
    )
    assert client.capture.call_args.kwargs["properties"] == {"service": "chat-bot-api"}


def test_trusted_revisions_override_caller_and_do_not_claim_faq_snapshot(client):
    posthog_client.init_posthog(
        Settings(
            APP_REVISION="a" * 40,
            RAG_SYNC_EXPECTED_SOURCE_COMMIT="b" * 40,
            POSTHOG_KEY="",
        )
    )
    posthog_client.capture(
        "test",
        "event",
        {"app_revision": "c" * 40, "document_corpus_revision": "d" * 40},
    )
    assert client.capture.call_args.kwargs["properties"] == {
        "service": "chat-bot-api",
        "app_revision": "a" * 40,
        "document_corpus_revision": "b" * 40,
    }


def test_capture_failures_do_not_interrupt_requests_and_shutdown_flushes(client):
    client.capture.side_effect = RuntimeError("offline")
    posthog_client.capture_exception(RuntimeError("private"))
    posthog_client.shutdown_posthog()
    client.flush.assert_called_once()
    client.shutdown.assert_called_once()
