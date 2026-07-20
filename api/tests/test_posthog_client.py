from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from posthog import Posthog

from app.utils import posthog_client


def test_capture_exception_forwards_the_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an initialized PostHog client and an exception with debugging context.
    client = create_autospec(Posthog, instance=True)
    monkeypatch.setattr(posthog_client, "_state", SimpleNamespace(client=client))
    exception = RuntimeError("database unavailable")

    # When: the application reports the handled exception.
    posthog_client.capture_exception(
        exception,
        distinct_id="user-123",
        properties={"path": "/chat/{chat_id}"},
    )

    # Then: the SDK receives the original exception and its safe request metadata.
    client.capture_exception.assert_called_once_with(
        exception,
        distinct_id="user-123",
        properties={
            "service": "chat-bot-api",
            "path": "/chat/{chat_id}",
        },
    )
