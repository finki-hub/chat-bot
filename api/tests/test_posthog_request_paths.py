import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.utils import posthog_client


def test_exception_capture_uses_matched_route_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a token-bearing route and an active exception telemetry sink.
    captured: list[dict[str, str]] = []

    def capture_exception(
        exc: BaseException,
        distinct_id: str = "server",
        properties: dict[str, str] | None = None,
    ) -> None:
        assert isinstance(exc, RuntimeError)
        assert distinct_id == "server"
        assert properties is not None
        captured.append(properties)

    monkeypatch.setattr(posthog_client, "capture_exception", capture_exception)
    app = FastAPI()

    @app.get("/chat/state/shared/{share_token}")
    async def load_shared(share_token: str) -> None:
        assert share_token
        raise RuntimeError

    @app.exception_handler(Exception)
    async def handle_exception(request: Request, exc: Exception) -> JSONResponse:
        posthog_client.capture_request_exception(request, exc)
        return JSONResponse(status_code=500, content={"detail": "failed"})

    # When: an unexpected failure occurs while resolving a bearer share token.
    response = TestClient(app, raise_server_exceptions=False).get(
        "/chat/state/shared/secret-bearer-token",
    )

    # Then: telemetry identifies the route without retaining the token.
    assert response.status_code == 500
    assert captured == [
        {
            "method": "GET",
            "path": "/chat/state/shared/{share_token}",
        },
    ]
