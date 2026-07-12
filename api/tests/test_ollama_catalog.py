import json

import httpx
import pytest
from anyio import lowlevel

from app.llms import ollama
from app.schemas.chat_credentials import ChatCredentialSecret


@pytest.mark.anyio
async def test_ollama_catalog_discovers_completion_models_and_loaded_status(
    monkeypatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await lowlevel.checkpoint()
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "llama3.2:latest"},
                        {"name": "bge-m3:latest"},
                    ],
                },
            )
        if request.url.path == "/api/show":
            model = json.loads(request.content)["model"]
            capabilities = (
                ["completion"] if model == "llama3.2:latest" else ["embedding"]
            )
            return httpx.Response(200, json={"capabilities": capabilities})
        return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            "app.llms.ollama_catalog.get_http_client",
            lambda: client,
        )

        models = await ollama.fetch_ollama_catalog(
            ChatCredentialSecret(
                provider="ollama",
                api_key="ollama-user-key",
                base_url="https://ollama.example",
            ),
        )

    assert [model.id for model in models] == ["llama3.2:latest"]
    assert models[0].loaded is True


@pytest.mark.anyio
async def test_ollama_catalog_keeps_models_when_loaded_status_is_unavailable(
    monkeypatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await lowlevel.checkpoint()
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "llama3.2:latest", "capabilities": ["completion"]},
                    ],
                },
            )
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            "app.llms.ollama_catalog.get_http_client",
            lambda: client,
        )

        models = await ollama.fetch_ollama_catalog(
            ChatCredentialSecret(
                provider="ollama",
                api_key="ollama-user-key",
                base_url="https://ollama.example",
            ),
        )

    assert [model.id for model in models] == ["llama3.2:latest"]
    assert models[0].loaded is None


@pytest.mark.anyio
async def test_ollama_catalog_caps_capability_discovery_fanout(monkeypatch) -> None:
    show_requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal show_requests
        await lowlevel.checkpoint()
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": f"model-{index}:latest"} for index in range(101)
                    ],
                },
            )
        if request.url.path == "/api/show":
            show_requests += 1
            return httpx.Response(200, json={"capabilities": ["completion"]})
        return httpx.Response(200, json={"models": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            "app.llms.ollama_catalog.get_http_client",
            lambda: client,
        )

        models = await ollama.fetch_ollama_catalog(
            ChatCredentialSecret(
                provider="ollama",
                api_key="ollama-user-key",
                base_url="https://ollama.example",
            ),
        )

    assert len(models) == 100
    assert show_requests == 100
