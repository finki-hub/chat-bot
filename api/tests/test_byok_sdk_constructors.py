import asyncio
import socket
from contextlib import AsyncExitStack
from typing import Literal

import anthropic
import httpx
import httpx2
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from app.llms.anthropic import get_anthropic_llm
from app.llms.models import Model
from app.llms.openai import get_openai_llm
from app.schemas.chat_credentials import ChatCredentialSecret


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_real_byok_sdk_clients_initialize_independent_credentials(
    monkeypatch: pytest.MonkeyPatch,
    provider: Literal["openai", "anthropic"],
) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        pytest.fail("Constructor smoke tests must not perform network requests")

    for module in (httpx, httpx2):
        monkeypatch.setattr(module.Client, "send", reject_network)
        monkeypatch.setattr(module.AsyncClient, "send", reject_network)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "fake-deployment-key")
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "ANTHROPIC_API_URL"):
        monkeypatch.setenv(name, "https://deployment.invalid/v1")
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "false")

    async def run() -> None:
        # Windows creates a local socket pair when starting the event loop. Block
        # connections after loop setup, before constructing any provider clients.
        monkeypatch.setattr(socket.socket, "connect", reject_network)
        monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
        monkeypatch.setattr(socket, "getaddrinfo", reject_network)
        clients: list[
            tuple[
                openai.OpenAI | anthropic.Anthropic,
                openai.AsyncOpenAI | anthropic.AsyncAnthropic,
                ChatCredentialSecret,
            ]
        ] = []
        async with AsyncExitStack() as stack:
            for user in ("first", "second"):
                credential = ChatCredentialSecret(
                    provider=provider,
                    api_key=f"fake-{provider}-{user}-key",
                    base_url=f"https://{provider}-{user}.invalid/v1",
                )
                if provider == "openai":
                    llm = get_openai_llm(
                        Model.GPT_5_4_MINI,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=128,
                        credential=credential,
                    )
                    assert isinstance(llm, ChatOpenAI)
                    sync_client = llm.root_client
                    assert isinstance(sync_client, openai.OpenAI)
                    stack.callback(sync_client.close)
                    async_client = llm.root_async_client
                    assert isinstance(async_client, openai.AsyncOpenAI)
                else:
                    anthropic_llm = get_anthropic_llm(
                        Model.CLAUDE_HAIKU_4_5,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=128,
                        credential=credential,
                    )
                    assert isinstance(anthropic_llm, ChatAnthropic)
                    # Anthropic SDK clients are cached properties: access both to
                    # exercise their real constructors, not just the Pydantic model.
                    sync_client = anthropic_llm._client
                    assert isinstance(sync_client, anthropic.Anthropic)
                    stack.callback(sync_client.close)
                    async_client = anthropic_llm._async_client
                    assert isinstance(async_client, anthropic.AsyncAnthropic)
                stack.push_async_callback(async_client.close)
                clients.append((sync_client, async_client, credential))

            assert clients[0][0] is not clients[1][0]
            assert clients[0][1] is not clients[1][1]
            # Check both users after construction to catch shared credential state.
            for sync_client, async_client, credential in clients:
                for client in (sync_client, async_client):
                    assert client.api_key == credential.api_key
                    assert str(client.base_url).rstrip("/") == credential.base_url
                    assert not client.is_closed()

        for sync_client, async_client, _ in clients:
            assert sync_client.is_closed()
            assert async_client.is_closed()

    asyncio.run(run())
