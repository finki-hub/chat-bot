from collections.abc import AsyncGenerator
from contextlib import ExitStack

import anyio
import httpx2
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI

from app.llms import openai as openai_module
from app.llms.models import Model
from app.llms.openai import get_openai_llm
from app.schemas.chat_credentials import ChatCredentialSecret


def test_sponsored_go_headers_are_request_scoped_and_not_added_to_byok() -> None:
    with ExitStack() as stack:
        clients: list[ChatOpenAI] = []
        for conversation_id in (
            "018f0f36-2b1d-7cc0-a50b-5f2d90c91d21",
            "018f0f36-2b1d-7cc0-a50b-5f2d90c91d22",
        ):
            llm = get_openai_llm(
                Model.GPT_6_LUNA,
                temperature=0.3,
                top_p=1.0,
                max_tokens=128,
                credential=ChatCredentialSecret(
                    provider="openai",
                    api_key="test-sponsored-key",
                    base_url="https://opencode.ai/inference/go/v1",
                ),
                conversation_id=conversation_id,
            )
            assert isinstance(llm, ChatOpenAI)
            clients.append(llm)
            stack.callback(llm.root_client.close)
            headers = llm.root_client.default_headers
            assert headers["User-Agent"] == "finki-hub/1.0"
            assert headers["x-opencode-session"] == conversation_id

        assert clients[0].root_client is not clients[1].root_client

        byok = get_openai_llm(
            Model.GPT_6_LUNA,
            temperature=0.3,
            top_p=1.0,
            max_tokens=128,
            credential=ChatCredentialSecret(
                provider="openai",
                api_key="test-byok-key",
                base_url="https://opencode.ai/inference/go/v1",
            ),
        )
        stack.callback(byok.root_client.close)
        assert "x-opencode-session" not in byok.root_client.default_headers


@pytest.mark.parametrize(
    ("conversation_id", "expected_conversation_id"),
    [
        (
            "018f0f36-2b1d-7cc0-a50b-5f2d90c91d21",
            "018f0f36-2b1d-7cc0-a50b-5f2d90c91d21",
        ),
        (
            "018f0f36-2b1d-7cc0-a50b-5f2d90c91d22",
            "018f0f36-2b1d-7cc0-a50b-5f2d90c91d22",
        ),
    ],
)
def test_go_sdk_sends_headers_on_http_request(
    monkeypatch: pytest.MonkeyPatch,
    conversation_id: str,
    expected_conversation_id: str,
) -> None:
    llm = get_openai_llm(
        Model.GPT_6_LUNA,
        temperature=0.3,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(
            provider="openai",
            api_key="test-sponsored-key",
            base_url="https://opencode.ai/inference/go/v1",
        ),
        conversation_id=conversation_id,
    )
    requests: list[httpx2.Request] = []

    def capture_send(
        client: httpx2.Client,
        request: httpx2.Request,
        **kwargs: object,
    ) -> httpx2.Response:
        del client, kwargs
        requests.append(request)
        return httpx2.Response(
            200,
            json={"id": "response-test", "object": "response", "output": []},
            request=request,
        )

    monkeypatch.setattr(httpx2.Client, "send", capture_send)
    try:
        llm.root_client.responses.create(
            model="gpt-6-luna",
            input="hello",
        )
    finally:
        llm.root_client.close()

    assert len(requests) == 1
    request = requests[0]
    assert request.headers["User-Agent"] == "finki-hub/1.0"
    assert request.headers["x-opencode-session"] == expected_conversation_id
    assert request.url.host == "opencode.ai"
    assert b"x-opencode-session" not in request.content

    byok = get_openai_llm(
        Model.GPT_6_LUNA,
        temperature=0.3,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(
            provider="openai",
            api_key="test-byok-key",
            base_url="https://opencode.ai/inference/go/v1",
        ),
    )
    try:
        byok.root_client.responses.create(model="gpt-6-luna", input="hello")
    finally:
        byok.root_client.close()

    byok_request = requests[-1]
    assert "x-opencode-session" not in byok_request.headers
    assert byok_request.headers["User-Agent"] != "finki-hub/1.0"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://opencode.ai/inference/other/v1",
        "https://opencode.ai.evil.example/inference/go/v1",
        "https://not-opencode.ai/inference/go/v1",
    ],
)
def test_lookalike_endpoints_do_not_receive_go_headers(base_url: str) -> None:
    llm = get_openai_llm(
        Model.GPT_6_LUNA,
        temperature=0.3,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(
            provider="openai",
            api_key="test-provider-key",
            base_url=base_url,
        ),
        conversation_id="018f0f36-2b1d-7cc0-a50b-5f2d90c91d21",
    )
    try:
        assert "x-opencode-session" not in llm.root_client.default_headers
        assert llm.root_client.default_headers["User-Agent"] != "finki-hub/1.0"
    finally:
        llm.root_client.close()


@pytest.mark.parametrize("force_agent_fallback", [False, True])
def test_go_headers_survive_agent_stream_and_setup_fallback(
    monkeypatch: pytest.MonkeyPatch,
    force_agent_fallback: bool,
) -> None:
    conversation_id = "018f0f36-2b1d-7cc0-a50b-5f2d90c91d21"
    requests: list[httpx2.Request] = []
    sse = (
        b'event: response.created\ndata: {"type":"response.created",'
        b'"response":{"id":"resp_1","object":"response",'
        b'"created_at":1,"status":"in_progress","model":"gpt-6-luna",'
        b'"output":[]}}\n\n'
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta",'
        b'"delta":"hello","item_id":"msg_1","output_index":0,'
        b'"content_index":0}\n\n'
        b'event: response.completed\ndata: {"type":"response.completed",'
        b'"response":{"id":"resp_1","object":"response",'
        b'"created_at":1,"status":"completed","model":"gpt-6-luna",'
        b'"output":[],"usage":{"input_tokens":1,"output_tokens":1,'
        b'"total_tokens":2}}}\n\n'
    )

    def capture_send(
        client: httpx2.Client,
        request: httpx2.Request,
        **kwargs: object,
    ) -> httpx2.Response:
        del client, kwargs
        requests.append(request)
        return httpx2.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    monkeypatch.setattr(httpx2.Client, "send", capture_send)

    async def capture_async_send(
        client: httpx2.AsyncClient,
        request: httpx2.Request,
        **kwargs: object,
    ) -> httpx2.Response:
        del client, kwargs
        requests.append(request)
        return httpx2.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    monkeypatch.setattr(httpx2.AsyncClient, "send", capture_async_send)

    async def run() -> list[str]:
        if force_agent_fallback:

            async def unavailable_tools() -> list[object]:
                raise RuntimeError("mock setup failure")

            monkeypatch.setattr(openai_module, "get_agent_tools", unavailable_tools)
        else:

            class MockAgent:
                def __init__(self, llm: ChatOpenAI) -> None:
                    self.llm = llm

                async def astream_events(
                    self,
                    input_data: object,
                    config: object,
                    *,
                    version: str,
                ) -> AsyncGenerator[dict[str, object]]:
                    del input_data, config, version
                    async for chunk in self.llm.astream(
                        [HumanMessage(content="hello")]
                    ):
                        assert isinstance(chunk, AIMessageChunk)
                        yield {
                            "event": "on_chat_model_stream",
                            "run_id": "test-run",
                            "data": {"chunk": chunk},
                        }

            async def no_tools() -> list[object]:
                return []

            monkeypatch.setattr(openai_module, "get_agent_tools", no_tools)
            monkeypatch.setattr(
                openai_module,
                "create_agent",
                lambda llm, tools: MockAgent(llm),
            )

        response = await openai_module.stream_openai_agent_response(
            "hello",
            Model.GPT_6_LUNA,
            system_prompt="test",
            temperature=0.3,
            top_p=1.0,
            max_tokens=128,
            credential=ChatCredentialSecret(
                provider="openai",
                api_key="test-sponsored-key",
                base_url="https://opencode.ai/inference/go/v1",
            ),
            conversation_id=conversation_id,
        )

        return [str(chunk) async for chunk in response.body_iterator]

    chunks = anyio.run(run)

    assert requests
    assert all(request.headers["User-Agent"] == "finki-hub/1.0" for request in requests)
    assert all(
        request.headers["x-opencode-session"] == conversation_id for request in requests
    )
    assert any("hello" in chunk for chunk in chunks), chunks
