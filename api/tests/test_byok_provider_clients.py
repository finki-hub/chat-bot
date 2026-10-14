import anyio
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI
from openrouter.components import ChatRequest
from pydantic import SecretStr

from app.llms import anthropic, google, ollama, openai, openrouter
from app.llms.models import Model
from app.llms.provider_credentials import ProviderCredentialRequiredError
from app.schemas.chat_credentials import ChatCredentialSecret


def test_openai_byok_client_does_not_inherit_deployment_base_url(monkeypatch) -> None:
    captured_base_urls: list[str | None] = []

    class OpenAICapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(openai, "ChatOpenAI", OpenAICapturingClient)

    openai.get_openai_llm(
        Model.GPT_5_4_MINI,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openai", api_key="sk-user-key"),
    )

    assert captured_base_urls == [openai.OPENAI_DEFAULT_BASE_URL]


def test_gpt_6_astra_uses_chat_completions_effort_without_temperature(
    monkeypatch,
) -> None:
    captured: list[dict] = []

    class OpenAICapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(openai, "ChatOpenAI", OpenAICapturingClient)

    openai.get_openai_llm(
        Model.GPT_6_ASTRA,
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        reasoning=True,
        credential=ChatCredentialSecret(provider="openai", api_key="user-key"),
    )

    request = captured[0]
    assert request["use_responses_api"] is False
    assert request["reasoning_effort"] == "medium"
    assert request["temperature"] is None


@pytest.mark.parametrize(
    ("model", "upstream_model"),
    [
        (Model.GPT_6_1_SOL, None),
        (Model.GPT_6_1_SOL, "sponsored/sol"),
        (Model.GPT_6_LUNA, None),
        (Model.GPT_6_LUNA, "sponsored/luna"),
    ],
)
@pytest.mark.parametrize("reasoning", [False, True])
def test_gpt6_serialized_payload_omits_sampling_and_keeps_tools(
    model: Model,
    upstream_model: str | None,
    reasoning: bool,
) -> None:
    llm = openai.get_openai_llm(
        model,
        temperature=0.2,
        top_p=0.8,
        max_tokens=1024,
        reasoning=reasoning,
        credential=ChatCredentialSecret(provider="openai", api_key="fake-key"),
        upstream_model=upstream_model,
    )
    assert isinstance(llm, ChatOpenAI)
    bound = llm.bind_tools(
        [
            {
                "name": "lookup",
                "description": "Look up a record",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
    )

    assert isinstance(bound, RunnableBinding)
    payload = llm._get_request_payload(
        [HumanMessage(content="test")],
        **bound.kwargs,
    )

    assert payload["model"] == (upstream_model or model.value)
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert payload["tools"][0]["name"] == "lookup"
    if reasoning:
        assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}


def test_sponsored_openai_client_receives_upstream_model_and_token_cap(
    monkeypatch,
) -> None:
    captured: list[dict] = []

    class OpenAICapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(openai, "ChatOpenAI", OpenAICapturingClient)

    openai.get_openai_llm(
        Model.GPT_6_LUNA,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        credential=ChatCredentialSecret(
            provider="openai",
            api_key="sponsored-key",
            base_url="https://sponsored.example/v1",
        ),
        upstream_model="upstream-luna",
    )

    assert captured == [
        {
            "model": "upstream-luna",
            "api_key": SecretStr("sponsored-key"),
            "base_url": "https://sponsored.example/v1",
            "temperature": None,
            "streaming": True,
            "stream_usage": True,
            "max_tokens": 1024,
            "use_responses_api": True,
        },
    ]


def test_sponsored_google_client_receives_upstream_model_and_token_cap(
    monkeypatch,
) -> None:
    captured: list[dict] = []

    class GoogleCapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(google, "ChatGoogleGenerativeAI", GoogleCapturingClient)

    google.get_google_llm(
        Model.GEMINI_3_5_FLASH_LITE,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        credential=ChatCredentialSecret(
            provider="google",
            api_key="sponsored-key",
            base_url="https://sponsored.example/v1",
        ),
        upstream_model="upstream-luna",
    )

    assert captured == [
        {
            "model": "upstream-luna",
            "google_api_key": "sponsored-key",
            "base_url": "https://sponsored.example/v1",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 1024,
        },
    ]


def test_sponsored_anthropic_client_receives_upstream_model_and_token_cap(
    monkeypatch,
) -> None:
    captured: list[dict] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(anthropic, "ChatAnthropic", AnthropicCapturingClient)

    anthropic.get_anthropic_llm(
        Model.CLAUDE_HAIKU_5_5,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        credential=ChatCredentialSecret(
            provider="anthropic",
            api_key="sponsored-key",
            base_url="https://sponsored.example/v1",
        ),
        upstream_model="upstream-luna",
    )

    assert captured == [
        {
            "model": "upstream-luna",
            "api_key": SecretStr("sponsored-key"),
            "base_url": "https://sponsored.example/v1",
            "temperature": None,
            "max_tokens": 1024,
            "thinking": {"type": "disabled"},
            "effort": "medium",
        },
    ]


def test_google_byok_client_does_not_inherit_deployment_base_url(monkeypatch) -> None:
    captured_base_urls: list[str | None] = []

    class GoogleCapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(google, "ChatGoogleGenerativeAI", GoogleCapturingClient)

    google.get_google_llm(
        Model.GEMINI_3_5_FLASH_LITE,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="google", api_key="user-key"),
    )

    assert captured_base_urls == [None]


def test_gemini_3_reasoning_uses_thinking_level(monkeypatch) -> None:
    captured: list[dict] = []

    class GoogleCapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(google, "ChatGoogleGenerativeAI", GoogleCapturingClient)

    google.get_google_llm(
        Model.GEMINI_3_5_FLASH_LITE,
        temperature=0.2,
        top_p=0.8,
        max_tokens=128,
        reasoning=True,
        credential=ChatCredentialSecret(provider="google", api_key="user-key"),
    )

    (request,) = captured
    assert request["thinking_level"] == "medium"
    assert request["include_thoughts"] is True
    assert "thinking_budget" not in request


def test_gemini_3_8_reasoning_uses_thinking_level(monkeypatch) -> None:
    captured: list[dict] = []

    class GoogleCapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(google, "ChatGoogleGenerativeAI", GoogleCapturingClient)

    google.get_google_llm(
        Model.GEMINI_3_8_FLASH,
        temperature=0.2,
        top_p=0.8,
        max_tokens=128,
        reasoning=True,
        credential=ChatCredentialSecret(provider="google", api_key="user-key"),
    )

    request = captured[0]
    assert request["thinking_level"] == "medium"
    assert request["include_thoughts"] is True
    assert "thinking_budget" not in request


def test_anthropic_byok_client_does_not_inherit_deployment_base_url(
    monkeypatch,
) -> None:
    captured_base_urls: list[str | None] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(anthropic, "ChatAnthropic", AnthropicCapturingClient)

    anthropic.get_anthropic_llm(
        Model.CLAUDE_HAIKU_5_5,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    assert captured_base_urls == [anthropic.ANTHROPIC_DEFAULT_BASE_URL]


def test_claude_haiku_omits_sampling_parameters(monkeypatch) -> None:
    captured: list[dict] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(anthropic, "ChatAnthropic", AnthropicCapturingClient)

    anthropic.get_anthropic_llm(
        Model.CLAUDE_HAIKU_5_5,
        temperature=0.2,
        top_p=0.8,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    assert len(captured) == 1
    request = captured[0]
    assert request["temperature"] is None
    assert "top_p" not in request
    assert request["thinking"] == {"type": "disabled"}
    assert request["effort"] == "medium"


def test_claude_sonnet_5_5_uses_between_tools_without_reasoning(monkeypatch) -> None:
    captured: list[dict] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(anthropic, "ChatAnthropic", AnthropicCapturingClient)

    anthropic.get_anthropic_llm(
        Model.CLAUDE_SONNET_5_5,
        temperature=0.2,
        top_p=0.8,
        max_tokens=1024,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    request = captured[0]
    assert request["thinking"] == {"type": "between_tools"}
    assert request["effort"] == "medium"
    assert request["temperature"] is None
    assert request["max_tokens"] == 1024


@pytest.mark.parametrize(
    ("model", "reasoning", "expected_thinking", "expected_effort"),
    [
        (
            Model.CLAUDE_HAIKU_5_5,
            False,
            {"type": "disabled"},
            "medium",
        ),
        (
            Model.CLAUDE_HAIKU_5_5,
            True,
            {"type": "adaptive", "display": "summarized"},
            "medium",
        ),
        (
            Model.CLAUDE_SONNET_5_5,
            False,
            {"type": "between_tools"},
            "medium",
        ),
        (
            Model.CLAUDE_SONNET_5_5,
            True,
            {"type": "adaptive", "display": "summarized"},
            "medium",
        ),
        (
            Model.CLAUDE_OPUS_5_5,
            False,
            {"type": "adaptive", "display": "summarized"},
            "medium",
        ),
        (
            Model.CLAUDE_OPUS_5_5,
            True,
            {"type": "adaptive", "display": "summarized"},
            "medium",
        ),
    ],
)
@pytest.mark.parametrize("upstream_model", [None, "sponsored/claude"])
def test_claude_serialized_payload_preserves_reasoning_tools_and_upstream(
    model: Model,
    reasoning: bool,
    expected_thinking: dict[str, str] | None,
    expected_effort: str | None,
    upstream_model: str | None,
) -> None:
    llm = anthropic.get_anthropic_llm(
        model,
        temperature=0.2,
        top_p=0.8,
        max_tokens=1024,
        reasoning=reasoning,
        credential=ChatCredentialSecret(provider="anthropic", api_key="fake-key"),
        upstream_model=upstream_model,
    )
    assert isinstance(llm, ChatAnthropic)
    bound = llm.bind_tools(
        [
            {
                "name": "lookup",
                "description": "Look up a record",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    )

    assert isinstance(bound, RunnableBinding)
    payload = llm._get_request_payload(
        [HumanMessage(content="test")],
        **bound.kwargs,
    )

    assert payload["model"] == (upstream_model or model.value)
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload
    assert payload["tools"][0]["name"] == "lookup"
    if expected_thinking is None:
        assert "thinking" not in payload
    else:
        assert payload["thinking"] == expected_thinking
    if expected_effort is None:
        assert "output_config" not in payload
    else:
        assert payload["output_config"]["effort"] == expected_effort


def test_claude_fable_uses_always_on_adaptive_thinking(monkeypatch) -> None:
    captured: list[dict] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(anthropic, "ChatAnthropic", AnthropicCapturingClient)

    anthropic.get_anthropic_llm(
        Model.CLAUDE_FABLE_5_1,
        temperature=0.2,
        top_p=0.8,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    request = captured[0]
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["effort"] == "medium"
    assert request["max_tokens"] == 128
    assert request["temperature"] is None
    assert "budget_tokens" not in request["thinking"]


def test_ollama_byok_client_uses_user_endpoint_and_bearer_key(monkeypatch) -> None:
    captured_clients: list[dict] = []

    class OllamaCapturingClient:
        def __init__(self, **kwargs):
            captured_clients.append(kwargs)

    monkeypatch.setattr(ollama, "ChatOllama", OllamaCapturingClient)

    ollama.get_llm(
        Model.QWEN3_14B,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(
            provider="ollama",
            api_key="ollama-user-key",
            base_url="https://ollama.example",
        ),
    )

    assert captured_clients == [
        {
            "base_url": "https://ollama.example",
            "client_kwargs": {
                "headers": {"Authorization": "Bearer ollama-user-key"},
            },
            "model": Model.QWEN3_14B.value,
            "num_predict": 128,
            "reasoning": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    ]


def test_ollama_byok_client_accepts_dynamic_model_tag(monkeypatch) -> None:
    captured_clients: list[dict] = []

    class OllamaCapturingClient:
        def __init__(self, **kwargs):
            captured_clients.append(kwargs)

    monkeypatch.setattr(ollama, "ChatOllama", OllamaCapturingClient)

    ollama.get_llm(
        "bge-m3:latest",
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(
            provider="ollama",
            api_key="ollama-user-key",
            base_url="https://ollama.example",
        ),
    )

    assert [client["model"] for client in captured_clients] == ["bge-m3:latest"]


def test_ollama_byok_clients_are_not_shared_between_requests(monkeypatch) -> None:
    captured_clients: list[tuple[str, str]] = []

    class OllamaCapturingClient:
        def __init__(self, **kwargs):
            captured_clients.append(
                (
                    kwargs["base_url"],
                    kwargs["client_kwargs"]["headers"]["Authorization"],
                ),
            )

    monkeypatch.setattr(ollama, "ChatOllama", OllamaCapturingClient)

    for api_key in ("first-user-key", "second-user-key"):
        ollama.get_llm(
            Model.QWEN3_14B,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            credential=ChatCredentialSecret(provider="ollama", api_key=api_key),
        )

    assert captured_clients == [
        ("https://ollama.com", "Bearer first-user-key"),
        ("https://ollama.com", "Bearer second-user-key"),
    ]


def test_openrouter_client_uses_upstream_id_and_privacy_routing(monkeypatch) -> None:
    captured: list[dict] = []

    class OpenRouterCapturingClient:
        @classmethod
        def model_validate(cls, value):
            captured.append(value)
            return cls()

    monkeypatch.setattr(openrouter, "ChatOpenRouter", OpenRouterCapturingClient)

    openrouter.get_openrouter_llm(
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        temperature=0.2,
        top_p=0.8,
        max_tokens=1024,
        reasoning=True,
        credential=ChatCredentialSecret(
            provider="openrouter",
            api_key="openrouter-user-key",
            base_url="https://openrouter-proxy.example/v1",
        ),
    )

    assert captured == [
        {
            "model_name": "deepseek/deepseek-v4-pro-0813",
            "openrouter_api_key": SecretStr("openrouter-user-key"),
            "base_url": "https://openrouter-proxy.example/v1",
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 1024,
            "streaming": True,
            "stream_usage": True,
            "reasoning": {"effort": "high"},
            "openrouter_provider": {
                "allow_fallbacks": True,
                "data_collection": "deny",
                "require_parameters": True,
            },
        },
    ]


def test_sponsored_openrouter_client_omits_alias_reasoning_for_unknown_upstream() -> (
    None
):
    llm = openrouter.get_openrouter_llm(
        Model.OPENROUTER_QWEN3_8_MAX_0902,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        credential=ChatCredentialSecret(
            provider="openrouter",
            api_key="sponsored-key",
        ),
        upstream_model="sponsored/upstream-model",
    )

    assert llm.model_name == "sponsored/upstream-model"
    assert llm.reasoning is None


def test_sponsored_openrouter_client_uses_known_upstream_reasoning() -> None:
    llm = openrouter.get_openrouter_llm(
        Model.OPENROUTER_QWEN3_8_MAX_0902,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        credential=ChatCredentialSecret(
            provider="openrouter",
            api_key="sponsored-key",
        ),
        upstream_model="minimax/minimax-m3",
    )

    assert llm.reasoning == {"effort": "none"}


@pytest.mark.parametrize(
    ("model", "expected_reasoning"),
    [
        (Model.OPENROUTER_KIMI_K3, {"effort": "high"}),
        (Model.OPENROUTER_QWEN3_8_MAX_0902, {"effort": "high"}),
        (Model.OPENROUTER_QWEN3_8_27B, {"effort": "medium"}),
        (Model.OPENROUTER_MINIMAX_M3, {}),
        (Model.OPENROUTER_GROK_4_7, {"effort": "high"}),
    ],
)
def test_openrouter_client_uses_supported_reasoning_configuration(
    monkeypatch,
    model: Model,
    expected_reasoning: dict[str, bool | str],
) -> None:
    captured: list[dict] = []

    class OpenRouterCapturingClient:
        @classmethod
        def model_validate(cls, value):
            captured.append(value)
            return cls()

    monkeypatch.setattr(openrouter, "ChatOpenRouter", OpenRouterCapturingClient)

    openrouter.get_openrouter_llm(
        model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        reasoning=True,
        credential=ChatCredentialSecret(
            provider="openrouter",
            api_key="openrouter-user-key",
        ),
    )

    assert captured[0]["reasoning"] == expected_reasoning


@pytest.mark.parametrize(
    "model",
    [Model.OPENROUTER_MINIMAX_M3],
)
def test_openrouter_sdk_serializes_default_reasoning_configuration(
    model: Model,
) -> None:
    llm = openrouter.get_openrouter_llm(
        model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        reasoning=True,
        credential=ChatCredentialSecret(
            provider="openrouter",
            api_key="openrouter-user-key",
        ),
    )
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "test"}],
            "model": llm.model_name,
            "reasoning": llm.reasoning,
        },
    )

    assert request.model_dump(mode="json")["reasoning"] == {}


def test_openrouter_client_disables_optional_reasoning(monkeypatch) -> None:
    captured: list[dict] = []

    class OpenRouterCapturingClient:
        @classmethod
        def model_validate(cls, value):
            captured.append(value)
            return cls()

    monkeypatch.setattr(openrouter, "ChatOpenRouter", OpenRouterCapturingClient)

    openrouter.get_openrouter_llm(
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        temperature=0.2,
        top_p=0.8,
        max_tokens=1024,
        reasoning=False,
        credential=ChatCredentialSecret(
            provider="openrouter",
            api_key="openrouter-user-key",
        ),
    )

    assert captured[0]["reasoning"] == {"effort": "none"}


@pytest.mark.parametrize(
    "model",
    [
        Model.OPENROUTER_GLM_5_3,
        Model.OPENROUTER_QWEN3_8_MAX_0902,
        Model.OPENROUTER_GROK_4_7,
    ],
)
def test_openrouter_client_keeps_mandatory_reasoning_enabled(
    monkeypatch,
    model: Model,
) -> None:
    captured: list[dict] = []

    class OpenRouterCapturingClient:
        @classmethod
        def model_validate(cls, value):
            captured.append(value)
            return cls()

    monkeypatch.setattr(openrouter, "ChatOpenRouter", OpenRouterCapturingClient)

    openrouter.get_openrouter_llm(
        model,
        temperature=0.2,
        top_p=0.8,
        max_tokens=1024,
        reasoning=False,
        credential=ChatCredentialSecret(
            provider="openrouter",
            api_key="openrouter-user-key",
        ),
    )

    assert captured[0]["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize(
    ("provider", "factory", "model"),
    [
        ("openai", openai.get_openai_llm, Model.GPT_5_4_MINI),
        ("google", google.get_google_llm, Model.GEMINI_3_5_FLASH_LITE),
        ("anthropic", anthropic.get_anthropic_llm, Model.CLAUDE_HAIKU_5_5),
        ("ollama", ollama.get_llm, Model.QWEN3_14B),
        (
            "openrouter",
            openrouter.get_openrouter_llm,
            Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        ),
    ],
)
def test_byok_llm_client_requires_user_credential(
    provider,
    factory,
    model: Model,
) -> None:
    with pytest.raises(ProviderCredentialRequiredError) as error:
        factory(
            model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

    assert error.value.provider == provider


@pytest.mark.parametrize(
    ("provider", "factory", "model"),
    [
        ("openai", openai.get_openai_embedder, Model.TEXT_EMBEDDING_3_LARGE),
        ("google", google.get_google_embedder, Model.GEMINI_EMBEDDING_001),
        ("ollama", ollama.get_embedder, Model.BGE_M3),
    ],
)
def test_byok_embedding_client_requires_user_credential(
    provider,
    factory,
    model: Model,
) -> None:
    with pytest.raises(ProviderCredentialRequiredError) as error:
        factory(model)

    assert error.value.provider == provider


def test_byok_clients_ignore_deployment_gateway_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/openai/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")

    openai_llm = openai.get_openai_llm(
        Model.GPT_5_4_MINI,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openai", api_key="sk-user-key"),
    )
    openai_embedder = openai.get_openai_embedder(
        Model.TEXT_EMBEDDING_3_LARGE,
        credential=ChatCredentialSecret(provider="openai", api_key="sk-user-key"),
    )
    anthropic_llm = anthropic.get_anthropic_llm(
        Model.CLAUDE_HAIKU_5_5,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    assert str(openai_llm.root_client.base_url) == "https://api.openai.com/v1/"
    assert openai_embedder.openai_api_base == "https://api.openai.com/v1"
    assert anthropic_llm.anthropic_api_url == "https://api.anthropic.com"


@pytest.mark.parametrize(
    ("model", "expected_reasoning"),
    [(Model.GPT_6_LUNA, {"effort": "none"}), (Model.GPT_5_4_MINI, None)],
)
def test_openai_query_transform_disables_supported_default_reasoning(
    monkeypatch,
    model: Model,
    expected_reasoning: dict[str, str] | None,
) -> None:
    captured: list[dict[str, object]] = []

    class _Reply:
        content = "rewritten"

    class OpenAICapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        async def ainvoke(self, messages):
            return _Reply()

    monkeypatch.setattr(openai, "ChatOpenAI", OpenAICapturingClient)

    result = anyio.run(
        lambda: openai.transform_query_with_openai(
            "query",
            model,
            system_prompt="system",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            credential=ChatCredentialSecret(provider="openai", api_key="sk-user-key"),
        ),
    )

    assert result == "rewritten"
    assert captured[0].get("reasoning") == expected_reasoning


def test_openai_chat_keeps_default_reasoning_when_off(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class OpenAICapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(openai, "ChatOpenAI", OpenAICapturingClient)

    openai.get_openai_llm(
        Model.GPT_6_LUNA,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openai", api_key="sk-user-key"),
    )

    assert "reasoning" not in captured[0]
