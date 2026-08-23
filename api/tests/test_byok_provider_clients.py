import pytest
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

    assert captured_base_urls == [None]


def test_sponsored_openai_client_receives_upstream_model_and_token_cap(
    monkeypatch,
) -> None:
    captured: list[dict] = []

    class OpenAICapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(openai, "ChatOpenAI", OpenAICapturingClient)

    openai.get_openai_llm(
        Model.GPT_5_6_LUNA,
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
            "temperature": 0.0,
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
        Model.GEMINI_3_5_FLASH,
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
        Model.CLAUDE_HAIKU_4_5,
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
            "temperature": 0.0,
            "max_tokens": 1024,
            "thinking": None,
        },
    ]


def test_google_byok_client_does_not_inherit_deployment_base_url(monkeypatch) -> None:
    captured_base_urls: list[str | None] = []

    class GoogleCapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(google, "ChatGoogleGenerativeAI", GoogleCapturingClient)

    google.get_google_llm(
        Model.GEMINI_3_5_FLASH,
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
        Model.GEMINI_3_5_FLASH,
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


def test_anthropic_byok_client_does_not_inherit_deployment_base_url(
    monkeypatch,
) -> None:
    captured_base_urls: list[str | None] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(anthropic, "ChatAnthropic", AnthropicCapturingClient)

    anthropic.get_anthropic_llm(
        Model.CLAUDE_HAIKU_4_5,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    assert captured_base_urls == [None]


def test_claude_haiku_keeps_temperature_and_omits_top_p(monkeypatch) -> None:
    captured: list[dict] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(anthropic, "ChatAnthropic", AnthropicCapturingClient)

    anthropic.get_anthropic_llm(
        Model.CLAUDE_HAIKU_4_5,
        temperature=0.2,
        top_p=0.8,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    assert len(captured) == 1
    request = captured[0]
    assert request["temperature"] == pytest.approx(0.2)
    assert "top_p" not in request
    assert request["thinking"] is None


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


def test_sponsored_openrouter_client_receives_upstream_model(monkeypatch) -> None:
    captured: list[dict] = []

    class OpenRouterCapturingClient:
        @classmethod
        def model_validate(cls, value):
            captured.append(value)
            return cls()

    monkeypatch.setattr(openrouter, "ChatOpenRouter", OpenRouterCapturingClient)

    openrouter.get_openrouter_llm(
        Model.OPENROUTER_DEEPSEEK_V4_PRO_0813,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        credential=ChatCredentialSecret(
            provider="openrouter",
            api_key="sponsored-key",
        ),
        upstream_model="sponsored/upstream-model",
    )

    assert captured[0]["model_name"] == "sponsored/upstream-model"


@pytest.mark.parametrize(
    ("model", "expected_reasoning"),
    [
        (Model.OPENROUTER_KIMI_K3, {"effort": "high"}),
        (Model.OPENROUTER_QWEN3_8_MAX, {"effort": "high"}),
        (Model.OPENROUTER_QWEN3_8_27B, {"effort": "medium"}),
        (Model.OPENROUTER_MINIMAX_M3, {}),
        (Model.OPENROUTER_GROK_4_6, {"effort": "high"}),
        (Model.OPENROUTER_HY3, {"effort": "high"}),
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


def test_openrouter_client_keeps_mandatory_reasoning_enabled(monkeypatch) -> None:
    captured: list[dict] = []

    class OpenRouterCapturingClient:
        @classmethod
        def model_validate(cls, value):
            captured.append(value)
            return cls()

    monkeypatch.setattr(openrouter, "ChatOpenRouter", OpenRouterCapturingClient)

    openrouter.get_openrouter_llm(
        Model.OPENROUTER_GLM_5_3,
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
        ("google", google.get_google_llm, Model.GEMINI_3_5_FLASH),
        ("anthropic", anthropic.get_anthropic_llm, Model.CLAUDE_HAIKU_4_5),
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
