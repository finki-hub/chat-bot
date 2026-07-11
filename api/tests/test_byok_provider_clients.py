import pytest

from app.llms import anthropic, google, ollama, openai
from app.llms.models import Model
from app.llms.provider_credentials import ProviderCredentialRequiredError
from app.schemas.chat_credentials import ChatCredentialSecret


def test_openai_byok_client_does_not_inherit_deployment_base_url(monkeypatch) -> None:
    captured_base_urls: list[str | None] = []

    class OpenAICapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(openai, "ChatOpenAI", OpenAICapturingClient)

    # When: a user supplies only their own OpenAI key.
    openai.get_openai_llm(
        Model.GPT_4O_MINI,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="openai", api_key="sk-user-key"),
    )

    # Then: the BYOK request uses the provider default, not the deployment base URL.
    assert captured_base_urls == [None]


def test_google_byok_client_does_not_inherit_deployment_base_url(monkeypatch) -> None:
    captured_base_urls: list[str | None] = []

    class GoogleCapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(google, "ChatGoogleGenerativeAI", GoogleCapturingClient)

    # When: a user supplies only their own Google key.
    google.get_google_llm(
        Model.GEMINI_2_5_FLASH,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="google", api_key="user-key"),
    )

    # Then: the BYOK request uses the provider default, not the deployment base URL.
    assert captured_base_urls == [None]


def test_anthropic_byok_client_does_not_inherit_deployment_base_url(
    monkeypatch,
) -> None:
    captured_base_urls: list[str | None] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(anthropic, "ChatAnthropic", AnthropicCapturingClient)

    # When: a user supplies only their own Anthropic key.
    anthropic.get_anthropic_llm(
        Model.CLAUDE_HAIKU_4_5,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        credential=ChatCredentialSecret(provider="anthropic", api_key="user-key"),
    )

    # Then: the BYOK request uses the provider default, not the deployment base URL.
    assert captured_base_urls == [None]


def test_ollama_byok_client_uses_user_endpoint_and_bearer_key(monkeypatch) -> None:
    captured_clients: list[dict] = []

    class OllamaCapturingClient:
        def __init__(self, **kwargs):
            captured_clients.append(kwargs)

    monkeypatch.setattr(ollama, "ChatOllama", OllamaCapturingClient)

    ollama.get_llm(
        Model.MISTRAL,
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
            "model": Model.MISTRAL.value,
            "num_predict": 128,
            "reasoning": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    ]


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
            Model.MISTRAL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            credential=ChatCredentialSecret(provider="ollama", api_key=api_key),
        )

    assert captured_clients == [
        ("https://ollama.com", "Bearer first-user-key"),
        ("https://ollama.com", "Bearer second-user-key"),
    ]


@pytest.mark.parametrize(
    ("provider", "factory", "model"),
    [
        ("openai", openai.get_openai_llm, Model.GPT_4O_MINI),
        ("google", google.get_google_llm, Model.GEMINI_2_5_FLASH),
        ("anthropic", anthropic.get_anthropic_llm, Model.CLAUDE_HAIKU_4_5),
        ("ollama", ollama.get_llm, Model.MISTRAL),
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
