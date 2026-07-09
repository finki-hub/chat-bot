from app.llms import anthropic, google, openai
from app.llms.models import Model
from app.schemas.chat_credentials import ChatCredentialSecret


def test_openai_byok_client_does_not_inherit_deployment_base_url(monkeypatch) -> None:
    # Given: deployment OpenAI traffic is routed through an operator-configured base URL.
    captured_base_urls: list[str | None] = []

    class OpenAICapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(
        openai.settings,
        "OPENAI_BASE_URL",
        "https://operator.example/openai",
    )
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
    # Given: deployment Google traffic is routed through an operator-configured base URL.
    captured_base_urls: list[str | None] = []

    class GoogleCapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(
        google.settings,
        "GOOGLE_BASE_URL",
        "https://operator.example/google",
    )
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
    # Given: deployment Anthropic traffic is routed through an operator-configured base URL.
    captured_base_urls: list[str | None] = []

    class AnthropicCapturingClient:
        def __init__(self, **kwargs):
            captured_base_urls.append(kwargs["base_url"])

    monkeypatch.setattr(
        anthropic.settings,
        "ANTHROPIC_BASE_URL",
        "https://operator.example/anthropic",
    )
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
