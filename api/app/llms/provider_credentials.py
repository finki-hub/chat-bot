from dataclasses import dataclass
from typing import Literal, assert_never

from app.llms.models import (
    ANTHROPIC_QUERY_TRANSFORM_MODELS,
    GOOGLE_QUERY_TRANSFORM_MODELS,
    OLLAMA_QUERY_TRANSFORM_MODELS,
    OPENAI_QUERY_TRANSFORM_MODELS,
    OPENCODE_GO_QUERY_TRANSFORM_MODELS,
    OPENROUTER_QUERY_TRANSFORM_MODELS,
    ChatModel,
    Model,
)
from app.schemas.chat_credentials import ChatCredentialSecret

ProviderName = Literal[
    "openai", "google", "anthropic", "ollama", "openrouter", "opencode-go"
]


class ProviderCredentialRequiredError(RuntimeError):
    def __init__(self, provider: ProviderName) -> None:
        self.provider = provider
        super().__init__(f"A user credential is required for {provider}")


def require_provider_credential(
    provider: ProviderName,
    credential: ChatCredentialSecret | None,
) -> ChatCredentialSecret:
    if credential is None:
        raise ProviderCredentialRequiredError(provider)
    return credential


def provider_for_model(model: ChatModel) -> ProviderName | None:
    if not isinstance(model, Model):
        if model.startswith("opencode-go:"):
            raise ValueError("Unsupported OpenCode Go model")
        return "ollama"
    if model in OPENAI_QUERY_TRANSFORM_MODELS or model == Model.TEXT_EMBEDDING_3_LARGE:
        return "openai"
    if model in GOOGLE_QUERY_TRANSFORM_MODELS or model == Model.GEMINI_EMBEDDING_001:
        return "google"
    if model in ANTHROPIC_QUERY_TRANSFORM_MODELS:
        return "anthropic"
    if model in OPENROUTER_QUERY_TRANSFORM_MODELS:
        return "openrouter"
    if model in OPENCODE_GO_QUERY_TRANSFORM_MODELS:
        return "opencode-go"
    if model in OLLAMA_QUERY_TRANSFORM_MODELS or model == Model.BGE_M3:
        return "ollama"
    return None


@dataclass(frozen=True, slots=True)
class LlmProviderCredentials:
    openai: ChatCredentialSecret | None = None
    google: ChatCredentialSecret | None = None
    anthropic: ChatCredentialSecret | None = None
    ollama: ChatCredentialSecret | None = None
    openrouter: ChatCredentialSecret | None = None
    opencode_go: ChatCredentialSecret | None = None
    conversation_id: str | None = None
    rejected_providers: frozenset[ProviderName] = frozenset()

    def available_providers(self) -> frozenset[ProviderName]:
        providers: tuple[ProviderName, ...] = (
            "openai",
            "google",
            "anthropic",
            "ollama",
            "openrouter",
            "opencode-go",
        )
        return frozenset(
            provider
            for provider in providers
            if self.for_provider(provider) is not None
        )

    def for_provider(self, provider: ProviderName) -> ChatCredentialSecret | None:
        match provider:
            case "openai":
                return self.openai
            case "google":
                return self.google
            case "anthropic":
                return self.anthropic
            case "ollama":
                return self.ollama
            case "openrouter":
                return self.openrouter
            case "opencode-go":
                return self.opencode_go
            case unreachable:
                assert_never(unreachable)
        raise AssertionError(f"Unhandled provider: {provider}")


def has_provider_credential(
    credentials: LlmProviderCredentials | None,
    model: ChatModel,
) -> bool:
    provider = provider_for_model(model)
    return provider is None or (
        credentials is not None and credentials.for_provider(provider) is not None
    )
