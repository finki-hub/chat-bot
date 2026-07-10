from dataclasses import dataclass
from typing import Literal, assert_never

from app.llms.models import (
    ANTHROPIC_QUERY_TRANSFORM_MODELS,
    GOOGLE_QUERY_TRANSFORM_MODELS,
    OLLAMA_QUERY_TRANSFORM_MODELS,
    OPENAI_QUERY_TRANSFORM_MODELS,
    Model,
)
from app.schemas.chat_credentials import ChatCredentialSecret

ProviderName = Literal["openai", "google", "anthropic", "ollama"]


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


def provider_for_model(model: Model) -> ProviderName | None:
    if model in OPENAI_QUERY_TRANSFORM_MODELS or model == Model.TEXT_EMBEDDING_3_LARGE:
        return "openai"
    if model in GOOGLE_QUERY_TRANSFORM_MODELS or model == Model.GEMINI_EMBEDDING_001:
        return "google"
    if model in ANTHROPIC_QUERY_TRANSFORM_MODELS:
        return "anthropic"
    if model in OLLAMA_QUERY_TRANSFORM_MODELS or model == Model.BGE_M3:
        return "ollama"
    return None


@dataclass(frozen=True, slots=True)
class LlmProviderCredentials:
    openai: ChatCredentialSecret | None = None
    google: ChatCredentialSecret | None = None
    anthropic: ChatCredentialSecret | None = None
    ollama: ChatCredentialSecret | None = None

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
            case unreachable:
                assert_never(unreachable)
        raise AssertionError(f"Unhandled provider: {provider}")


def has_provider_credential(
    credentials: LlmProviderCredentials | None,
    model: Model,
) -> bool:
    provider = provider_for_model(model)
    return provider is None or (
        credentials is not None and credentials.for_provider(provider) is not None
    )
