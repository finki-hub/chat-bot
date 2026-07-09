from dataclasses import dataclass
from typing import Literal, assert_never

from app.schemas.chat_credentials import ChatCredentialSecret

ProviderName = Literal["openai", "google", "anthropic"]


@dataclass(frozen=True, slots=True)
class LlmProviderCredentials:
    openai: ChatCredentialSecret | None = None
    google: ChatCredentialSecret | None = None
    anthropic: ChatCredentialSecret | None = None

    def for_provider(self, provider: ProviderName) -> ChatCredentialSecret | None:
        match provider:
            case "openai":
                return self.openai
            case "google":
                return self.google
            case "anthropic":
                return self.anthropic
            case unreachable:
                assert_never(unreachable)
        raise AssertionError(f"Unhandled provider: {provider}")
