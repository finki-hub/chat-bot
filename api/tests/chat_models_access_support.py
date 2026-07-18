from datetime import UTC, date, datetime
from uuid import UUID

from fastapi.testclient import TestClient

from app.data.sponsored_usage import SponsoredUsageSnapshot
from app.llms.model_catalog_types import (
    ExecutionPolicy,
    ModelCatalogResponse,
    ModelDescriptor,
    OllamaCatalogModel,
)
from app.llms.provider_credentials import LlmProviderCredentials
from app.main import make_app
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings

LUNA_ID = "gpt-5.6-luna"
OTHER_MODEL_ID = "gpt-5.6-sol"
USER_WITH_KEY = UUID("00000000-0000-4000-8000-000000000001")
USER_WITHOUT_KEY = UUID("00000000-0000-4000-8000-000000000002")
RESET = datetime(2026, 7, 19, tzinfo=UTC)
USER_CREDENTIAL = "test-user-openai-key"
BASE_URL = "https://user.example/v1"


class ModelsDatabase:
    pool = True

    def __init__(self, dsn: str, min_size: int, max_size: int) -> None:
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size

    async def init(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None


class StubCatalogService:
    def __init__(self, catalog: ModelCatalogResponse) -> None:
        self.catalog = catalog
        self.ollama_calls: list[tuple[OllamaCatalogModel, ...]] = []

    async def get_catalog(
        self,
        ollama_models: tuple[OllamaCatalogModel, ...] = (),
    ) -> ModelCatalogResponse:
        self.ollama_calls.append(ollama_models)
        ollama_descriptors = tuple(
            ModelDescriptor(
                id=model.id,
                provider="ollama",
                name=model.name,
                execution=ExecutionPolicy(
                    reasoning=False,
                    sampling=True,
                    tool_call=False,
                    structured_output=False,
                ),
                loaded=model.loaded,
            )
            for model in ollama_models
        )
        return self.catalog.model_copy(
            update={"models": (*self.catalog.models, *ollama_descriptors)},
        )


def base_catalog() -> ModelCatalogResponse:
    execution = ExecutionPolicy(
        reasoning=True,
        sampling=True,
        tool_call=True,
        structured_output=True,
    )
    return ModelCatalogResponse(
        source="live",
        models=(
            ModelDescriptor(
                id=LUNA_ID,
                provider="openai",
                name="Luna",
                execution=execution,
            ),
            ModelDescriptor(
                id=OTHER_MODEL_ID,
                provider="openai",
                name="Sol",
                execution=execution,
            ),
        ),
    )


def credentials(*, openai: bool, ollama: bool = False) -> LlmProviderCredentials:
    return LlmProviderCredentials(
        openai=(
            ChatCredentialSecret(
                provider="openai",
                api_key=USER_CREDENTIAL,
                base_url=BASE_URL,
            )
            if openai
            else None
        ),
        ollama=(
            ChatCredentialSecret(
                provider="ollama",
                api_key="ollama-secret",
                base_url="https://ollama.example",
            )
            if ollama
            else None
        ),
    )


def usage_snapshot(
    user_id: UUID,
    *,
    personal_remaining: int = 5,
    global_remaining: int = 10,
) -> SponsoredUsageSnapshot:
    return SponsoredUsageSnapshot(
        user_id=user_id,
        usage_date=date(2026, 7, 18),
        user_request_count=5 - personal_remaining,
        global_request_count=10 - global_remaining,
        user_limit=5,
        global_limit=10,
        remaining_user_requests=personal_remaining,
        remaining_global_requests=global_remaining,
        reset_at=RESET,
    )


def settings(*, enabled: bool = True) -> Settings:
    return Settings(
        API_KEY="test-api-key",
        CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
        MCP_API_KEY="test-mcp-key",
        SPONSORED_LUNA_ENABLED=enabled,
        SPONSORED_OPENAI_API_KEY="sponsored-secret" if enabled else None,
        SPONSORED_DAILY_GLOBAL_LIMIT=10 if enabled else None,
    )


def client(monkeypatch, current_settings: Settings) -> TestClient:
    monkeypatch.setattr("app.main.Database", ModelsDatabase)
    return TestClient(make_app(current_settings))
