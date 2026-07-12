from uuid import UUID

import anyio
import pytest
from pydantic import ValidationError

from app.api import chat_title
from app.data.chat_credentials import upsert_chat_credential
from app.llms.models import ChatModel, Model
from app.schemas.chat import ConversationTurn
from app.schemas.chat_credentials import ChatCredentialUpsert
from app.schemas.chat_title import ChatTitleSchema
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase


def test_provider_model_validation_names_the_invalid_field() -> None:
    with pytest.raises(
        ValidationError,
        match="provider_model must be a chat-capable model",
    ):
        ChatTitleSchema.model_validate(
            {
                "messages": [
                    ConversationTurn(role="user", content="Кога е јунската сесија?"),
                ],
                "provider_model": Model.BGE_M3_LOCAL,
            },
        )


def test_chat_title_uses_cheapest_model_for_active_provider(monkeypatch):
    selected_models: list[Model] = []

    async def fake_transform_query(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        credentials=None,
    ) -> str:
        selected_models.append(model)
        return "Испитен рок"

    monkeypatch.setattr(chat_title, "transform_query", fake_transform_query)
    db = FakeChatDatabase()
    user_id = UUID("00000000-0000-4000-8000-000000000001")
    settings = Settings(
        API_KEY="test-api-key",
        CREDENTIAL_ENCRYPTION_KEY="runtime-credential-key",
        MCP_API_KEY="test-mcp-key",
    )

    async def run_title():
        for provider in ("anthropic", "openai"):
            await upsert_chat_credential(
                db,
                user_id=user_id,
                credential=ChatCredentialUpsert(
                    api_key=f"{provider}-user-key",
                    provider=provider,
                ),
                settings=settings,
            )
        return await chat_title.generate_chat_title(
            ChatTitleSchema.model_validate(
                {
                    "user_id": user_id,
                    "messages": [
                        ConversationTurn(
                            role="user",
                            content="Кога е јунската сесија?",
                        ),
                    ],
                    "provider_model": Model.CLAUDE_SONNET_5,
                },
            ),
            db,
            settings,
        )

    response = anyio.run(run_title)

    assert response.title == "Испитен рок"
    assert selected_models == [Model.CLAUDE_HAIKU_4_5]


def test_chat_title_uses_dynamic_ollama_provider_model_directly(monkeypatch):
    selected_models: list[ChatModel] = []

    async def fake_transform_query(
        query: str,
        model: ChatModel,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        credentials=None,
    ) -> str:
        selected_models.append(model)
        return "Испитен рок"

    monkeypatch.setattr(chat_title, "transform_query", fake_transform_query)
    monkeypatch.setattr(chat_title, "has_provider_credential", lambda *args: True)
    dynamic_model = "llama3.2:latest"
    payload = ChatTitleSchema.model_validate(
        {
            "messages": [
                ConversationTurn(role="user", content="Кога е јунската сесија?"),
            ],
            "provider_model": dynamic_model,
        },
    )

    response = anyio.run(chat_title.generate_chat_title, payload)

    assert response.title == "Испитен рок"
    assert selected_models == [dynamic_model]
