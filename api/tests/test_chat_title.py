import logging
from uuid import UUID

import anyio

from app.api import chat_title
from app.data.chat_credentials import upsert_chat_credential
from app.llms import query_transform
from app.llms.models import Model
from app.llms.provider_credentials import LlmProviderCredentials
from app.schemas.chat_credentials import ChatCredentialUpsert
from app.schemas.chat_title import ChatTitleSchema
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase


def test_chat_title_uses_transform_prompt_and_normalizes_quotes(monkeypatch):
    seen: dict[str, str | int | float | Model] = {}

    async def fake_transform_query(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        credentials: object | None = None,
    ) -> str:
        seen.update(
            {
                "max_tokens": max_tokens,
                "model": model,
                "query": query,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
        return '"Испитна сесија"\n'

    monkeypatch.setattr(chat_title, "transform_query", fake_transform_query)
    payload = ChatTitleSchema(
        messages=[{"role": "user", "content": "Кога е јунската сесија?"}],
        query_transform_model=Model.CLAUDE_HAIKU_4_5,
    )

    response = anyio.run(chat_title.generate_chat_title, payload)

    assert response.title == "Испитна сесија"
    assert seen["model"] == Model.CLAUDE_HAIKU_4_5
    assert seen["max_tokens"] == 32
    assert "Кога е јунската сесија?" in str(seen["query"])
    assert "conversation title" in str(seen["system_prompt"])


def test_chat_title_falls_back_to_first_user_message_when_model_returns_empty(
    monkeypatch,
):
    async def fake_transform_query(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        credentials: object | None = None,
    ) -> str:
        return "   "

    monkeypatch.setattr(chat_title, "transform_query", fake_transform_query)
    payload = ChatTitleSchema(
        messages=[
            {
                "role": "user",
                "content": "  Како да пријавам испит?\nИ кои се роковите?  ",
            },
        ],
    )

    response = anyio.run(chat_title.generate_chat_title, payload)

    assert response.title == "Како да пријавам испит?"


def test_chat_title_uses_runtime_settings_for_user_credentials(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_transform_query(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        credentials: object | None = None,
    ) -> str:
        seen["credentials"] = credentials
        return "Испитна сесија"

    monkeypatch.setattr(chat_title, "transform_query", fake_transform_query)
    db = FakeChatDatabase()
    user_id = UUID("00000000-0000-4000-8000-000000000001")
    credential_key = "anthropic-user-token".replace("token", "key")
    settings = Settings(
        API_KEY="test-api-key",
        CREDENTIAL_ENCRYPTION_KEY="runtime-credential-key",
        MCP_API_KEY="test-mcp-key",
    )

    async def run_title() -> None:
        await upsert_chat_credential(
            db,
            user_id=user_id,
            credential=ChatCredentialUpsert(
                api_key=credential_key,
                provider="anthropic",
            ),
            settings=settings,
        )
        await chat_title.generate_chat_title(
            ChatTitleSchema(
                messages=[{"role": "user", "content": "Кога е јунската сесија?"}],
                query_transform_model=Model.CLAUDE_HAIKU_4_5,
                user_id=user_id,
            ),
            db,
            settings,
        )

    anyio.run(run_title)

    credentials = seen["credentials"]
    assert isinstance(credentials, LlmProviderCredentials)
    assert credentials.anthropic is not None
    assert credentials.anthropic.api_key == credential_key


def test_query_transform_logs_metadata_without_raw_query(monkeypatch, caplog):
    async def fake_transform_query_with_openai(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        credential: object | None = None,
    ) -> str:
        return "Испитен рок"

    monkeypatch.setattr(
        query_transform,
        "transform_query_with_openai",
        fake_transform_query_with_openai,
    )
    caplog.set_level(logging.INFO, logger=query_transform.__name__)

    async def run_transform_query() -> None:
        await query_transform.transform_query(
            "Conversation transcript: private student question",
            Model.GPT_5_4_MINI,
            system_prompt="system",
            temperature=0.2,
            top_p=1.0,
            max_tokens=32,
        )

    anyio.run(run_transform_query)

    assert "private student question" not in caplog.text
    assert "query_length" in caplog.text
