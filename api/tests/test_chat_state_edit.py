from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import chat_state
from app.data.db import get_db
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase

OWNER_ID = "00000000-0000-4000-8000-000000000001"
HEADERS = {"x-api-key": "test-api-key"}


def _client() -> TestClient:
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            BYOK_ALLOWED_BASE_URLS="https://api.openai.com/v1",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )
    app.dependency_overrides[get_db] = FakeChatDatabase
    return TestClient(app)


@pytest.mark.parametrize(("pruned", "status"), [(True, 204), (False, 404)])
def test_prune_after_user_message_reports_whether_it_applied(
    monkeypatch: pytest.MonkeyPatch,
    pruned: bool,
    status: int,
) -> None:
    calls: list[dict[str, UUID]] = []

    async def prune(_db: object, **kwargs: UUID) -> bool:
        calls.append(kwargs)
        return pruned

    monkeypatch.setattr(chat_state, "prune_messages_after_user_message", prune)
    conversation_id, message_id = uuid4(), uuid4()

    response = _client().delete(
        f"/chat/state/conversations/{conversation_id}/messages/user/{message_id}/following",
        headers=HEADERS,
        params={"user_id": OWNER_ID},
    )

    assert response.status_code == status
    assert calls == [
        {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "user_id": UUID(OWNER_ID),
        },
    ]
