from datetime import timedelta
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.data.db import get_db
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase

OWNER_ID = "00000000-0000-4000-8000-000000000001"


def _client(db: FakeChatDatabase) -> TestClient:
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            BYOK_ALLOWED_BASE_URLS="https://api.openai.com/v1",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _message(
    db: FakeChatDatabase,
    *,
    conversation_id: UUID,
    index: int,
    message_id: UUID,
    role: str,
) -> None:
    db.messages[message_id] = {
        "content": f"{role} {index}",
        "conversation_id": conversation_id,
        "created_at": db.now + timedelta(seconds=index),
        "id": message_id,
        "metadata": {},
        "response_id": None,
        "role": role,
        "updated_at": db.now,
    }


def test_replacement_prunes_every_message_outside_retained_window() -> None:
    # Given: an earlier stale assistant and a later turn surround the target answer.
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    setup_user_id = uuid4()
    stale_assistant_id = uuid4()
    current_user_id = uuid4()
    target_message_id = uuid4()
    later_user_id = uuid4()
    active_stream_id = uuid4()
    headers = {"x-api-key": "test-api-key"}
    client.post(
        "/chat/state/conversations",
        headers=headers,
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )
    for index, message_id, role in (
        (0, setup_user_id, "user"),
        (1, stale_assistant_id, "assistant"),
        (2, current_user_id, "user"),
        (3, target_message_id, "assistant"),
        (4, later_user_id, "user"),
    ):
        _message(
            db,
            conversation_id=conversation_id,
            index=index,
            message_id=message_id,
            role=role,
        )
    client.put(
        f"/chat/state/conversations/{conversation_id}/active-stream",
        headers=headers,
        json={
            "active_replacement_message_id": str(target_message_id),
            "active_response_id": str(active_stream_id),
            "active_status": "streaming",
            "active_stream_id": str(active_stream_id),
            "user_id": OWNER_ID,
        },
    )

    # When: replacement completion declares the authoritative retained window.
    retained_ids = [setup_user_id, current_user_id, target_message_id]
    response = client.put(
        f"/chat/state/conversations/{conversation_id}/messages/assistant/{target_message_id}/replacement/{active_stream_id}",
        headers=headers,
        json={
            "active_stream_id": str(active_stream_id),
            "content": "Replacement answer",
            "retained_message_ids": [str(message_id) for message_id in retained_ids],
            "user_id": OWNER_ID,
        },
    )

    # Then: only the retained server-owned message window remains persisted.
    assert response.status_code == 200
    assert set(db.messages) == set(retained_ids)
    assert db.messages[target_message_id]["content"] == "Replacement answer"
