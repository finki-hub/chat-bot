from uuid import uuid4

from fastapi.testclient import TestClient

from app.data.db import get_db
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase


def _client(db: FakeChatDatabase) -> TestClient:
    app = make_app(Settings(API_KEY="test-api-key", MCP_API_KEY="test-mcp-key"))
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"x-api-key": "test-api-key"}


OWNER_ID = "00000000-0000-4000-8000-000000000001"
INTRUDER_ID = "00000000-0000-4000-8000-000000000002"


def test_chat_state_active_stream_set_rejects_wrong_user() -> None:
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    stream_id = uuid4()
    response_id = uuid4()
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )

    response = client.put(
        f"/chat/state/conversations/{conversation_id}/active-stream",
        headers=_auth_headers(),
        json={
            "active_response_id": str(response_id),
            "active_status": "streaming",
            "active_stream_id": str(stream_id),
            "user_id": INTRUDER_ID,
        },
    )

    assert response.status_code == 404
    assert db.conversations[conversation_id]["active_stream_id"] is None


def test_chat_state_active_stream_stop_and_clear_reject_wrong_user() -> None:
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    stream_id = uuid4()
    response_id = uuid4()
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )
    client.put(
        f"/chat/state/conversations/{conversation_id}/active-stream",
        headers=_auth_headers(),
        json={
            "active_response_id": str(response_id),
            "active_status": "streaming",
            "active_stream_id": str(stream_id),
            "user_id": OWNER_ID,
        },
    )

    stopped = client.post(
        f"/chat/state/conversations/{conversation_id}/active-stream/{stream_id}/stop",
        headers=_auth_headers(),
        json={"user_id": INTRUDER_ID},
    )
    cleared = client.delete(
        f"/chat/state/conversations/{conversation_id}/active-stream/{stream_id}",
        headers=_auth_headers(),
        params={"user_id": INTRUDER_ID},
    )

    assert stopped.status_code == 404
    assert cleared.status_code == 404
    assert db.conversations[conversation_id]["active_status"] == "streaming"
    assert db.conversations[conversation_id]["active_stream_id"] == stream_id
