from datetime import timedelta
from uuid import UUID, uuid4

from chat_persistence_fake import FakeChatDatabase
from fastapi.testclient import TestClient

from app.data.db import get_db
from app.main import make_app
from app.utils.settings import Settings


def _client(db: FakeChatDatabase) -> TestClient:
    app = make_app(Settings(API_KEY="test-api-key", MCP_API_KEY="test-mcp-key"))
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"x-api-key": "test-api-key"}


OWNER_ID = "00000000-0000-4000-8000-000000000001"
INTRUDER_ID = "00000000-0000-4000-8000-000000000002"


def test_chat_state_upserts_google_user() -> None:
    # Given: an authenticated BFF has a Google subject from Auth.js.
    db = FakeChatDatabase()
    client = _client(db)

    # When: the same Google subject is upserted twice with a changed email.
    first = client.post(
        "/chat/state/users/google",
        headers=_auth_headers(),
        json={
            "provider_subject": "google-sub-1",
            "email": "old@example.com",
            "name": "Old Name",
            "avatar_url": "https://example.com/old.png",
        },
    )
    second = client.post(
        "/chat/state/users/google",
        headers=_auth_headers(),
        json={
            "provider_subject": "google-sub-1",
            "email": "new@example.com",
            "name": "New Name",
            "avatar_url": "https://example.com/new.png",
        },
    )

    # Then: the API-owned user id remains stable across profile changes.
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert second.json()["email"] == "new@example.com"


def test_chat_state_requires_api_key() -> None:
    # Given: a BFF-owned chat state endpoint.
    db = FakeChatDatabase()
    client = _client(db)

    # When: the BFF API key is omitted.
    response = client.post(
        "/chat/state/conversations",
        json={
            "id": str(uuid4()),
            "user_id": OWNER_ID,
            "model": "claude-sonnet-5",
        },
    )

    # Then: state mutation is rejected like feedback/write endpoints.
    assert response.status_code == 401


def test_chat_state_lifecycle_persists_messages_and_active_stream() -> None:
    # Given: an authenticated BFF client owns an anonymous user's conversation.
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    user_message_id = uuid4()
    assistant_message_id = uuid4()
    stream_id = uuid4()
    response_id = uuid4()

    created = client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={
            "id": str(conversation_id),
            "user_id": OWNER_ID,
            "model": "claude-sonnet-5",
            "title": "Enrollment",
        },
    )
    user_message = client.post(
        f"/chat/state/conversations/{conversation_id}/messages/user",
        headers=_auth_headers(),
        json={"id": str(user_message_id), "user_id": OWNER_ID, "content": "How?"},
    )

    # When: active stream state and assistant content are set, replaced, and loaded.
    active = client.put(
        f"/chat/state/conversations/{conversation_id}/active-stream",
        headers=_auth_headers(),
        json={
            "user_id": OWNER_ID,
            "active_stream_id": str(stream_id),
            "active_response_id": str(response_id),
            "active_status": "streaming",
        },
    )
    assistant = client.put(
        f"/chat/state/conversations/{conversation_id}/messages/assistant/{response_id}",
        headers=_auth_headers(),
        json={
            "id": str(assistant_message_id),
            "user_id": OWNER_ID,
            "content": "First draft",
            "metadata": {"responseId": str(response_id)},
        },
    )
    replaced = client.put(
        f"/chat/state/conversations/{conversation_id}/messages/assistant/{response_id}",
        headers=_auth_headers(),
        json={
            "id": str(uuid4()),
            "user_id": OWNER_ID,
            "content": "Final answer",
            "metadata": {"responseId": str(response_id), "done": True},
        },
    )
    loaded = client.get(
        f"/chat/state/conversations/{conversation_id}",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: the endpoint keeps public streaming untouched and owns persisted chat state.
    assert created.status_code == 200
    assert user_message.status_code == 200
    assert active.status_code == 200
    assert assistant.status_code == 200
    assert replaced.status_code == 200
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["conversation"]["active_stream_id"] == str(stream_id)
    assert body["conversation"]["active_response_id"] == str(response_id)
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert body["messages"][1]["id"] == str(assistant_message_id)
    assert body["messages"][1]["content"] == "Final answer"
    assert body["messages"][1]["metadata"] == {
        "responseId": str(response_id),
        "done": True,
    }


def test_chat_state_wrong_user_cannot_load_or_mutate_state() -> None:
    # Given: a conversation owned by one anonymous user.
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    response_id = uuid4()
    create_response = client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )

    # When: a different user attempts to read and mutate by conversation id.
    loaded = client.get(
        f"/chat/state/conversations/{conversation_id}",
        headers=_auth_headers(),
        params={"user_id": INTRUDER_ID},
    )
    message = client.put(
        f"/chat/state/conversations/{conversation_id}/messages/assistant/{response_id}",
        headers=_auth_headers(),
        json={"id": str(uuid4()), "user_id": INTRUDER_ID, "content": "steal"},
    )

    # Then: ownership failures are indistinguishable from missing state.
    assert create_response.status_code == 200
    assert loaded.status_code == 404
    assert loaded.json() == {"detail": "Conversation not found"}
    assert message.status_code == 404
    assert all(row["content"] != "steal" for row in db.messages.values())


def test_chat_state_clear_and_stop_are_current_stream_guarded() -> None:
    # Given: an active stream was superseded by a newer stream for the same owner.
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    stale_stream_id = uuid4()
    current_stream_id = uuid4()
    current_response_id = uuid4()
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )
    for stream_id in (stale_stream_id, current_stream_id):
        client.put(
            f"/chat/state/conversations/{conversation_id}/active-stream",
            headers=_auth_headers(),
            json={
                "user_id": OWNER_ID,
                "active_stream_id": str(stream_id),
                "active_response_id": str(current_response_id),
                "active_status": "streaming",
            },
        )

    # When: stale clear/stop requests arrive before the current stream is stopped and cleared.
    stale_clear = client.delete(
        f"/chat/state/conversations/{conversation_id}/active-stream/{stale_stream_id}",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )
    stale_stop = client.post(
        f"/chat/state/conversations/{conversation_id}/active-stream/{stale_stream_id}/stop",
        headers=_auth_headers(),
        json={"user_id": OWNER_ID},
    )
    stopped = client.post(
        f"/chat/state/conversations/{conversation_id}/active-stream/{current_stream_id}/stop",
        headers=_auth_headers(),
        json={"user_id": OWNER_ID},
    )
    cleared = client.delete(
        f"/chat/state/conversations/{conversation_id}/active-stream/{current_stream_id}",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: late stale operations do not erase or stop the newer active stream.
    assert stale_clear.status_code == 404
    assert stale_stop.status_code == 404
    assert stopped.status_code == 200
    assert stopped.json()["active_status"] == "stopped"
    assert stopped.json()["active_stream_id"] == str(current_stream_id)
    assert cleared.status_code == 200
    assert cleared.json()["active_stream_id"] is None


def test_chat_state_clear_stale_active_streams_uses_cutoff() -> None:
    # Given: one stale active stream and one fresh active stream.
    db = FakeChatDatabase()
    client = _client(db)
    stale_conversation_id = uuid4()
    fresh_conversation_id = uuid4()
    for conversation_id in (stale_conversation_id, fresh_conversation_id):
        client.post(
            "/chat/state/conversations",
            headers=_auth_headers(),
            json={"id": str(conversation_id), "user_id": OWNER_ID},
        )
        client.put(
            f"/chat/state/conversations/{conversation_id}/active-stream",
            headers=_auth_headers(),
            json={
                "user_id": OWNER_ID,
                "active_stream_id": str(uuid4()),
                "active_response_id": str(uuid4()),
                "active_status": "streaming",
            },
        )
    db.conversations[stale_conversation_id]["updated_at"] = db.now - timedelta(hours=2)

    # When: the BFF asks the API to clear stale active stream markers.
    response = client.post(
        "/chat/state/active-streams/clear-stale",
        headers=_auth_headers(),
        json={"stale_before": (db.now - timedelta(hours=1)).isoformat()},
    )

    # Then: only stale active state is cleared.
    assert response.status_code == 200
    assert response.json() == {"cleared_count": 1}
    assert db.conversations[stale_conversation_id]["active_stream_id"] is None
    assert isinstance(db.conversations[fresh_conversation_id]["active_stream_id"], UUID)
