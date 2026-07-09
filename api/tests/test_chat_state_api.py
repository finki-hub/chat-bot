from datetime import timedelta
from uuid import UUID, uuid4

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


def test_chat_state_upserts_provider_user() -> None:
    # Given: an authenticated BFF has a provider subject from Auth.js.
    db = FakeChatDatabase()
    client = _client(db)

    # When: the same provider subject is upserted twice with a changed email.
    first = client.post(
        "/chat/state/users",
        headers=_auth_headers(),
        json={
            "provider": "microsoft-entra-id",
            "provider_subject": "microsoft-sub-1",
            "email": "old@example.com",
            "name": "Old Name",
            "avatar_url": "https://example.com/old.png",
        },
    )
    second = client.post(
        "/chat/state/users",
        headers=_auth_headers(),
        json={
            "provider": "microsoft-entra-id",
            "provider_subject": "microsoft-sub-1",
            "email": "new@example.com",
            "name": "New Name",
            "avatar_url": "https://example.com/new.png",
        },
    )

    # Then: the API-owned user id remains stable across profile changes.
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert second.json()["provider"] == "microsoft-entra-id"
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


def test_chat_state_replaces_regenerated_assistant_and_prunes_later_messages() -> None:
    # Given: a conversation with two completed turns.
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    setup_message_id = uuid4()
    first_user_id = uuid4()
    first_assistant_id = uuid4()
    second_user_id = uuid4()
    second_assistant_id = uuid4()
    new_response_id = uuid4()
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )
    for message_id, role, content in (
        (setup_message_id, "user", "Earlier setup"),
        (first_user_id, "user", "First question"),
        (first_assistant_id, "assistant", "Old first answer"),
        (second_user_id, "user", "Second question"),
        (second_assistant_id, "assistant", "Second answer"),
    ):
        db.messages[message_id] = {
            "content": content,
            "conversation_id": conversation_id,
            "created_at": db.now + timedelta(seconds=len(db.messages)),
            "id": message_id,
            "metadata": {},
            "response_id": None,
            "role": role,
            "updated_at": db.now,
        }

    # When: the first assistant answer is regenerated.
    replaced = client.put(
        f"/chat/state/conversations/{conversation_id}/messages/assistant/{first_assistant_id}/replacement/{new_response_id}",
        headers=_auth_headers(),
        json={
            "content": "New first answer",
            "metadata": {"responseId": str(new_response_id)},
            "retained_message_ids": [str(first_user_id), str(first_assistant_id)],
            "user_id": OWNER_ID,
        },
    )
    loaded = client.get(
        f"/chat/state/conversations/{conversation_id}",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: the original assistant id is preserved, prior messages survive, and later turns are pruned.
    assert replaced.status_code == 200
    assert replaced.json()["id"] == str(first_assistant_id)
    assert loaded.status_code == 200
    messages = loaded.json()["messages"]
    assert [message["id"] for message in messages] == [
        str(setup_message_id),
        str(first_user_id),
        str(first_assistant_id),
    ]
    assert messages[2]["content"] == "New first answer"
    assert messages[2]["response_id"] == str(new_response_id)


def test_chat_state_rejects_replacement_when_target_is_not_retained() -> None:
    # Given: a conversation with a completed assistant answer.
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    user_message_id = uuid4()
    assistant_message_id = uuid4()
    new_response_id = uuid4()
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )
    db.messages[user_message_id] = {
        "content": "Question",
        "conversation_id": conversation_id,
        "created_at": db.now,
        "id": user_message_id,
        "metadata": {},
        "response_id": None,
        "role": "user",
        "updated_at": db.now,
    }
    db.messages[assistant_message_id] = {
        "content": "Original answer",
        "conversation_id": conversation_id,
        "created_at": db.now + timedelta(seconds=1),
        "id": assistant_message_id,
        "metadata": {},
        "response_id": None,
        "role": "assistant",
        "updated_at": db.now,
    }

    # When: a direct state API caller omits the target assistant from retained ids.
    replaced = client.put(
        f"/chat/state/conversations/{conversation_id}/messages/assistant/{assistant_message_id}/replacement/{new_response_id}",
        headers=_auth_headers(),
        json={
            "content": "Unsafe replacement",
            "retained_message_ids": [str(user_message_id)],
            "user_id": OWNER_ID,
        },
    )

    # Then: replacement is rejected before any message is updated or pruned.
    assert replaced.status_code == 422
    assert replaced.json() == {"detail": "Replacement target must be retained"}
    assert db.messages[assistant_message_id]["content"] == "Original answer"
    assert user_message_id in db.messages
    assert assistant_message_id in db.messages


def test_chat_state_delete_removes_owned_conversation_and_messages() -> None:
    # Given: an authenticated owner has persisted conversation state.
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    message_id = uuid4()
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )
    client.post(
        f"/chat/state/conversations/{conversation_id}/messages/user",
        headers=_auth_headers(),
        json={"content": "delete me", "id": str(message_id), "user_id": OWNER_ID},
    )

    # When: the owner deletes the conversation through the state API.
    deleted = client.delete(
        f"/chat/state/conversations/{conversation_id}",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )
    loaded = client.get(
        f"/chat/state/conversations/{conversation_id}",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: the conversation and its messages are no longer visible.
    assert deleted.status_code == 200
    assert deleted.json()["id"] == str(conversation_id)
    assert loaded.status_code == 404
    assert conversation_id not in db.conversations
    assert message_id not in db.messages


def test_chat_state_lists_and_updates_owned_conversations() -> None:
    # Given: two conversations owned by the same authenticated web user.
    db = FakeChatDatabase()
    client = _client(db)
    first_conversation_id = uuid4()
    second_conversation_id = uuid4()
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={
            "id": str(first_conversation_id),
            "model": "model-a",
            "title": "First",
            "user_id": OWNER_ID,
        },
    )
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={
            "id": str(second_conversation_id),
            "model": "model-b",
            "title": "Second",
            "user_id": OWNER_ID,
        },
    )

    # When: the BFF renames the older conversation and asks for the server list.
    renamed = client.patch(
        f"/chat/state/conversations/{first_conversation_id}",
        headers=_auth_headers(),
        json={"title": "Renamed", "user_id": OWNER_ID},
    )
    listed = client.get(
        "/chat/state/conversations",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: the server-owned list reflects the rename and recency ordering.
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"
    assert listed.status_code == 200
    assert [row["id"] for row in listed.json()] == [
        str(first_conversation_id),
        str(second_conversation_id),
    ]


def test_chat_state_clear_all_removes_only_owned_conversations() -> None:
    # Given: two owners have persisted conversations and messages.
    db = FakeChatDatabase()
    client = _client(db)
    owned_conversation_id = uuid4()
    other_conversation_id = uuid4()
    owned_message_id = uuid4()
    for conversation_id, user_id in (
        (owned_conversation_id, OWNER_ID),
        (other_conversation_id, INTRUDER_ID),
    ):
        client.post(
            "/chat/state/conversations",
            headers=_auth_headers(),
            json={"id": str(conversation_id), "user_id": user_id},
        )
    client.post(
        f"/chat/state/conversations/{owned_conversation_id}/messages/user",
        headers=_auth_headers(),
        json={"content": "delete me", "id": str(owned_message_id), "user_id": OWNER_ID},
    )

    # When: one owner clears all of their conversations.
    cleared = client.delete(
        "/chat/state/conversations",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: only that owner's conversations and messages are removed.
    assert cleared.status_code == 200
    assert [row["id"] for row in cleared.json()] == [str(owned_conversation_id)]
    assert owned_conversation_id not in db.conversations
    assert owned_message_id not in db.messages
    assert other_conversation_id in db.conversations


def test_chat_state_wrong_user_cannot_delete_conversation() -> None:
    # Given: a conversation owned by one user.
    db = FakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    create_response = client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(conversation_id), "user_id": OWNER_ID},
    )

    # When: another user attempts to delete it.
    deleted = client.delete(
        f"/chat/state/conversations/{conversation_id}",
        headers=_auth_headers(),
        params={"user_id": INTRUDER_ID},
    )

    # Then: ownership failures look like missing state and the row remains.
    assert create_response.status_code == 200
    assert deleted.status_code == 404
    assert conversation_id in db.conversations


def test_chat_state_message_id_collision_is_not_cross_tenant_write() -> None:
    # Given: a victim conversation already has a user message id.
    db = FakeChatDatabase()
    client = _client(db)
    victim_conversation_id = uuid4()
    attacker_conversation_id = uuid4()
    shared_message_id = uuid4()
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(victim_conversation_id), "user_id": OWNER_ID},
    )
    client.post(
        f"/chat/state/conversations/{victim_conversation_id}/messages/user",
        headers=_auth_headers(),
        json={
            "content": "victim text",
            "id": str(shared_message_id),
            "user_id": OWNER_ID,
        },
    )
    client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={"id": str(attacker_conversation_id), "user_id": INTRUDER_ID},
    )

    # When: another owner tries to reuse that id in their own conversation.
    overwrite = client.post(
        f"/chat/state/conversations/{attacker_conversation_id}/messages/user",
        headers=_auth_headers(),
        json={
            "content": "attacker overwrite",
            "id": str(shared_message_id),
            "user_id": INTRUDER_ID,
        },
    )
    loaded = client.get(
        f"/chat/state/conversations/{victim_conversation_id}",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: the collision is rejected and the victim row is untouched.
    assert overwrite.status_code == 404
    assert loaded.json()["messages"][0]["content"] == "victim text"


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
