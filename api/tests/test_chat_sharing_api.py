from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.data.db import get_db
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase

OWNER_ID = "00000000-0000-4000-8000-000000000001"
INTRUDER_ID = "00000000-0000-4000-8000-000000000002"


class SharingFakeChatDatabase(FakeChatDatabase):
    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        if "SELECT share_token FROM chat_conversation" in query:
            conversation_id, user_id = args
            conversation = self.conversations.get(conversation_id)
            if conversation is None or conversation["user_id"] != user_id:
                return None
            return {"share_token": conversation.get("share_token")}

        if "SET share_token = NULL" in query:
            conversation_id, user_id = args
            conversation = self.conversations.get(conversation_id)
            if conversation is None or conversation["user_id"] != user_id:
                return None
            conversation["share_token"] = None
            return {"id": conversation_id}

        if "RETURNING share_token" in query:
            conversation_id, user_id = args
            conversation = self.conversations.get(conversation_id)
            if conversation is None or conversation["user_id"] != user_id:
                return None
            share_token = conversation.get("share_token")
            if not isinstance(share_token, UUID):
                share_token = uuid4()
                conversation["share_token"] = share_token
            return {"share_token": share_token}

        if "WHERE share_token = $1" in query:
            (share_token,) = args
            return next(
                (
                    conversation
                    for conversation in self.conversations.values()
                    if conversation.get("share_token") == share_token
                ),
                None,
            )

        return await super().fetchrow(query, *args)


def _client(db: SharingFakeChatDatabase) -> TestClient:
    app = make_app(
        Settings(
            API_KEY="test-api-key",
            CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
            MCP_API_KEY="test-mcp-key",
        ),
    )
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"x-api-key": "test-api-key"}


def _create_conversation(client: TestClient, conversation_id: UUID) -> None:
    response = client.post(
        "/chat/state/conversations",
        headers=_auth_headers(),
        json={
            "id": str(conversation_id),
            "title": "Shared enrollment chat",
            "user_id": OWNER_ID,
        },
    )
    assert response.status_code == 200


def _share_conversation(client: TestClient, conversation_id: UUID) -> str:
    response = client.post(
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        json={"user_id": OWNER_ID},
    )
    assert response.status_code == 200
    return response.json()["share_token"]


def test_owner_gets_stable_share_token() -> None:
    # Given: an authenticated BFF owns a persisted conversation.
    client = _client(SharingFakeChatDatabase())
    conversation_id = uuid4()
    _create_conversation(client, conversation_id)

    # When: the owner shares the same conversation twice.
    first = client.post(
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        json={"user_id": OWNER_ID},
    )
    second = client.post(
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        json={"user_id": OWNER_ID},
    )

    # Then: both requests return one stable opaque token.
    assert first.status_code == 200
    assert second.status_code == 200
    assert UUID(first.json()["share_token"])
    assert second.json() == first.json()


def test_non_owner_cannot_create_share_link() -> None:
    # Given: a conversation owned by another user.
    client = _client(SharingFakeChatDatabase())
    conversation_id = uuid4()
    _create_conversation(client, conversation_id)

    # When: an intruder asks the BFF-owned endpoint to share it.
    response = client.post(
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        json={"user_id": INTRUDER_ID},
    )

    # Then: ownership failure is indistinguishable from a missing conversation.
    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}


def test_owner_reads_unshared_conversation_status() -> None:
    # Given: an owned conversation without a share token.
    client = _client(SharingFakeChatDatabase())
    conversation_id = uuid4()
    _create_conversation(client, conversation_id)

    # When: the owner checks whether it is shared.
    response = client.get(
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: the API reports the owned conversation as unshared.
    assert response.status_code == 204
    assert response.content == b""


def test_owner_reads_shared_conversation_status() -> None:
    # Given: an owned conversation with an active share token.
    client = _client(SharingFakeChatDatabase())
    conversation_id = uuid4()
    _create_conversation(client, conversation_id)
    share_token = _share_conversation(client, conversation_id)

    # When: the owner checks whether it is shared.
    response = client.get(
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        params={"user_id": OWNER_ID},
    )

    # Then: the API returns the persisted token needed to rebuild the share URL.
    assert response.status_code == 200
    assert response.json() == {"share_token": share_token}


def test_owner_revokes_share_link() -> None:
    # Given: an owned conversation with an active public link.
    client = _client(SharingFakeChatDatabase())
    conversation_id = uuid4()
    _create_conversation(client, conversation_id)
    share_token = _share_conversation(client, conversation_id)

    # When: the owner revokes the share.
    response = client.request(
        "DELETE",
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        json={"user_id": OWNER_ID},
    )

    # Then: revocation succeeds and the old bearer link no longer resolves.
    assert response.status_code == 204
    loaded = client.get(
        f"/chat/state/shared/{share_token}",
        headers=_auth_headers(),
    )
    assert loaded.status_code == 404


def test_non_owner_cannot_revoke_share_link() -> None:
    # Given: another user's conversation has an active public link.
    client = _client(SharingFakeChatDatabase())
    conversation_id = uuid4()
    _create_conversation(client, conversation_id)
    share_token = _share_conversation(client, conversation_id)

    # When: an intruder attempts to revoke it.
    response = client.request(
        "DELETE",
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        json={"user_id": INTRUDER_ID},
    )

    # Then: ownership remains hidden and the original link still works.
    assert response.status_code == 404
    loaded = client.get(
        f"/chat/state/shared/{share_token}",
        headers=_auth_headers(),
    )
    assert loaded.status_code == 200


def test_resharing_after_revoke_uses_fresh_token() -> None:
    # Given: an owner revoked an existing conversation share.
    client = _client(SharingFakeChatDatabase())
    conversation_id = uuid4()
    _create_conversation(client, conversation_id)
    previous_token = _share_conversation(client, conversation_id)
    revoked = client.request(
        "DELETE",
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        json={"user_id": OWNER_ID},
    )
    assert revoked.status_code == 204

    # When: the owner shares the conversation again.
    current_token = _share_conversation(client, conversation_id)

    # Then: the new link does not reactivate the revoked bearer token.
    assert current_token != previous_token


def test_shared_conversation_loads_by_token() -> None:
    # Given: an owner shared a conversation with a persisted user message.
    db = SharingFakeChatDatabase()
    client = _client(db)
    conversation_id = uuid4()
    message_id = uuid4()
    _create_conversation(client, conversation_id)
    message = client.post(
        f"/chat/state/conversations/{conversation_id}/messages/user",
        headers=_auth_headers(),
        json={
            "content": "What are the enrollment requirements?",
            "id": str(message_id),
            "user_id": OWNER_ID,
        },
    )
    shared = client.post(
        f"/chat/state/conversations/{conversation_id}/share",
        headers=_auth_headers(),
        json={"user_id": OWNER_ID},
    )
    share_token = shared.json()["share_token"]

    # When: the BFF resolves the conversation through the share token.
    response = client.get(
        f"/chat/state/shared/{share_token}",
        headers=_auth_headers(),
    )

    # Then: the internal response contains the ordered persisted transcript.
    assert message.status_code == 200
    assert shared.status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert body["conversation"]["title"] == "Shared enrollment chat"
    assert [item["id"] for item in body["messages"]] == [str(message_id)]
    assert body["messages"][0]["content"] == "What are the enrollment requirements?"


def test_unknown_share_token_is_not_found() -> None:
    # Given: no conversation has this share token.
    client = _client(SharingFakeChatDatabase())

    # When: the BFF resolves the unknown token.
    response = client.get(
        f"/chat/state/shared/{uuid4()}",
        headers=_auth_headers(),
    )

    # Then: it receives the generic conversation-not-found response.
    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}


def test_share_endpoints_require_api_key() -> None:
    # Given: a persisted conversation and a caller without the BFF API key.
    client = _client(SharingFakeChatDatabase())
    conversation_id = uuid4()
    _create_conversation(client, conversation_id)

    # When: share creation and reading omit the key.
    created = client.post(
        f"/chat/state/conversations/{conversation_id}/share",
        json={"user_id": OWNER_ID},
    )
    loaded = client.get(f"/chat/state/shared/{uuid4()}")

    # Then: both internal operations remain private to trusted API clients.
    assert created.status_code == 401
    assert loaded.status_code == 401
