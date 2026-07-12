import json
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.data.db import get_db
from app.main import make_app
from app.utils.settings import Settings
from tests.chat_persistence_fake import FakeChatDatabase

OWNER_ID = "00000000-0000-4000-8000-000000000001"
INTRUDER_ID = "00000000-0000-4000-8000-000000000002"


def _client(db: FakeChatDatabase) -> TestClient:
    app = make_app(Settings(API_KEY="test-api-key", MCP_API_KEY="test-mcp-key"))
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"x-api-key": "test-api-key"}


def _seed_owned_response(db: FakeChatDatabase) -> tuple[UUID, UUID]:
    conversation_id = uuid4()
    response_id = uuid4()
    db.conversations[conversation_id] = {
        "active_response_id": None,
        "active_status": None,
        "active_stream_id": None,
        "created_at": db.now,
        "id": conversation_id,
        "model": "claude-sonnet-5",
        "title": "Enrollment",
        "updated_at": db.now,
        "user_id": UUID(OWNER_ID),
    }
    db.messages[uuid4()] = {
        "content": "Server question",
        "conversation_id": conversation_id,
        "created_at": db.now,
        "id": uuid4(),
        "metadata": {},
        "response_id": None,
        "role": "user",
        "updated_at": db.now,
    }
    db.messages[uuid4()] = {
        "content": "Server answer",
        "conversation_id": conversation_id,
        "created_at": db.now,
        "id": uuid4(),
        "metadata": json.dumps({"inferenceModel": "server-model"}),
        "parts": [
            {"state": "done", "text": "Stored reasoning", "type": "reasoning"},
            {"state": "done", "text": "Server answer", "type": "text"},
        ],
        "response_id": response_id,
        "role": "assistant",
        "updated_at": db.now,
    }
    return conversation_id, response_id


def test_web_feedback_uses_server_owned_message_context() -> None:
    # Given: persisted chat state for a web-owned response.
    db = FakeChatDatabase()
    client = _client(db)
    _, response_id = _seed_owned_response(db)

    # When: the browser submits feedback with forged metadata fields.
    response = client.post(
        "/chat/feedback",
        headers=_auth_headers(),
        json={
            "answer_text": "forged answer",
            "client": "web",
            "feedback_type": "like",
            "inference_model": "forged-model",
            "question_text": "forged question",
            "response_id": str(response_id),
            "user_id": OWNER_ID,
        },
    )

    # Then: stored feedback uses server-owned chat state, not the browser fields.
    assert response.status_code == 200
    stored = next(iter(db.feedback.values()))
    assert stored["answer_text"] == "Server answer"
    assert stored["question_text"] == "Server question"
    assert stored["inference_model"] == "server-model"
    assistant = next(
        row for row in db.messages.values() if row["response_id"] == response_id
    )
    assert assistant["metadata"] == {
        "feedback": "like",
        "inferenceModel": "server-model",
    }
    assert assistant["parts"] == [
        {"state": "done", "text": "Stored reasoning", "type": "reasoning"},
        {"state": "done", "text": "Server answer", "type": "text"},
    ]


def test_web_feedback_rejects_unowned_response_id() -> None:
    # Given: persisted chat state owned by another web user.
    db = FakeChatDatabase()
    client = _client(db)
    _, response_id = _seed_owned_response(db)

    # When: a different user submits feedback for that response id.
    response = client.post(
        "/chat/feedback",
        headers=_auth_headers(),
        json={
            "client": "web",
            "feedback_type": "like",
            "response_id": str(response_id),
            "user_id": INTRUDER_ID,
        },
    )

    # Then: the response is hidden and no feedback row is written.
    assert response.status_code == 404
    assert db.feedback == {}
