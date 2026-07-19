import json
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.data.db import get_db
from app.main import make_app
from app.utils.settings import Settings
from tests.feedback_fake import FakeFeedbackDatabase

OWNER_ID = "00000000-0000-4000-8000-000000000001"
INTRUDER_ID = "00000000-0000-4000-8000-000000000002"


def make_client(db: FakeFeedbackDatabase) -> TestClient:
    app = make_app(Settings(API_KEY="test-api-key", MCP_API_KEY="test-mcp-key"))
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def auth_headers() -> dict[str, str]:
    return {"x-api-key": "test-api-key"}


def seed_owned_response(db: FakeFeedbackDatabase) -> tuple[UUID, UUID]:
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
