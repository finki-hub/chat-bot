from tests.feedback_fake import FakeFeedbackDatabase
from tests.feedback_test_support import (
    INTRUDER_ID,
    OWNER_ID,
    auth_headers,
    make_client,
    seed_owned_response,
)


def test_web_feedback_uses_server_owned_message_context() -> None:
    # Given: persisted chat state for a web-owned response.
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)

    # When: the browser submits feedback with forged metadata fields.
    response = client.post(
        "/chat/feedback",
        headers=auth_headers(),
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
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)

    # When: a different user submits feedback for that response id.
    response = client.post(
        "/chat/feedback",
        headers=auth_headers(),
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
