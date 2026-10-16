from uuid import uuid4

from tests.discord_feedback_fake import DiscordFeedbackDatabase
from tests.feedback_test_support import auth_headers, make_client, seed_owned_response


def _pending_case():
    db = DiscordFeedbackDatabase()
    owner, rater = uuid4(), uuid4()
    for user_id, subject, generation in ((owner, "111", 3), (rater, "222", 7)):
        db.users[("discord", subject)] = {
            "id": user_id,
            "provider": "discord",
            "provider_subject": subject,
            "history_generation": generation,
        }
    response_id = uuid4()
    payload = {
        "response_id": str(response_id),
        "client": "discord",
        "user_id": "222",
        "feedback_type": "like",
    }
    return db, owner, rater, response_id, payload


def _persist(db, owner, response_id):
    conversation, _ = seed_owned_response(db)
    db.conversations[conversation]["user_id"] = owner
    messages = list(db.messages.values())
    db.messages.clear()
    for message in messages:
        # The existing web fixture has distinct dictionary keys vs message ids.
        db.messages[message["id"]] = message
        if message["role"] == "assistant":
            message["response_id"] = response_id
            message["metadata"] = {
                "interface": "discord",
                "inferenceModel": "stored-model",
            }


def test_pending_feedback_retry_token_cannot_cross_rater_history_deletion():
    db, owner, rater, response_id, payload = _pending_case()
    client = make_client(db)
    pending = client.post("/chat/feedback", headers=auth_headers(), json=payload)
    assert pending.status_code == 404
    token = pending.headers["X-Feedback-Generation"]
    assert int(token) == 7
    deleted = client.delete(
        f"/chat/state/users/{rater}/history", headers=auth_headers()
    )
    assert deleted.status_code == 200
    assert db.users[("discord", "222")]["history_generation"] == 8
    # A stale token is terminal even while the response is still pending.
    retry_headers = {**auth_headers(), "X-Feedback-Generation": token}
    assert (
        client.post("/chat/feedback", headers=retry_headers, json=payload).status_code
        == 409
    )
    _persist(db, owner, response_id)
    retry = client.post("/chat/feedback", headers=retry_headers, json=payload)
    assert retry.status_code == 409
    assert "X-Feedback-Generation" not in retry.headers
    assert db.feedback == {}


def test_pending_feedback_retry_same_generation_succeeds_without_token_refresh():
    db, owner, _, response_id, payload = _pending_case()
    client = make_client(db)
    pending = client.post("/chat/feedback", headers=auth_headers(), json=payload)
    token = pending.headers["X-Feedback-Generation"]
    retry_headers = {**auth_headers(), "X-Feedback-Generation": token}
    still_pending = client.post("/chat/feedback", headers=retry_headers, json=payload)
    assert still_pending.status_code == 404
    assert still_pending.headers["X-Feedback-Generation"] == token
    _persist(db, owner, response_id)
    retry = client.post("/chat/feedback", headers=retry_headers, json=payload)
    assert retry.status_code == 200
    assert len(db.feedback) == 1
    assert next(iter(db.feedback))[2] == "222"


def test_missing_rater_is_not_retryable_even_with_supplied_generation():
    db, _, _, _, payload = _pending_case()
    db.users.pop(("discord", "222"))
    result = make_client(db).post(
        "/chat/feedback",
        headers={**auth_headers(), "X-Feedback-Generation": "7"},
        json=payload,
    )
    assert result.status_code == 404
    assert "X-Feedback-Generation" not in result.headers
    assert db.feedback == {}
