from tests.feedback_fake import FakeFeedbackDatabase
from tests.feedback_test_support import (
    INTRUDER_ID,
    OWNER_ID,
    auth_headers,
    make_client,
    seed_owned_response,
)


def test_web_feedback_is_persisted_with_one_database_statement() -> None:
    # Given: persisted chat state for a web-owned response.
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)

    # When: the browser submits feedback.
    response = client.post(
        "/chat/feedback",
        headers=auth_headers(),
        json={
            "client": "web",
            "feedback_type": "like",
            "response_id": str(response_id),
            "user_id": OWNER_ID,
        },
    )

    # Then: ownership, metadata, and the feedback row are handled atomically.
    assert response.status_code == 200
    assert db.fetchrow_calls == 1


def test_web_feedback_can_be_retracted() -> None:
    # Given: an owned response with persisted feedback and unrelated metadata.
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)
    submitted = client.post(
        "/chat/feedback",
        headers=auth_headers(),
        json={
            "client": "web",
            "feedback_type": "like",
            "response_id": str(response_id),
            "user_id": OWNER_ID,
        },
    )
    assert submitted.status_code == 200

    # When: the owner retracts the feedback.
    response = client.request(
        "DELETE",
        "/chat/feedback",
        headers=auth_headers(),
        json={
            "client": "web",
            "response_id": str(response_id),
            "user_id": OWNER_ID,
        },
    )

    # Then: the row and only the feedback metadata are removed.
    assert response.status_code == 200
    assert response.json() == {
        "feedback_type": None,
        "response_id": str(response_id),
    }
    assert db.feedback == {}
    assistant = next(
        row for row in db.messages.values() if row["response_id"] == response_id
    )
    assert assistant["metadata"] == {"inferenceModel": "server-model"}


def test_web_feedback_retraction_rejects_unowned_response_id() -> None:
    # Given: an owned response with persisted feedback.
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)
    submitted = client.post(
        "/chat/feedback",
        headers=auth_headers(),
        json={
            "client": "web",
            "feedback_type": "dislike",
            "response_id": str(response_id),
            "user_id": OWNER_ID,
        },
    )
    assert submitted.status_code == 200

    # When: a different user tries to retract the feedback.
    response = client.request(
        "DELETE",
        "/chat/feedback",
        headers=auth_headers(),
        json={
            "client": "web",
            "response_id": str(response_id),
            "user_id": INTRUDER_ID,
        },
    )

    # Then: the response is hidden and both persisted representations remain.
    assert response.status_code == 404
    stored = next(iter(db.feedback.values()))
    assert stored["feedback_type"] == "dislike"
    assistant = next(
        row for row in db.messages.values() if row["response_id"] == response_id
    )
    assert assistant["metadata"] == {
        "feedback": "dislike",
        "inferenceModel": "server-model",
    }


def test_web_feedback_retraction_is_idempotent() -> None:
    # Given: an owned response without a feedback row.
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)

    # When: the owner retracts feedback that is already absent.
    response = client.request(
        "DELETE",
        "/chat/feedback",
        headers=auth_headers(),
        json={
            "client": "web",
            "response_id": str(response_id),
            "user_id": OWNER_ID,
        },
    )

    # Then: the empty state is confirmed without disturbing other metadata.
    assert response.status_code == 200
    assert response.json() == {
        "feedback_type": None,
        "response_id": str(response_id),
    }
    assert db.feedback == {}
    assistant = next(
        row for row in db.messages.values() if row["response_id"] == response_id
    )
    assert assistant["metadata"] == {"inferenceModel": "server-model"}
