from uuid import UUID

from fastapi.testclient import TestClient
from httpx2 import Response

from tests.feedback_fake import FakeFeedbackDatabase
from tests.feedback_test_support import (
    INTRUDER_ID,
    OWNER_ID,
    auth_headers,
    make_client,
    seed_owned_response,
)


def _retract_feedback(
    client: TestClient,
    response_id: UUID,
    *,
    user_id: str = OWNER_ID,
) -> Response:
    return client.request(
        "DELETE",
        "/chat/feedback",
        headers=auth_headers(),
        json={
            "client": "web",
            "response_id": str(response_id),
            "user_id": user_id,
        },
    )


def _assert_retracted(
    db: FakeFeedbackDatabase,
    response_id: UUID,
    response: Response,
) -> None:
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
    response = _retract_feedback(client, response_id)

    # Then: the row and only the feedback metadata are removed.
    _assert_retracted(db, response_id, response)


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
    response = _retract_feedback(client, response_id, user_id=INTRUDER_ID)

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
    response = _retract_feedback(client, response_id)

    # Then: the empty state is confirmed without disturbing other metadata.
    _assert_retracted(db, response_id, response)


def _rate(client: TestClient, response_id: UUID, **fields: object) -> Response:
    return client.post(
        "/chat/feedback",
        headers=auth_headers(),
        json={
            "client": "web",
            "response_id": str(response_id),
            "user_id": OWNER_ID,
            **fields,
        },
    )


def test_a_dislike_keeps_its_reason_and_comment() -> None:
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)

    response = _rate(
        client,
        response_id,
        feedback_type="dislike",
        reason="outdated",
        comment="  Цената е од минатата година.  ",
    )

    assert response.status_code == 200
    [stored] = db.feedback.values()
    assert stored["reason"] == "outdated"
    assert stored["comment"] == "Цената е од минатата година."


def test_a_later_like_clears_the_dislike_reason() -> None:
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)

    _rate(client, response_id, feedback_type="dislike", reason="wrong")
    _rate(client, response_id, feedback_type="like", reason="wrong")

    [stored] = db.feedback.values()
    assert stored["feedback_type"] == "like"
    assert stored["reason"] is None


def test_an_unknown_reason_is_rejected() -> None:
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)

    response = _rate(client, response_id, feedback_type="dislike", reason="meh")

    assert response.status_code == 422
    assert db.feedback == {}
