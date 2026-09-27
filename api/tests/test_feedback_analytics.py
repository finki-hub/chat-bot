from posthog import Posthog

from app.utils import posthog_client
from tests.feedback_fake import FakeFeedbackDatabase
from tests.feedback_test_support import (
    OWNER_ID,
    auth_headers,
    make_client,
    seed_owned_response,
)


def test_feedback_events_join_by_response_without_authenticated_identity(monkeypatch):
    events = []

    def before_send(event):
        events.append(event)

    # Exercise SDK enrichment as well as our wrapper, with network sending disabled.
    sdk = Posthog("test-project", send=False, before_send=before_send)
    monkeypatch.setattr(posthog_client._state, "client", sdk)
    db = FakeFeedbackDatabase()
    client = make_client(db)
    _, response_id = seed_owned_response(db)
    payload = {
        "client": "web",
        "feedback_type": "like",
        "response_id": str(response_id),
        "user_id": OWNER_ID,
    }
    assert (
        client.post("/chat/feedback", headers=auth_headers(), json=payload).status_code
        == 200
    )
    assert (
        client.request(
            "DELETE",
            "/chat/feedback",
            headers=auth_headers(),
            json={
                "client": "web",
                "response_id": str(response_id),
                "user_id": OWNER_ID,
            },
        ).status_code
        == 200
    )
    feedback_events = [
        event for event in events if event["event"].startswith("chat_feedback")
    ]
    assert len(feedback_events) == 2
    assert OWNER_ID not in repr(events)
    for event in feedback_events:
        assert event["distinct_id"] == "feedback"
        assert event["properties"]["response_id"] == str(response_id)
        assert event["properties"]["$process_person_profile"] is False
        assert OWNER_ID not in repr(event)
        assert "Server answer" not in repr(event)
        assert "Server question" not in repr(event)
    sdk.shutdown()
