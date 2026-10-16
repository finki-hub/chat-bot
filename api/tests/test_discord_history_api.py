from uuid import uuid4

from tests.feedback_fake import FakeFeedbackDatabase
from tests.feedback_test_support import auth_headers, make_client


def test_history_delete_empty_account_advances_fence_erases_orphans_and_counts():
    db = FakeFeedbackDatabase()
    user = uuid4()
    db.users[("discord", "111")] = {
        "id": user,
        "provider": "discord",
        "provider_subject": "111",
        "history_generation": 4,
    }
    for namespace, subject in (
        ("discord", "111"),
        ("web", str(user)),
        ("web", "111"),
        ("discord", "222"),
    ):
        db.feedback[(uuid4(), namespace, subject)] = {}
    client = make_client(db)
    response = client.delete(
        f"/chat/state/users/{user}/history", headers=auth_headers()
    )
    assert response.status_code == 200
    assert response.json() == {"deleted_conversations": 0, "deleted_feedback": 2}
    assert db.users[("discord", "111")]["history_generation"] == 5
    assert len(db.feedback) == 2
    again = client.delete(f"/chat/state/users/{user}/history", headers=auth_headers())
    assert again.json() == {"deleted_conversations": 0, "deleted_feedback": 0}
    assert db.users[("discord", "111")]["history_generation"] == 6


def test_history_delete_requires_service_key_and_missing_account_is_safe():
    db = FakeFeedbackDatabase()
    client = make_client(db)
    path = f"/chat/state/users/{uuid4()}/history"
    assert client.delete(path).status_code == 401
    assert client.delete(path, headers=auth_headers()).json() == {
        "deleted_conversations": 0,
        "deleted_feedback": 0,
    }
