from copy import deepcopy
from uuid import UUID, uuid4

import pytest

from tests.feedback_fake import FakeFeedbackDatabase
from tests.feedback_test_support import (
    OWNER_ID,
    auth_headers,
    make_client,
    seed_owned_response,
)


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize(
    "interfaces", [("web", "web"), ("web", "discord"), ("discord", "web")]
)
def test_delete_scopes_colliding_feedback_by_owner_and_interface(bulk, interfaces):
    db = FakeFeedbackDatabase()
    conversation, response_id = seed_owned_response(db)
    other_user, other_conversation = uuid4(), uuid4()
    db.conversations[other_conversation] = {
        **deepcopy(db.conversations[conversation]),
        "id": other_conversation,
        "user_id": other_user,
    }
    for message in list(db.messages.values()):
        if message["role"] == "assistant":
            message["metadata"] = {"interface": interfaces[0]}
        copied = {
            **deepcopy(message),
            "id": uuid4(),
            "conversation_id": other_conversation,
        }
        if copied["role"] == "assistant":
            copied["metadata"] = {"interface": interfaces[1]}
        db.messages[copied["id"]] = copied
    db.users[("google", "owner")] = {
        "id": UUID(OWNER_ID),
        "provider": "google",
        "provider_subject": "owner",
        "history_generation": 0,
    }
    mine = (response_id, "web", OWNER_ID)
    theirs = (response_id, "web", str(other_user))
    discord_copy = (response_id, "discord", "333")
    own_orphan = (uuid4(), "web", OWNER_ID)
    for key in (mine, theirs, discord_copy, own_orphan):
        db.feedback[key] = {}
    path = (
        "/chat/state/conversations"
        if bulk
        else f"/chat/state/conversations/{conversation}"
    )
    response = make_client(db).delete(
        path, headers=auth_headers(), params={"user_id": OWNER_ID}
    )
    assert response.status_code == 200
    assert mine not in db.feedback
    assert theirs in db.feedback
    assert (discord_copy in db.feedback) is (interfaces[0] == "web")
    assert (own_orphan in db.feedback) is not bulk
    assert other_conversation in db.conversations
