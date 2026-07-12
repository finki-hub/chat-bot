from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from app.data.chat_persistence import (
    ChatMessageConflictError,
    clear_active_stream_if_current,
    clear_stale_active_streams,
    create_conversation,
    get_conversation_owner,
    list_conversations,
    load_conversation,
    set_active_stream,
    update_conversation,
    upsert_message,
)
from app.schemas.chat_persistence import (
    ActiveStreamStatus,
    ChatConversationCreate,
    ChatConversationUpdate,
    ChatMessageRole,
    ChatMessageUpsert,
    JsonValue,
)
from tests.chat_persistence_fake import FakeChatDatabase

OWNER_ID = UUID("00000000-0000-4000-8000-000000000001")
INTRUDER_ID = UUID("00000000-0000-4000-8000-000000000002")


@pytest.mark.anyio
async def test_chat_persistence_happy_path_orders_messages_and_clears_current_stream() -> (
    None
):
    # Given: a new web chat conversation owned by one anonymous user.
    db = FakeChatDatabase()
    response_id = uuid4()
    assistant_parts: list[JsonValue] = [
        {"state": "done", "text": "Check the enrollment rules.", "type": "reasoning"},
        {"state": "done", "text": "Submit the semester form.", "type": "text"},
    ]
    conversation = await create_conversation(
        db,
        ChatConversationCreate(
            id=uuid4(),
            user_id=OWNER_ID,
            model="claude-sonnet-5",
            title="Enrollment",
        ),
    )
    await upsert_message(
        db,
        ChatMessageUpsert(
            id=uuid4(),
            conversation_id=conversation.id,
            role=ChatMessageRole.USER,
            content="How do I enroll?",
        ),
    )
    active = await set_active_stream(
        db,
        conversation_id=conversation.id,
        user_id=OWNER_ID,
        active_stream_id=response_id,
        active_response_id=response_id,
        active_status=ActiveStreamStatus.STREAMING,
    )

    # When: the assistant answer is persisted and the matching active stream is cleared.
    assistant = await upsert_message(
        db,
        ChatMessageUpsert(
            id=uuid4(),
            conversation_id=conversation.id,
            role=ChatMessageRole.ASSISTANT,
            content="Submit the semester form.",
            response_id=response_id,
            metadata={"responseId": str(response_id)},
            parts=assistant_parts,
        ),
    )
    cleared = await clear_active_stream_if_current(
        db,
        conversation_id=conversation.id,
        user_id=OWNER_ID,
        active_stream_id=response_id,
    )
    loaded = await load_conversation(
        db,
        conversation_id=conversation.id,
        user_id=OWNER_ID,
    )

    # Then: response_id compatibility and message order are preserved.
    assert active is not None
    assert active.active_stream_id == response_id
    assert assistant.response_id == response_id
    assert cleared is not None
    assert cleared.active_stream_id is None
    assert loaded is not None
    assert [message.role for message in loaded.messages] == [
        ChatMessageRole.USER,
        ChatMessageRole.ASSISTANT,
    ]
    assert loaded.messages[1].metadata == {"responseId": str(response_id)}
    assert loaded.messages[1].parts == assistant_parts


@pytest.mark.anyio
async def test_chat_persistence_clear_with_stale_stream_id_keeps_newer_active_stream() -> (
    None
):
    # Given: an active stream was superseded by a newer one.
    db = FakeChatDatabase()
    conversation = await create_conversation(
        db,
        ChatConversationCreate(id=uuid4(), user_id=OWNER_ID, model=None, title=None),
    )
    stale_stream_id = uuid4()
    current_stream_id = uuid4()
    for stream_id in (stale_stream_id, current_stream_id):
        await set_active_stream(
            db,
            conversation_id=conversation.id,
            user_id=OWNER_ID,
            active_stream_id=stream_id,
            active_response_id=stream_id,
            active_status=ActiveStreamStatus.STREAMING,
        )

    # When: a stale clear arrives late.
    stale_clear = await clear_active_stream_if_current(
        db,
        conversation_id=conversation.id,
        user_id=OWNER_ID,
        active_stream_id=stale_stream_id,
    )
    loaded = await load_conversation(
        db,
        conversation_id=conversation.id,
        user_id=OWNER_ID,
    )

    # Then: the newer stream remains current.
    assert stale_clear is None
    assert loaded is not None
    assert loaded.conversation.active_stream_id == current_stream_id


@pytest.mark.anyio
async def test_chat_persistence_lists_updates_owner_and_clears_stale_streams() -> None:
    # Given: two conversations and one stale active stream.
    db = FakeChatDatabase()
    first = await create_conversation(
        db,
        ChatConversationCreate(
            id=uuid4(),
            user_id=OWNER_ID,
            model="old-model",
            title="Old",
        ),
    )
    second = await create_conversation(
        db,
        ChatConversationCreate(
            id=uuid4(),
            user_id=OWNER_ID,
            model=None,
            title="Second",
        ),
    )
    await set_active_stream(
        db,
        conversation_id=first.id,
        user_id=OWNER_ID,
        active_stream_id=uuid4(),
        active_response_id=uuid4(),
        active_status=ActiveStreamStatus.STREAMING,
    )
    db.conversations[first.id]["updated_at"] = db.now - timedelta(hours=2)

    # When: metadata is updated, ownership is queried, and stale active streams are cleared.
    updated = await update_conversation(
        db,
        conversation_id=second.id,
        user_id=OWNER_ID,
        update=ChatConversationUpdate(model="new-model", title="Updated"),
    )
    owner = await get_conversation_owner(db, first.id)
    missing_for_wrong_owner = await load_conversation(
        db,
        conversation_id=first.id,
        user_id=INTRUDER_ID,
    )
    cleared_count = await clear_stale_active_streams(
        db,
        stale_before=db.now - timedelta(hours=1),
    )
    conversations = await list_conversations(db, user_id=OWNER_ID, limit=10)

    # Then: only owner-visible rows are returned and stale active state is removed.
    assert updated is not None
    assert updated.model == "new-model"
    assert updated.title == "Updated"
    assert owner == OWNER_ID
    assert missing_for_wrong_owner is None
    assert cleared_count == 1
    assert db.conversations[first.id]["active_stream_id"] is None
    assert [conversation.id for conversation in conversations] == [second.id, first.id]


@pytest.mark.anyio
async def test_chat_persistence_rejects_message_id_collision_across_conversations() -> (
    None
):
    # Given: two users have separate conversations and one persisted victim message id.
    db = FakeChatDatabase()
    victim_conversation = await create_conversation(
        db,
        ChatConversationCreate(id=uuid4(), user_id=OWNER_ID, model=None, title=None),
    )
    attacker_conversation = await create_conversation(
        db,
        ChatConversationCreate(id=uuid4(), user_id=INTRUDER_ID, model=None, title=None),
    )
    victim_message_id = uuid4()
    await upsert_message(
        db,
        ChatMessageUpsert(
            id=victim_message_id,
            conversation_id=victim_conversation.id,
            role=ChatMessageRole.USER,
            content="victim text",
        ),
    )

    # When / Then: reusing that id in another conversation is rejected and unchanged.
    with pytest.raises(ChatMessageConflictError):
        await upsert_message(
            db,
            ChatMessageUpsert(
                id=victim_message_id,
                conversation_id=attacker_conversation.id,
                role=ChatMessageRole.USER,
                content="attacker overwrite",
            ),
        )

    loaded = await load_conversation(
        db,
        conversation_id=victim_conversation.id,
        user_id=OWNER_ID,
    )

    assert loaded is not None
    assert loaded.messages[0].content == "victim text"


def test_chat_persistence_rejects_malformed_role_and_status() -> None:
    # Given / When / Then: malformed boundary inputs never reach SQL.
    with pytest.raises(ValueError, match="Input should be"):
        ChatMessageUpsert.model_validate(
            {
                "content": "bad",
                "conversation_id": uuid4(),
                "id": uuid4(),
                "role": "system",
            },
        )

    with pytest.raises(ValueError, match="Input should be"):
        ChatConversationUpdate.model_validate({"active_status": "lost"})
