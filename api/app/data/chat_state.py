import json
from uuid import UUID

from app.data.chat_persistence import ChatPersistenceDatabase
from app.data.chat_rows import conversation_from_row, message_from_row
from app.schemas.chat_persistence import (
    ChatConversation,
    ChatMessage,
    ChatMessageUpsert,
)


async def upsert_assistant_message_by_response_id(
    db: ChatPersistenceDatabase,
    message: ChatMessageUpsert,
) -> ChatMessage:
    row = await db.fetchrow(
        """
        INSERT INTO chat_message (id, conversation_id, role, content, response_id, metadata)
        VALUES ($3, $1, 'assistant', $4, $2, $5::jsonb)
        ON CONFLICT (conversation_id, response_id)
        WHERE response_id IS NOT NULL AND role = 'assistant'
        DO UPDATE SET
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        RETURNING *
        """,
        message.conversation_id,
        message.response_id,
        message.id,
        message.content,
        json.dumps(message.metadata),
    )
    if row is None:
        raise RuntimeError("assistant chat_message upsert returned no row")
    return message_from_row(row)


async def mark_active_stream_stopped_if_current(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
    active_stream_id: UUID,
) -> ChatConversation | None:
    row = await db.fetchrow(
        """
        UPDATE chat_conversation
        SET active_status = 'stopped',
            updated_at = NOW()
        WHERE id = $1 AND user_id = $2 AND active_stream_id = $3
        RETURNING *
        """,
        conversation_id,
        user_id,
        active_stream_id,
    )
    return None if row is None else conversation_from_row(row)
