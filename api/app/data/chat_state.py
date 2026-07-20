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
        INSERT INTO chat_message (id, conversation_id, role, content, response_id, metadata, parts)
        VALUES ($3, $1, 'assistant', $4, $2, $5::jsonb, $6::jsonb)
        ON CONFLICT (conversation_id, response_id)
        WHERE response_id IS NOT NULL AND role = 'assistant'
        DO UPDATE SET
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            parts = EXCLUDED.parts,
            updated_at = NOW()
        RETURNING *
        """,
        message.conversation_id,
        message.response_id,
        message.id,
        message.content,
        json.dumps(message.metadata),
        None if message.parts is None else json.dumps(message.parts),
    )
    if row is None:
        raise RuntimeError("assistant chat_message upsert returned no row")
    return message_from_row(row)


async def replace_assistant_message_and_prune_after(
    db: ChatPersistenceDatabase,
    message: ChatMessageUpsert,
    *,
    retained_message_ids: list[UUID],
    user_id: UUID,
) -> ChatMessage | None:
    row = await db.fetchrow(
        """
        WITH target AS (
            SELECT assistant.id, assistant.conversation_id, assistant.created_at
            FROM chat_message assistant
            JOIN chat_conversation conversation
              ON conversation.id = assistant.conversation_id
            WHERE assistant.id = $1
              AND assistant.conversation_id = $2
              AND assistant.role = 'assistant'
              AND conversation.user_id = $3
        ), updated AS (
            UPDATE chat_message assistant
            SET content = $4,
                response_id = $5,
                metadata = $6::jsonb,
                parts = $7::jsonb,
                updated_at = NOW()
            FROM target
            WHERE assistant.id = target.id
            RETURNING assistant.*
        ), deleted AS (
            DELETE FROM chat_message stale
            USING target
            WHERE stale.conversation_id = target.conversation_id
              AND stale.created_at > target.created_at
              AND NOT (stale.id = ANY($8::uuid[]))
            RETURNING stale.id
        )
        SELECT * FROM updated
        """,
        message.id,
        message.conversation_id,
        user_id,
        message.content,
        message.response_id,
        json.dumps(message.metadata),
        None if message.parts is None else json.dumps(message.parts),
        retained_message_ids,
    )
    return None if row is None else message_from_row(row)


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


async def mark_active_stream_streaming_if_pending(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
    active_stream_id: UUID,
) -> ChatConversation | None:
    row = await db.fetchrow(
        """
        UPDATE chat_conversation
        SET active_status = 'streaming',
            updated_at = NOW()
        WHERE id = $1
          AND user_id = $2
          AND active_stream_id = $3
          AND active_status = 'pending'
        RETURNING *
        """,
        conversation_id,
        user_id,
        active_stream_id,
    )
    return None if row is None else conversation_from_row(row)
