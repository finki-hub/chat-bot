import json
from uuid import UUID

from app.data.chat_persistence import ChatPersistenceDatabase
from app.data.chat_rows import conversation_from_row, message_from_row
from app.data.connection import Database
from app.schemas.chat_persistence import (
    ChatConversation,
    ChatMessage,
    ChatMessageUpsert,
)


async def upsert_assistant_message_by_response_id(
    db: ChatPersistenceDatabase,
    message: ChatMessageUpsert,
    *,
    active_stream_id: UUID,
    user_id: UUID,
) -> ChatMessage | None:
    row = await db.fetchrow(
        """
        INSERT INTO chat_message (id, conversation_id, role, content, response_id, metadata, parts)
        SELECT $3, conversation.id, 'assistant', $4, $2, $5::jsonb, $6::jsonb
        FROM chat_conversation conversation
        WHERE conversation.id = $1
          AND conversation.user_id = $7
          AND conversation.active_stream_id = $8
        FOR UPDATE OF conversation
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
        user_id,
        active_stream_id,
    )
    return None if row is None else message_from_row(row)


async def replace_assistant_message_and_prune_after(
    db: Database,
    message: ChatMessageUpsert,
    *,
    active_stream_id: UUID,
    retained_message_ids: list[UUID],
    user_id: UUID,
) -> ChatMessage | None:
    async with db.transaction() as connection:
        await connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        conversation = await connection.fetchrow(
            """
            SELECT id FROM chat_conversation
            WHERE id = $1 AND user_id = $2 AND active_stream_id = $3
            FOR UPDATE
            """,
            message.conversation_id,
            user_id,
            active_stream_id,
        )
        if conversation is None:
            return None
        # Match deletion's lock order: conversation, messages by id, feedback.
        rows = await connection.fetch(
            """
            SELECT * FROM chat_message
            WHERE conversation_id = $1
              AND (id = $2 OR NOT (id = ANY($3::uuid[])))
            ORDER BY id
            FOR UPDATE
            """,
            message.conversation_id,
            message.id,
            retained_message_ids,
        )
        target = next((row for row in rows if row["id"] == message.id), None)
        if (
            target is None
            or target["role"] != "assistant"
            or message.id not in retained_message_ids
        ):
            return None
        invalidated_response_ids = list(
            {
                row["response_id"]
                for row in rows
                if row["role"] == "assistant"
                and row["response_id"] is not None
                and (
                    row["id"] != message.id or row["response_id"] != message.response_id
                )
            }
        )
        if invalidated_response_ids:
            # A new statement snapshot sees feedback committed while the locks
            # above waited. Delete copies before losing their response-id links.
            await connection.execute(
                """
                DELETE FROM feedback
                WHERE client = 'web' AND user_id = $1
                  AND response_id = ANY($2::uuid[])
                """,
                str(user_id),
                invalidated_response_ids,
            )
        metadata = dict(message.metadata)
        metadata.pop("feedback", None)
        if target["response_id"] == message.response_id:
            # Replaying the same completion must not retract an intervening vote.
            stored_metadata = message_from_row(dict(target)).metadata
            if "feedback" in stored_metadata:
                metadata["feedback"] = stored_metadata["feedback"]
        updated = await connection.fetchrow(
            """
            UPDATE chat_message
            SET content = $3, response_id = $4, metadata = $5::jsonb,
                parts = $6::jsonb, updated_at = NOW()
            WHERE id = $1 AND conversation_id = $2
            RETURNING *
            """,
            message.id,
            message.conversation_id,
            message.content,
            message.response_id,
            json.dumps(metadata),
            None if message.parts is None else json.dumps(message.parts),
        )
        await connection.execute(
            """
            DELETE FROM chat_message
            WHERE conversation_id = $1 AND NOT (id = ANY($2::uuid[]))
            """,
            message.conversation_id,
            retained_message_ids,
        )
        return None if updated is None else message_from_row(dict(updated))


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
