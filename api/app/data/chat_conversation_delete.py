from uuid import UUID

from asyncpg import Record
from asyncpg.pool import PoolConnectionProxy

from app.data.chat_rows import conversation_from_row
from app.data.connection import Database
from app.schemas.chat_persistence import ChatConversation


async def delete_conversation(
    db: Database,
    *,
    conversation_id: UUID,
    user_id: UUID,
) -> ChatConversation | None:
    rows = await _delete_conversations(
        db, user_id=user_id, conversation_id=conversation_id
    )
    return rows[0] if rows else None


async def _delete_conversations(
    db: Database,
    *,
    user_id: UUID,
    conversation_id: UUID | None = None,
) -> list[ChatConversation]:
    async with db.transaction() as connection:
        await connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        # Lock parents first to prevent new messages and serialize state mutations.
        rows = await connection.fetch(
            """
            SELECT id FROM chat_conversation
            WHERE user_id = $1 AND ($2::uuid IS NULL OR id = $2)
            ORDER BY id
            FOR UPDATE
            """,
            user_id,
            conversation_id,
        )
        conversation_ids = [row["id"] for row in rows]
        if not conversation_ids:
            return []
        await _delete_feedback(connection, conversation_ids, user_id)
        deleted = await connection.fetch(
            """
            DELETE FROM chat_conversation
            WHERE id = ANY($1::uuid[]) AND user_id = $2
            RETURNING *
            """,
            conversation_ids,
            user_id,
        )
        return [conversation_from_row(dict(row)) for row in deleted]


async def _delete_feedback(
    connection: PoolConnectionProxy[Record],
    conversation_ids: list[UUID],
    user_id: UUID,
) -> None:
    # Web feedback upsert/retraction locks these same rows. Wait before taking
    # the DELETE snapshot: a single CTE could miss feedback committed while
    # waiting for the message lock under READ COMMITTED.
    await connection.fetch(
        """
        SELECT id FROM chat_message
        WHERE conversation_id = ANY($1::uuid[])
        ORDER BY id
        FOR UPDATE
        """,
        conversation_ids,
    )
    await connection.execute(
        """
        DELETE FROM feedback AS stored
        USING chat_message AS message
        WHERE message.conversation_id = ANY($1::uuid[])
          AND stored.response_id = message.response_id
          AND stored.client = 'web'
          AND stored.user_id = $2
        """,
        conversation_ids,
        str(user_id),
    )


async def delete_conversations(
    db: Database,
    *,
    user_id: UUID,
) -> list[ChatConversation]:
    return await _delete_conversations(db, user_id=user_id)
