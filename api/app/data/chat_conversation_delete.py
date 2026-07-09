from uuid import UUID

from app.data.chat_persistence import ChatPersistenceDatabase
from app.data.chat_rows import conversation_from_row
from app.schemas.chat_persistence import ChatConversation


async def delete_conversation(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
) -> ChatConversation | None:
    row = await db.fetchrow(
        """
        DELETE FROM chat_conversation
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        conversation_id,
        user_id,
    )
    return None if row is None else conversation_from_row(row)


async def delete_conversations(
    db: ChatPersistenceDatabase,
    *,
    user_id: UUID,
) -> list[ChatConversation]:
    rows = await db.fetch(
        """
        DELETE FROM chat_conversation
        WHERE user_id = $1
        RETURNING *
        """,
        user_id,
    )
    return [conversation_from_row(row) for row in rows]
