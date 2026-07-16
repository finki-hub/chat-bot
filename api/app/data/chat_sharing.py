from uuid import UUID

from app.data.chat_persistence import ChatPersistenceDatabase
from app.data.chat_rows import conversation_from_row, message_from_row
from app.schemas.chat_persistence import ChatConversationWithMessages


async def create_conversation_share(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
) -> UUID | None:
    row = await db.fetchrow(
        """
        UPDATE chat_conversation
        SET share_token = COALESCE(share_token, gen_random_uuid())
        WHERE id = $1 AND user_id = $2
        RETURNING share_token
        """,
        conversation_id,
        user_id,
    )
    if row is None:
        return None

    match row["share_token"]:
        case UUID() as share_token:
            return share_token
        case str() as share_token:
            return UUID(share_token)
        case invalid_share_token:
            message = f"chat_conversation share_token was invalid: {invalid_share_token!r}"
            raise RuntimeError(message)


async def load_shared_conversation(
    db: ChatPersistenceDatabase,
    *,
    share_token: UUID,
) -> ChatConversationWithMessages | None:
    conversation_row = await db.fetchrow(
        """
        SELECT * FROM chat_conversation
        WHERE share_token = $1
        """,
        share_token,
    )
    if conversation_row is None:
        return None

    message_rows = await db.fetch(
        """
        SELECT *
        FROM chat_message
        WHERE conversation_id = $1
        ORDER BY created_at ASC
        """,
        conversation_row["id"],
    )
    return ChatConversationWithMessages(
        conversation=conversation_from_row(conversation_row),
        messages=[message_from_row(row) for row in message_rows],
    )
