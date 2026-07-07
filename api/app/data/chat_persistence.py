import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.data.chat_rows import conversation_from_row, message_from_row
from app.schemas.chat_persistence import (
    ActiveStreamStatus,
    ChatConversation,
    ChatConversationCreate,
    ChatConversationUpdate,
    ChatConversationWithMessages,
    ChatMessage,
    ChatMessageUpsert,
)


class ChatPersistenceDatabase(Protocol):
    async def fetch(
        self,
        query: str,
        *args: object,
    ) -> Sequence[Mapping[str, object]]:
        raise NotImplementedError

    async def fetchrow(
        self,
        query: str,
        *args: object,
    ) -> Mapping[str, object] | None:
        raise NotImplementedError

    async def fetchval(self, query: str, *args: object, column: int = 0) -> object:
        raise NotImplementedError


async def create_conversation(
    db: ChatPersistenceDatabase,
    conversation: ChatConversationCreate,
) -> ChatConversation:
    row = await db.fetchrow(
        """
        INSERT INTO chat_conversation (id, user_id, model, title)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (id) DO UPDATE SET
            user_id = EXCLUDED.user_id,
            model = COALESCE(EXCLUDED.model, chat_conversation.model),
            title = COALESCE(EXCLUDED.title, chat_conversation.title),
            updated_at = NOW()
        RETURNING *
        """,
        conversation.id,
        conversation.user_id,
        conversation.model,
        conversation.title,
    )
    if row is None:
        raise RuntimeError("chat_conversation upsert returned no row")
    return conversation_from_row(row)


async def update_conversation(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
    update: ChatConversationUpdate,
) -> ChatConversation | None:
    row = await db.fetchrow(
        """
        UPDATE chat_conversation
        SET model = COALESCE($3, model),
            title = COALESCE($4, title),
            active_status = COALESCE($5, active_status),
            updated_at = NOW()
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        conversation_id,
        user_id,
        update.model,
        update.title,
        update.active_status,
    )
    return None if row is None else conversation_from_row(row)


async def list_conversations(
    db: ChatPersistenceDatabase,
    *,
    user_id: UUID,
    limit: int = 50,
) -> list[ChatConversation]:
    rows = await db.fetch(
        """
        SELECT *
        FROM chat_conversation
        WHERE user_id = $1
        ORDER BY updated_at DESC
        LIMIT $2
        """,
        user_id,
        limit,
    )
    return [conversation_from_row(row) for row in rows]


async def load_conversation(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
) -> ChatConversationWithMessages | None:
    conversation_row = await db.fetchrow(
        """
        SELECT * FROM chat_conversation
        WHERE id = $1 AND user_id = $2
        """,
        conversation_id,
        user_id,
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
        conversation_id,
    )
    return ChatConversationWithMessages(
        conversation=conversation_from_row(conversation_row),
        messages=[message_from_row(row) for row in message_rows],
    )


async def get_conversation_owner(
    db: ChatPersistenceDatabase,
    conversation_id: UUID,
) -> UUID | None:
    row = await db.fetchrow(
        """
        SELECT user_id FROM chat_conversation
        WHERE id = $1
        """,
        conversation_id,
    )
    if row is None:
        return None
    owner = row["user_id"]
    if isinstance(owner, UUID):
        return owner
    if isinstance(owner, str):
        return UUID(owner)
    raise RuntimeError("chat_conversation user_id was not a UUID")


async def upsert_message(
    db: ChatPersistenceDatabase,
    message: ChatMessageUpsert,
) -> ChatMessage:
    row = await db.fetchrow(
        """
        INSERT INTO chat_message (id, conversation_id, role, content, response_id, metadata)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (id) DO UPDATE SET
            role = EXCLUDED.role,
            content = EXCLUDED.content,
            response_id = EXCLUDED.response_id,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        RETURNING *
        """,
        message.id,
        message.conversation_id,
        message.role.value,
        message.content,
        message.response_id,
        json.dumps(message.metadata),
    )
    if row is None:
        raise RuntimeError("chat_message upsert returned no row")
    return message_from_row(row)


async def set_active_stream(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
    active_stream_id: UUID,
    active_response_id: UUID,
    active_status: ActiveStreamStatus,
) -> ChatConversation | None:
    row = await db.fetchrow(
        """
        UPDATE chat_conversation
        SET active_stream_id = $3,
            active_response_id = $4,
            active_status = $5,
            updated_at = NOW()
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        conversation_id,
        user_id,
        active_stream_id,
        active_response_id,
        active_status.value,
    )
    return None if row is None else conversation_from_row(row)


async def clear_active_stream_if_current(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
    active_stream_id: UUID,
) -> ChatConversation | None:
    row = await db.fetchrow(
        """
        UPDATE chat_conversation
        SET active_stream_id = NULL,
            active_response_id = NULL,
            active_status = NULL,
            updated_at = NOW()
        WHERE id = $1 AND user_id = $2 AND active_stream_id = $3
        RETURNING *
        """,
        conversation_id,
        user_id,
        active_stream_id,
    )
    return None if row is None else conversation_from_row(row)


async def clear_stale_active_streams(
    db: ChatPersistenceDatabase,
    *,
    stale_before: datetime,
) -> int:
    count = await db.fetchval(
        """
        WITH cleared AS (
            UPDATE chat_conversation
            SET active_stream_id = NULL,
                active_response_id = NULL,
                active_status = NULL
            WHERE active_stream_id IS NOT NULL AND updated_at < $1
            RETURNING id
        )
        SELECT COUNT(*) FROM cleared
        """,
        stale_before,
    )
    if isinstance(count, int):
        return count
    if isinstance(count, str) and count.isdecimal():
        return int(count)
    raise RuntimeError("stale active stream clear count was not numeric")
