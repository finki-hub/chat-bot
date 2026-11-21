import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final, Protocol, cast
from uuid import UUID

from app.data.chat_rows import conversation_from_row, message_from_row
from app.schemas.chat_persistence import (
    ActiveStreamStatus,
    ChatConversation,
    ChatConversationCreate,
    ChatConversationUpdate,
    ChatConversationWithMessages,
    ChatMessage,
    ChatMessageRole,
    ChatMessageUpsert,
    ChatSearchMessage,
    ChatSearchResult,
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

    async def execute(self, query: str, *args: object) -> str:
        raise NotImplementedError


class ChatMessageConflictError(Exception):
    pass


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


_SEARCH_HEADLINE_OPTIONS: Final = (
    "StartSel=\x01, StopSel=\x02, MaxWords=28, MinWords=12, MaxFragments=2"
)


async def search_conversations(
    db: ChatPersistenceDatabase,
    *,
    user_id: UUID,
    query: str,
    limit: int = 20,
    offset: int = 0,
) -> list[ChatSearchResult]:
    """Search one user's conversations and messages with safe lexical matching."""
    if not query.strip():
        return []

    rows = await db.fetch(
        """
        -- Lexemes are reduced to letters and digits so they are safe inside
        -- to_tsquery; prefix matching covers Macedonian inflection.
        WITH lexemes AS (
            SELECT lexeme
            FROM unnest(tsvector_to_array(to_tsvector('simple', $2))) AS lexeme
            WHERE lexeme ~ '^[[:alnum:]]+$'
        ), terms AS (
            SELECT
                lexeme || CASE WHEN char_length(lexeme) >= 3 THEN ':*' ELSE '' END
                    AS term
            FROM lexemes
            WHERE char_length(lexeme) >= 2
               OR NOT EXISTS (
                   SELECT 1 FROM lexemes WHERE char_length(lexeme) >= 2
               )
        ), candidate_queries AS (
            SELECT
                to_tsquery('simple', string_agg(term, ' & ')) AS all_terms,
                to_tsquery('simple', string_agg(term, ' | ')) AS any_terms
            FROM terms
            HAVING count(*) > 0
        ), lexical_query AS (
            -- Require every word, and fall back to any word only when nothing
            -- matches them all, so pagination stays on one query.
            SELECT
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM chat_conversation c
                        WHERE c.user_id = $1
                          AND (
                              to_tsvector('simple', coalesce(c.title, ''))
                                  @@ cq.all_terms
                              OR EXISTS (
                                  SELECT 1
                                  FROM chat_message m
                                  WHERE m.conversation_id = c.id
                                    AND to_tsvector('simple', m.content)
                                        @@ cq.all_terms
                              )
                          )
                    ) THEN cq.all_terms
                    ELSE cq.any_terms
                END AS value
            FROM candidate_queries cq
        ), ranked AS (
            SELECT
                c.id AS conversation_id,
                c.title AS conversation_title,
                c.updated_at AS conversation_updated_at,
                best.id AS message_id,
                best.role AS message_role,
                best.created_at AS message_created_at,
                best.content AS message_content,
                COALESCE(
                    best.message_rank,
                    ts_rank_cd(
                        setweight(
                            to_tsvector('simple', coalesce(c.title, '')),
                            'A'
                        ),
                        q.value,
                        32
                    )
                ) AS score
            FROM chat_conversation c
            CROSS JOIN lexical_query q
            LEFT JOIN LATERAL (
                SELECT
                    m.id,
                    m.role,
                    m.created_at,
                    m.content,
                    ts_rank_cd(
                        setweight(
                            to_tsvector('simple', coalesce(c.title, '')),
                            'A'
                        ) ||
                        setweight(to_tsvector('simple', m.content), 'B'),
                        q.value,
                        32
                    ) AS message_rank
                FROM chat_message m
                WHERE m.conversation_id = c.id
                  AND to_tsvector('simple', m.content) @@ q.value
                ORDER BY message_rank DESC, m.created_at ASC
                LIMIT 1
            ) best ON TRUE
            WHERE c.user_id = $1
              AND (
                  best.id IS NOT NULL
                  OR to_tsvector('simple', coalesce(c.title, '')) @@ q.value
              )
        ), page AS (
            SELECT *
            FROM ranked
            ORDER BY score DESC, conversation_updated_at DESC, conversation_id DESC
            LIMIT $4 OFFSET $5
        )
        -- Headlines are expensive, so build them only for the returned page.
        SELECT
            p.conversation_id,
            p.conversation_title,
            p.conversation_updated_at,
            p.message_id,
            p.message_role,
            p.message_created_at,
            CASE
                WHEN p.message_id IS NOT NULL THEN
                    ts_headline('simple', p.message_content, q.value, $3)
            END AS snippet
        FROM page p
        CROSS JOIN lexical_query q
        ORDER BY p.score DESC, p.conversation_updated_at DESC, p.conversation_id DESC
        """,
        user_id,
        query,
        _SEARCH_HEADLINE_OPTIONS,
        limit,
        offset,
    )

    results: list[ChatSearchResult] = []
    for row in rows:
        message: ChatSearchMessage | None = None
        if row["message_id"] is not None:
            message = ChatSearchMessage(
                id=cast(UUID, row["message_id"]),
                role=cast(ChatMessageRole, row["message_role"]),
                created_at=cast(datetime, row["message_created_at"]),
            )
        results.append(
            ChatSearchResult(
                conversation_id=cast(UUID, row["conversation_id"]),
                conversation_title=cast(str | None, row["conversation_title"]),
                conversation_updated_at=cast(
                    datetime,
                    row["conversation_updated_at"],
                ),
                message=message,
                snippet=cast(str | None, row["snippet"]),
            ),
        )
    return results


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
        INSERT INTO chat_message (id, conversation_id, role, content, response_id, metadata, parts)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
        ON CONFLICT (id) DO UPDATE SET
            role = EXCLUDED.role,
            content = EXCLUDED.content,
            response_id = EXCLUDED.response_id,
            metadata = EXCLUDED.metadata,
            parts = EXCLUDED.parts,
            updated_at = NOW()
        WHERE chat_message.conversation_id = EXCLUDED.conversation_id
          AND chat_message.role = EXCLUDED.role
          AND chat_message.response_id IS NOT DISTINCT FROM EXCLUDED.response_id
        RETURNING *
        """,
        message.id,
        message.conversation_id,
        message.role.value,
        message.content,
        message.response_id,
        json.dumps(message.metadata),
        None if message.parts is None else json.dumps(message.parts),
    )
    if row is None:
        raise ChatMessageConflictError
    return message_from_row(row)


async def set_active_stream(
    db: ChatPersistenceDatabase,
    *,
    conversation_id: UUID,
    user_id: UUID,
    active_stream_id: UUID,
    active_response_id: UUID,
    active_replacement_message_id: UUID | None,
    active_status: ActiveStreamStatus,
) -> ChatConversation | None:
    row = await db.fetchrow(
        """
        UPDATE chat_conversation
        SET active_stream_id = $3,
            active_response_id = $4,
            active_replacement_message_id = $5,
            active_status = $6,
            updated_at = NOW()
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        conversation_id,
        user_id,
        active_stream_id,
        active_response_id,
        active_replacement_message_id,
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
            active_replacement_message_id = NULL,
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
                active_replacement_message_id = NULL,
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
