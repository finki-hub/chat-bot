import json
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID, uuid4

from app.data.chat_conversation_delete import delete_history_on_connection
from app.data.connection import Database

_TITLE_CHARS = 80


@dataclass(frozen=True, slots=True)
class DiscordExchange:
    conversation_id: UUID
    user_id: UUID
    response_id: UUID
    model: str
    question: str
    answer: str
    metadata: Mapping[str, object]
    history_generation: int


@dataclass(frozen=True, slots=True)
class DeletedChatHistory:
    conversations: int
    feedback: int


def _title(question: str) -> str:
    line = " ".join(question.split())
    return line if len(line) <= _TITLE_CHARS else line[: _TITLE_CHARS - 1] + "…"


async def store_discord_exchange(db: Database, exchange: DiscordExchange) -> bool:
    """Append one question and answer to the user's conversation.

    Returns False when the conversation id belongs to another user or the response
    is already stored.
    """
    async with db.transaction() as connection:
        await connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        generation = await connection.fetchval(
            "SELECT history_generation FROM chat_user WHERE id = $1 FOR NO KEY UPDATE",
            exchange.user_id,
        )
        if generation is None or generation != exchange.history_generation:
            return False
        if await connection.fetchval(
            "SELECT 1 FROM chat_conversation_tombstone WHERE id = $1",
            exchange.conversation_id,
        ):
            return False
        owned = await connection.fetchrow(
            """
            INSERT INTO chat_conversation (id, user_id, model, title)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO UPDATE SET
                model = EXCLUDED.model,
                updated_at = NOW()
            WHERE chat_conversation.user_id = EXCLUDED.user_id
            RETURNING id
            """,
            exchange.conversation_id,
            exchange.user_id,
            exchange.model,
            _title(exchange.question),
        )
        if owned is None:
            return False
        stored = await connection.fetchval(
            """
            SELECT 1 FROM chat_message
            WHERE conversation_id = $1 AND response_id = $2 AND role = 'assistant'
            """,
            exchange.conversation_id,
            exchange.response_id,
        )
        if stored is not None:
            return False
        # clock_timestamp() keeps the question ordered before its answer.
        await connection.executemany(
            """
            INSERT INTO chat_message (
                id, conversation_id, role, content, response_id, metadata, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, clock_timestamp(), clock_timestamp())
            """,
            [
                (
                    uuid4(),
                    exchange.conversation_id,
                    "user",
                    exchange.question,
                    None,
                    json.dumps({"interface": "discord"}),
                ),
                (
                    uuid4(),
                    exchange.conversation_id,
                    "assistant",
                    exchange.answer,
                    exchange.response_id,
                    json.dumps(dict(exchange.metadata)),
                ),
            ],
        )
    return True


async def delete_chat_history(db: Database, *, user_id: UUID) -> DeletedChatHistory:
    """Delete a user's conversations and feedback, including Discord feedback."""
    async with db.transaction() as connection:
        await connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        conversations, feedback = await delete_history_on_connection(
            connection,
            user_id=user_id,
            erase_authored_feedback=True,
        )
        return DeletedChatHistory(conversations=len(conversations), feedback=feedback)
