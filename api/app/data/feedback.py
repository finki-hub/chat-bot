from dataclasses import dataclass
from uuid import UUID

from asyncpg import Record
from asyncpg.pool import PoolConnectionProxy

from app.data.connection import Database
from app.schemas.feedback import FeedbackAckSchema, FeedbackSchema


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True, slots=True)
class StoredWebFeedback:
    ack: FeedbackAckSchema
    feedback: FeedbackSchema


class DiscordFeedbackUnavailableError(Exception):
    def __init__(self, generation: int | None) -> None:
        super().__init__("Discord response or rater unavailable")
        self.generation = generation


class DiscordFeedbackStaleError(Exception):
    """A deletion crossed the first attempt or an immutable retry token."""


async def upsert_web_feedback(
    db: Database,
    feedback: FeedbackSchema,
) -> StoredWebFeedback | None:
    row = await db.fetchrow(
        """
        WITH owned_response AS (
            SELECT
                assistant.id AS message_id,
                assistant.content AS answer_text,
                assistant.metadata ->> 'inferenceModel' AS inference_model,
                (
                    SELECT user_message.content
                    FROM chat_message user_message
                    WHERE user_message.conversation_id = assistant.conversation_id
                      AND user_message.role = 'user'
                      AND user_message.created_at <= assistant.created_at
                    ORDER BY user_message.created_at DESC
                    LIMIT 1
                ) AS question_text
            FROM chat_message assistant
            JOIN chat_conversation conversation
              ON conversation.id = assistant.conversation_id
            WHERE assistant.response_id = $1
              AND assistant.role = 'assistant'
              AND conversation.user_id::text = $2
            ORDER BY assistant.updated_at DESC
            LIMIT 1
            FOR UPDATE OF assistant
        ),
        updated_message AS (
            UPDATE chat_message AS assistant
            SET metadata = jsonb_set(
                    COALESCE(assistant.metadata, '{}'::jsonb),
                    '{feedback}',
                    to_jsonb($3::text),
                    true
                ),
                updated_at = NOW()
            FROM owned_response
            WHERE assistant.id = owned_response.message_id
            RETURNING assistant.id
        ),
        stored_feedback AS (
            INSERT INTO feedback (
                response_id, client, user_id, feedback_type, client_ref,
                channel_id, guild_id, question_text, answer_text,
                inference_model, embeddings_model, query_transform_model
            )
            SELECT
                $1, 'web', $2, $3, $4, $5, $6,
                owned_response.question_text,
                owned_response.answer_text,
                owned_response.inference_model,
                $7,
                $8
            FROM owned_response
            JOIN updated_message ON TRUE
            ON CONFLICT (response_id, client, user_id)
            DO UPDATE SET feedback_type = EXCLUDED.feedback_type, updated_at = NOW()
            RETURNING id, response_id, feedback_type
        )
        SELECT
            stored_feedback.id,
            stored_feedback.response_id,
            stored_feedback.feedback_type,
            assistant.content AS answer_text,
            owned_response.inference_model,
            owned_response.question_text
        FROM stored_feedback
        JOIN owned_response ON TRUE
        JOIN chat_message assistant
          ON assistant.id = owned_response.message_id
        """,
        feedback.response_id,
        feedback.user_id,
        feedback.feedback_type,
        feedback.client_ref,
        feedback.channel_id,
        feedback.guild_id,
        feedback.embeddings_model,
        feedback.query_transform_model,
    )
    if row is None:
        return None

    stored_feedback = feedback.model_copy(
        update={
            "answer_text": _optional_text(row["answer_text"]),
            "inference_model": _optional_text(row["inference_model"]),
            "question_text": _optional_text(row["question_text"]),
        },
    )
    return StoredWebFeedback(
        ack=FeedbackAckSchema(
            id=row["id"],
            response_id=row["response_id"],
            feedback_type=row["feedback_type"],
        ),
        feedback=stored_feedback,
    )


async def upsert_feedback(
    db: Database,
    feedback: FeedbackSchema,
) -> FeedbackAckSchema | None:
    try:
        stored = (
            await upsert_discord_feedback(db, feedback)
            if feedback.client == "discord"
            else await upsert_web_feedback(db, feedback)
        )
    except DiscordFeedbackUnavailableError, DiscordFeedbackStaleError:
        return None
    return stored.ack if stored else None


async def upsert_discord_feedback(
    db: Database,
    feedback: FeedbackSchema,
    *,
    generation: int | None = None,
) -> StoredWebFeedback:
    """Capture the rater epoch before lookup, including when storage is pending."""
    rater = await db.fetchrow(
        """
        SELECT id, history_generation FROM chat_user
        WHERE provider = 'discord' AND provider_subject = $1
        """,
        feedback.user_id,
    )
    if rater is None:
        raise DiscordFeedbackUnavailableError(None)
    expected_generation = (
        rater["history_generation"] if generation is None else generation
    )
    candidates = await db.fetch(
        """
        SELECT assistant.id, assistant.conversation_id, conversation.user_id AS owner_id,
               owner.history_generation AS owner_generation
        FROM chat_message AS assistant
        JOIN chat_conversation AS conversation ON conversation.id = assistant.conversation_id
        JOIN chat_user AS owner ON owner.id = conversation.user_id
        WHERE assistant.response_id = $1 AND assistant.role = 'assistant'
          AND assistant.metadata->>'interface' = 'discord'
        ORDER BY assistant.id LIMIT 2
        """,
        feedback.response_id,
    )
    candidate = candidates[0] if len(candidates) == 1 else None
    account_ids = {rater["id"]}
    if candidate is not None:
        account_ids.add(candidate["owner_id"])
    async with db.transaction() as connection:
        await connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        accounts = await connection.fetch(
            """
            SELECT id, history_generation FROM chat_user WHERE id = ANY($1::uuid[])
            ORDER BY id FOR NO KEY UPDATE
            """,
            sorted(account_ids),
        )
        generations = {row["id"]: row["history_generation"] for row in accounts}
        if generations.get(rater["id"]) != expected_generation:
            raise DiscordFeedbackStaleError
        if candidate is None:
            raise DiscordFeedbackUnavailableError(expected_generation)
        if generations.get(candidate["owner_id"]) != candidate["owner_generation"]:
            raise DiscordFeedbackStaleError
        parent = await connection.fetchrow(
            "SELECT id FROM chat_conversation WHERE id = $1 AND user_id = $2 FOR UPDATE",
            candidate["conversation_id"],
            candidate["owner_id"],
        )
        if parent is None:
            raise DiscordFeedbackUnavailableError(expected_generation)
        assistant = await connection.fetchrow(
            """
            SELECT content, metadata->>'inferenceModel' AS inference_model,
                   metadata->'diagnostics'->>'embeddingsModel' AS embeddings_model,
                   metadata->'diagnostics'->>'queryTransformModel' AS query_transform_model
            FROM chat_message WHERE id = $1 AND conversation_id = $2
              AND response_id = $3 AND role = 'assistant'
              AND metadata->>'interface' = 'discord'
            FOR UPDATE
            """,
            candidate["id"],
            candidate["conversation_id"],
            feedback.response_id,
        )
        if assistant is None:
            raise DiscordFeedbackUnavailableError(expected_generation)
        question = await connection.fetchval(
            """
            SELECT question.content FROM chat_message AS question
            JOIN chat_message AS answer ON answer.id = $1
            WHERE question.conversation_id = answer.conversation_id AND question.role = 'user'
              AND question.created_at <= answer.created_at
            ORDER BY question.created_at DESC, question.id DESC LIMIT 1
            """,
            candidate["id"],
        )
        verified = feedback.model_copy(
            update={
                "question_text": _optional_text(question),
                "answer_text": _optional_text(assistant["content"]),
                "inference_model": _optional_text(assistant["inference_model"]),
                "embeddings_model": _optional_text(assistant["embeddings_model"]),
                "query_transform_model": _optional_text(
                    assistant["query_transform_model"]
                ),
            }
        )
        ack = await _insert_discord_feedback(connection, verified)
        if ack is None:
            raise DiscordFeedbackUnavailableError(expected_generation)
        return StoredWebFeedback(ack=ack, feedback=verified)


async def _insert_discord_feedback(
    connection: PoolConnectionProxy[Record],
    feedback: FeedbackSchema,
) -> FeedbackAckSchema | None:
    query = """
    INSERT INTO feedback (
        response_id, client, user_id, feedback_type, client_ref,
        channel_id, guild_id, question_text, answer_text,
        inference_model, embeddings_model, query_transform_model
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ON CONFLICT (response_id, client, user_id)
    DO UPDATE SET feedback_type = EXCLUDED.feedback_type, updated_at = NOW()
    RETURNING id, response_id, feedback_type
    """
    result = await connection.fetchrow(
        query,
        feedback.response_id,
        feedback.client,
        feedback.user_id,
        feedback.feedback_type,
        feedback.client_ref,
        feedback.channel_id,
        feedback.guild_id,
        feedback.question_text,
        feedback.answer_text,
        feedback.inference_model,
        feedback.embeddings_model,
        feedback.query_transform_model,
    )

    if not result:
        return None

    return FeedbackAckSchema(
        id=result["id"],
        response_id=result["response_id"],
        feedback_type=result["feedback_type"],
    )


async def retract_web_feedback(
    db: Database,
    *,
    response_id: UUID,
    user_id: str,
) -> bool:
    row = await db.fetchrow(
        """
        WITH owned_response AS (
            SELECT assistant.id
            FROM chat_message AS assistant
            JOIN chat_conversation AS conversation
              ON conversation.id = assistant.conversation_id
            WHERE assistant.response_id = $1
              AND assistant.role = 'assistant'
              AND conversation.user_id::text = $2
            FOR UPDATE OF assistant
        ),
        deleted_feedback AS (
            DELETE FROM feedback AS stored
            USING owned_response
            WHERE stored.response_id = $1
              AND stored.client = 'web'
              AND stored.user_id = $2
            RETURNING stored.id
        )
        UPDATE chat_message AS assistant
        SET metadata = COALESCE(assistant.metadata, '{}'::jsonb) - 'feedback',
            updated_at = NOW()
        FROM owned_response
        LEFT JOIN deleted_feedback ON TRUE
        WHERE assistant.id = owned_response.id
        RETURNING assistant.id
        """,
        response_id,
        user_id,
    )
    return row is not None
