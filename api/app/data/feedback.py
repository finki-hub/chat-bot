from typing import Literal
from uuid import UUID

from app.data.connection import Database
from app.schemas.feedback import FeedbackAckSchema, FeedbackSchema


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


async def server_owned_web_feedback(
    db: Database,
    feedback: FeedbackSchema,
) -> FeedbackSchema | None:
    row = await db.fetchrow(
        """
        SELECT
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
        """,
        feedback.response_id,
        feedback.user_id,
    )
    if row is None:
        return None

    return feedback.model_copy(
        update={
            "answer_text": _optional_text(row["answer_text"]),
            "inference_model": _optional_text(row["inference_model"]),
            "question_text": _optional_text(row["question_text"]),
        },
    )


async def upsert_feedback(
    db: Database,
    feedback: FeedbackSchema,
) -> FeedbackAckSchema | None:
    query = """
    INSERT INTO feedback (
        response_id, client, user_id, feedback_type, client_ref,
        channel_id, guild_id, question_text, answer_text,
        inference_model, embeddings_model, query_transform_model,
        dislike_reason_category, dislike_reason_detail
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
    ON CONFLICT (response_id, client, user_id)
    DO UPDATE SET
        feedback_type = EXCLUDED.feedback_type,
        dislike_reason_category = EXCLUDED.dislike_reason_category,
        dislike_reason_detail = EXCLUDED.dislike_reason_detail,
        updated_at = NOW()
    RETURNING id, response_id, feedback_type
    """
    result = await db.fetchrow(
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
        feedback.dislike_reason_category,
        feedback.dislike_reason_detail,
    )

    if not result:
        return None

    return FeedbackAckSchema(
        id=result["id"],
        response_id=result["response_id"],
        feedback_type=result["feedback_type"],
    )


async def set_web_chat_message_feedback(
    db: Database,
    *,
    feedback_type: Literal["like", "dislike"],
    response_id: UUID,
    user_id: str,
) -> bool:
    row = await db.fetchrow(
        """
        UPDATE chat_message AS assistant
        SET metadata = jsonb_set(
                COALESCE(assistant.metadata, '{}'::jsonb),
                '{feedback}',
                to_jsonb($3::text),
                true
            ),
            updated_at = NOW()
        FROM chat_conversation AS conversation
        WHERE conversation.id = assistant.conversation_id
          AND assistant.response_id = $1
          AND assistant.role = 'assistant'
          AND conversation.user_id::text = $2
        RETURNING assistant.id
        """,
        response_id,
        user_id,
        feedback_type,
    )
    return row is not None
