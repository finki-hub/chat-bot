from collections.abc import Mapping

from app.data.chat_persistence import ChatPersistenceDatabase
from app.schemas.chat_user import ChatUser, ChatUserUpsert

_GOOGLE_PROVIDER = "google"


def _chat_user_from_row(row: Mapping[str, object]) -> ChatUser:
    return ChatUser.model_validate(dict(row))


async def upsert_google_chat_user(
    db: ChatPersistenceDatabase,
    user: ChatUserUpsert,
) -> ChatUser:
    row = await db.fetchrow(
        """
        INSERT INTO chat_user (provider, provider_subject, email, name, avatar_url)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (provider, provider_subject) DO UPDATE SET
            email = EXCLUDED.email,
            name = EXCLUDED.name,
            avatar_url = EXCLUDED.avatar_url,
            updated_at = NOW()
        RETURNING *
        """,
        _GOOGLE_PROVIDER,
        user.provider_subject,
        user.email,
        user.name,
        user.avatar_url,
    )
    if row is None:
        raise RuntimeError("chat_user upsert returned no row")
    return _chat_user_from_row(row)
