import base64
import hashlib
from collections.abc import Mapping, Sequence
from typing import Protocol
from uuid import UUID

from asyncpg import Record
from cryptography.fernet import Fernet

from app.schemas.chat_credentials import (
    ChatCredentialProvider,
    ChatCredentialPublic,
    ChatCredentialSecret,
    ChatCredentialUpsert,
)
from app.utils.settings import Settings

ChatCredentialRow = Mapping[str, object] | Record


class ChatCredentialDatabase(Protocol):
    async def fetch(
        self,
        query: str,
        *args: object,
    ) -> Sequence[ChatCredentialRow]:
        raise NotImplementedError

    async def fetchrow(
        self,
        query: str,
        *args: object,
    ) -> ChatCredentialRow | None:
        raise NotImplementedError

    async def execute(self, query: str, *args: object) -> str:
        raise NotImplementedError


def _fernet(settings: Settings) -> Fernet:
    digest = hashlib.sha256(settings.CREDENTIAL_ENCRYPTION_KEY.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _public_from_row(row: ChatCredentialRow) -> ChatCredentialPublic:
    data = dict(row)
    data["has_api_key"] = True
    return ChatCredentialPublic.model_validate(data)


async def list_chat_credentials(
    db: ChatCredentialDatabase,
    *,
    user_id: UUID,
) -> list[ChatCredentialPublic]:
    rows = await db.fetch(
        """
        SELECT user_id, provider, base_url, created_at, updated_at
        FROM chat_user_credential
        WHERE user_id = $1
        ORDER BY provider ASC
        """,
        user_id,
    )
    return [_public_from_row(row) for row in rows]


async def upsert_chat_credential(
    db: ChatCredentialDatabase,
    *,
    user_id: UUID,
    credential: ChatCredentialUpsert,
    settings: Settings,
) -> ChatCredentialPublic:
    encrypted_api_key = _fernet(settings).encrypt(credential.api_key.encode()).decode()
    row = await db.fetchrow(
        """
        INSERT INTO chat_user_credential (user_id, provider, encrypted_api_key, base_url)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, provider) DO UPDATE SET
            encrypted_api_key = EXCLUDED.encrypted_api_key,
            base_url = EXCLUDED.base_url,
            updated_at = NOW()
        RETURNING user_id, provider, base_url, created_at, updated_at
        """,
        user_id,
        credential.provider,
        encrypted_api_key,
        credential.base_url,
    )
    if row is None:
        raise RuntimeError("chat_user_credential upsert returned no row")
    return _public_from_row(row)


async def get_chat_credential_secret(
    db: ChatCredentialDatabase,
    *,
    user_id: UUID | None,
    provider: ChatCredentialProvider,
    settings: Settings,
) -> ChatCredentialSecret | None:
    if user_id is None:
        return None
    row = await db.fetchrow(
        """
        SELECT provider, encrypted_api_key, base_url
        FROM chat_user_credential
        WHERE user_id = $1 AND provider = $2
        """,
        user_id,
        provider,
    )
    if row is None:
        return None
    encrypted = row["encrypted_api_key"]
    if not isinstance(encrypted, str):
        msg = "encrypted API key must be stored as text"
        raise TypeError(msg)
    api_key = _fernet(settings).decrypt(encrypted.encode()).decode()
    return ChatCredentialSecret(
        provider=provider,
        api_key=api_key,
        base_url=row["base_url"] if isinstance(row["base_url"], str) else None,
    )


async def delete_chat_credential(
    db: ChatCredentialDatabase,
    *,
    user_id: UUID,
    provider: ChatCredentialProvider,
) -> None:
    await db.execute(
        """
        DELETE FROM chat_user_credential
        WHERE user_id = $1 AND provider = $2
        """,
        user_id,
        provider,
    )
