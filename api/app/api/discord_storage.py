import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from app.data.connection import Database
from app.data.discord_conversations import DiscordExchange, store_discord_exchange
from app.llms.retrieval_result import RetrievalSourcePayload
from app.schemas.chat import ChatSchema

logger = logging.getLogger(__name__)


class DiscordWriteRegistry:
    """Application-owned best-effort writes; stop admission before bounded drain."""

    def __init__(self) -> None:
        self.pending: set[asyncio.Task[None]] = set()
        self.accepting = True

    def schedule(self, storage: DiscordStorage, exchange: DiscordExchange) -> None:
        if not self.accepting:
            return
        task = asyncio.create_task(_store(storage, exchange))
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def drain(self, *, grace_seconds: float = 5.0) -> None:
        self.accepting = False
        if not self.pending:
            return
        tasks = tuple(self.pending)
        try:
            await asyncio.wait(tasks, timeout=grace_seconds)
        finally:
            # Also finish rollback/connection release if shutdown itself is cancelled.
            pending = {task for task in tasks if not task.done()}
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.pending.difference_update(tasks)


@dataclass(frozen=True, slots=True)
class DiscordStorage:
    db: Database
    conversation_id: UUID
    user_id: UUID
    sources: list[RetrievalSourcePayload]
    history_generation: int
    registry: DiscordWriteRegistry


async def discord_storage(
    payload: ChatSchema,
    db: Database,
    conversation_id: str | None,
    registry: DiscordWriteRegistry | None,
) -> DiscordStorage | None:
    """Capture an immutable epoch before any credentials/retrieval await. Fail closed."""
    if registry is None or (
        payload.interface != "discord"
        or payload.user_id is None
        or conversation_id is None
    ):
        return None
    try:
        generation = await db.fetchval(
            "SELECT history_generation FROM chat_user WHERE id = $1",
            payload.user_id,
        )
    # ruff: ignore[BLE001] -- losing persistence must not affect chat availability
    except Exception as exc:
        logger.log(
            logging.ERROR,
            "Discord storage capture failed error_type=%s",
            type(exc).__name__,
        )
        return None
    if not isinstance(generation, int):
        return None
    return DiscordStorage(
        db, UUID(conversation_id), payload.user_id, [], generation, registry
    )


def discord_context_metadata(payload: ChatSchema) -> dict[str, str | None]:
    context = payload.discord_context
    if context is None:
        return {}
    return {
        "guildId": context.guild_id,
        "channelId": context.channel_id,
        "messageId": context.message_id,
    }


async def _store(storage: DiscordStorage, exchange: DiscordExchange) -> None:
    try:
        stored = await store_discord_exchange(storage.db, exchange)
    # ruff: ignore[BLE001] -- storage must never affect the answered request
    except Exception as exc:
        logger.log(
            logging.ERROR,
            "Discord conversation storage failed response_id=%s error_type=%s",
            exchange.response_id,
            type(exc).__name__,
        )
        return
    if not stored:
        # Epoch/tombstone suppression and idempotency are normal, not backend failures.
        logger.debug("Discord exchange suppressed response_id=%s", exchange.response_id)


def schedule_discord_exchange(
    storage: DiscordStorage, exchange: DiscordExchange
) -> None:
    storage.registry.schedule(storage, exchange)
