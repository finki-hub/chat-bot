from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.data.connection import Database
from app.data.discord_conversations import DiscordExchange, store_discord_exchange


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case", ["missing", "old_epoch", "tombstone", "owner", "duplicate", "stored"]
)
async def test_storage_locks_account_and_fails_closed_before_any_message_insert(case):
    exchange = DiscordExchange(
        uuid4(), uuid4(), uuid4(), "model", "question", "answer", {}, 3
    )
    generation = None if case == "missing" else 4 if case == "old_epoch" else 3
    conn = SimpleNamespace(
        execute=AsyncMock(return_value="SET"),
        fetchval=AsyncMock(
            side_effect=[
                generation,
                1 if case == "tombstone" else None,
                1 if case == "duplicate" else None,
            ]
        ),
        fetchrow=AsyncMock(
            return_value=None if case == "owner" else {"id": exchange.conversation_id}
        ),
        executemany=AsyncMock(),
    )

    @asynccontextmanager
    async def transaction():
        yield conn

    db = SimpleNamespace(transaction=transaction)
    assert await store_discord_exchange(cast("Database", db), exchange) is (
        case == "stored"
    )
    assert "FOR NO KEY UPDATE" in conn.fetchval.call_args_list[0].args[0]
    if case in {"missing", "old_epoch", "tombstone"}:
        conn.fetchrow.assert_not_awaited()
    if case == "stored":
        conn.executemany.assert_awaited_once()
        records = conn.executemany.call_args.args[1]
        assert len(records) == 2
        assert records[0][2:5] == ("user", "question", None)
        assert records[1][2:5] == ("assistant", "answer", exchange.response_id)
    else:
        conn.executemany.assert_not_awaited()
