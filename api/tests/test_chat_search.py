from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.data.chat_persistence import search_conversations
from app.schemas.chat_persistence import ChatMessageRole

USER_ID = UUID("00000000-0000-4000-8000-000000000101")


class SearchDatabase:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = rows
        self.query = ""
        self.args: tuple[object, ...] = ()
        self.calls = 0

    async def fetch(
        self,
        query: str,
        *args: object,
    ) -> Sequence[Mapping[str, object]]:
        self.calls += 1
        self.query = query
        self.args = args
        return self.rows

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


@pytest.mark.anyio
async def test_search_conversations_maps_messages_and_scopes_the_query() -> None:
    conversation_id = uuid4()
    message_id = uuid4()
    created_at = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    db = SearchDatabase(
        [
            {
                "conversation_id": conversation_id,
                "conversation_title": "Испити",
                "conversation_updated_at": created_at,
                "message_id": message_id,
                "message_role": "assistant",
                "message_created_at": created_at,
                "snippet": "\x01испити\x02 се објавени",
            },
        ],
    )

    results = await search_conversations(
        db,
        user_id=USER_ID,
        query="испити",
        limit=10,
        offset=5,
    )

    assert db.calls == 1
    assert "to_tsvector('simple', m.content)" in db.query
    assert "conversation_id DESC" in db.query
    assert db.args[0] == USER_ID
    assert db.args[1] == "испити"
    assert "StartSel=" in str(db.args[2])
    assert db.args[3:] == (10, 5)
    assert len(results) == 1
    assert results[0].conversation_id == conversation_id
    assert results[0].conversation_title == "Испити"
    assert results[0].message is not None
    assert results[0].message.id == message_id
    assert results[0].message.role is ChatMessageRole.ASSISTANT
    assert results[0].snippet == "\x01испити\x02 се објавени"


@pytest.mark.anyio
async def test_search_conversations_supports_title_only_matches() -> None:
    db = SearchDatabase(
        [
            {
                "conversation_id": uuid4(),
                "conversation_title": "Рокови",
                "conversation_updated_at": datetime.now(UTC),
                "message_id": None,
                "message_role": None,
                "message_created_at": None,
                "snippet": None,
            },
        ],
    )

    results = await search_conversations(
        db,
        user_id=USER_ID,
        query="рокови",
    )

    assert results[0].message is None
    assert results[0].snippet is None


@pytest.mark.anyio
async def test_search_conversations_skips_blank_queries() -> None:
    db = SearchDatabase([])

    results = await search_conversations(
        db,
        user_id=USER_ID,
        query="  ",
    )

    assert results == []
    assert db.calls == 0
