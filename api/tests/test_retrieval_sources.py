import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import anyio
import pytest
from pydantic import HttpUrl

from app.data.connection import Database
from app.llms.agents import sources_event
from app.llms.context import (
    _chunk_candidate,
    _expand_and_render,
    _question_candidate,
)
from app.llms.models import Model
from app.llms.retrieval_result import visible_sources
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema


class WindowState:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def fetch(self, query: str, *_args: object) -> list[dict[str, object]]:
        return self.rows[:1] if "embedding_bge_m3_version" in query else self.rows


def _chunk(
    *,
    document_id: UUID,
    chunk_index: int,
    content: str,
) -> ChunkSchema:
    return ChunkSchema(
        id=uuid4(),
        document_id=document_id,
        document_name="document",
        document_title="Document",
        chunk_index=chunk_index,
        section=None,
        content=content,
    )


def _database(state: WindowState, monkeypatch) -> Database:
    database = Database("postgresql://context-test")
    monkeypatch.setattr(database, "fetch", AsyncMock(side_effect=state.fetch))
    return database


@pytest.mark.parametrize("model", [Model.BGE_M3, Model.BGE_M3_LOCAL])
def test_bge_context_expansion_excludes_dirty_neighbor(
    model: Model,
    monkeypatch,
) -> None:
    document_id = uuid4()
    center = _chunk(document_id=document_id, chunk_index=0, content="CURRENT CENTER")
    dirty_neighbor = "DIRTY NEIGHBOR AFTER EDIT"
    state = WindowState(
        [
            {
                "id": center.id,
                "document_id": document_id,
                "chunk_index": 0,
                "content": center.content,
                "section": None,
                "document_name": "document",
                "document_title": "Document",
            },
            {
                "id": uuid4(),
                "document_id": document_id,
                "chunk_index": 1,
                "content": dirty_neighbor,
                "section": None,
                "document_name": "document",
                "document_title": "Document",
            },
        ],
    )
    database = _database(state, monkeypatch)

    async def run() -> None:
        text = await _expand_and_render(
            database,
            [_chunk_candidate(center)],
            model,
        )

        assert center.content in text
        assert dirty_neighbor not in text

    anyio.run(run)


def test_non_bge_context_expansion_keeps_existing_neighbor_behavior(
    monkeypatch,
) -> None:
    document_id = uuid4()
    center = _chunk(document_id=document_id, chunk_index=0, content="CURRENT CENTER")
    neighbor = "NON-BGE NEIGHBOR"
    state = WindowState(
        [
            {
                "id": center.id,
                "document_id": document_id,
                "chunk_index": 0,
                "content": center.content,
                "section": None,
                "document_name": "document",
                "document_title": "Document",
            },
            {
                "id": uuid4(),
                "document_id": document_id,
                "chunk_index": 1,
                "content": neighbor,
                "section": None,
                "document_name": "document",
                "document_title": "Document",
            },
        ],
    )
    database = _database(state, monkeypatch)

    async def run() -> None:
        text = await _expand_and_render(
            database,
            [_chunk_candidate(center)],
            Model.TEXT_EMBEDDING_3_LARGE,
        )

        assert center.content in text
        assert neighbor in text

    anyio.run(run)


def test_question_candidate_keeps_structured_links():
    q = QuestionSchema(
        id=uuid4(),
        name="Упис",
        content="Уписот се прави преку iKnow.",
        links={"iKnow": HttpUrl(url="https://iknow.ukim.mk/")},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    cand = _question_candidate(q)

    assert cand.retrieval_source.as_payload() == {
        "id": str(q.id),
        "kind": "faq",
        "links": [{"label": "iKnow", "url": "https://iknow.ukim.mk/"}],
        "snippet": "Уписот се прави преку iKnow.",
        "title": "Упис",
    }


def test_chunk_candidate_keeps_document_section_and_index():
    chunk_id = uuid4()
    doc_id = uuid4()
    c = ChunkSchema(
        id=chunk_id,
        document_id=doc_id,
        document_name="statut-finki",
        document_title="Статут на ФИНКИ",
        chunk_index=4,
        section="Член 12",
        content="Правилата се наведени во членот.",
    )

    cand = _chunk_candidate(c)

    assert cand.retrieval_source.as_payload() == {
        "chunk_index": 4,
        "id": str(chunk_id),
        "kind": "chunk",
        "section": "Член 12",
        "snippet": "Правилата се наведени во членот.",
        "title": "Статут на ФИНКИ",
    }


def test_sources_event_serializes_sources_frame():
    frame = sources_event(
        [
            {
                "id": "c1",
                "kind": "chunk",
                "section": "Член 12",
                "title": "Статут на ФИНКИ",
            },
        ],
    )

    assert frame.startswith("event: sources\n")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {
        "sources": [
            {
                "id": "c1",
                "kind": "chunk",
                "section": "Член 12",
                "title": "Статут на ФИНКИ",
            },
        ],
    }


def test_visible_sources_uses_stricter_source_score_gate():
    q = QuestionSchema(
        id=uuid4(),
        name="Упис",
        content="Уписот се прави преку iKnow.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    weak = QuestionSchema(
        id=uuid4(),
        name="Нерелевантно",
        content="Ова е слаб кандидат.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    visible = visible_sources(
        [
            (_question_candidate(q).retrieval_source, 0.45),
            (_question_candidate(weak).retrieval_source, 0.29),
        ],
        source_score_floor=0.3,
    )

    assert [source.title for source in visible] == ["Упис"]


def test_visible_sources_are_empty_for_unscored_vector_fallback():
    q = QuestionSchema(
        id=uuid4(),
        name="Упис",
        content="Уписот се прави преку iKnow.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    assert (
        visible_sources(
            [(_question_candidate(q).retrieval_source, None)],
            source_score_floor=0.3,
        )
        == ()
    )
