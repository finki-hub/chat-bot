import json
from datetime import UTC, datetime
from uuid import uuid4

from app.llms.agents import sources_event
from app.llms.context import _chunk_candidate, _question_candidate
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema


def test_question_candidate_keeps_structured_links():
    q = QuestionSchema(
        id=uuid4(),
        name="Упис",
        content="Уписот се прави преку iKnow.",
        links={"iKnow": "https://iknow.ukim.mk/"},
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
