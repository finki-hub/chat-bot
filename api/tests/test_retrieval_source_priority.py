from datetime import UTC, datetime
from uuid import uuid4

from app.llms.context import (
    _Candidate,
    _chunk_candidate,
    _question_candidate,
    _select_with_source_priority,
)
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema


def _make_faq_candidate(name: str) -> _Candidate:
    now = datetime.now(UTC)
    return _question_candidate(
        QuestionSchema(
            id=uuid4(),
            name=name,
            content=f"Одговор за {name}.",
            links={},
            created_at=now,
            updated_at=now,
        ),
    )


def _make_chunk_candidate(title: str, chunk_index: int) -> _Candidate:
    return _chunk_candidate(
        ChunkSchema(
            id=uuid4(),
            document_id=uuid4(),
            document_name=title.lower(),
            document_title=title,
            chunk_index=chunk_index,
            content=f"Содржина од {title}.",
        ),
    )


def test_faq_priority_precedes_documents_and_preserves_source_order() -> None:
    # Given
    chunk_first = _make_chunk_candidate("Документ 1", 0)
    faq_first = _make_faq_candidate("FAQ 1")
    chunk_second = _make_chunk_candidate("Документ 2", 1)
    faq_second = _make_faq_candidate("FAQ 2")

    # When
    selected = _select_with_source_priority(
        [chunk_first, faq_first, chunk_second, faq_second],
        top_k=4,
    )

    # Then
    assert [candidate.key for candidate in selected] == [
        faq_first.key,
        faq_second.key,
        chunk_first.key,
        chunk_second.key,
    ]


def test_faq_priority_is_applied_before_top_k_without_a_slot_cap() -> None:
    # Given
    chunk = _make_chunk_candidate("Документ", 0)
    faqs = [_make_faq_candidate(f"FAQ {index}") for index in range(4)]

    # When
    selected = _select_with_source_priority([chunk, *faqs], top_k=4)

    # Then
    assert [candidate.key for candidate in selected] == [
        candidate.key for candidate in faqs
    ]
