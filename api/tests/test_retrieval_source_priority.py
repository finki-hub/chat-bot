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


def test_reranker_order_is_preserved_across_source_types() -> None:
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
        chunk_first.key,
        faq_first.key,
        chunk_second.key,
        faq_second.key,
    ]


def test_top_k_limits_reranked_candidates_without_source_promotion() -> None:
    # Given
    chunk = _make_chunk_candidate("Документ", 0)
    faqs = [_make_faq_candidate(f"FAQ {index}") for index in range(4)]

    # When
    selected = _select_with_source_priority([chunk, *faqs], top_k=4)

    # Then
    assert [candidate.key for candidate in selected] == [
        chunk.key,
        *[faq.key for faq in faqs[:3]],
    ]


def test_top_k_and_order_are_invariant_for_normal_faq_results() -> None:
    faqs = [_make_faq_candidate(f"FAQ {index}") for index in range(3)]

    selected = _select_with_source_priority(faqs, top_k=2)

    assert [candidate.key for candidate in selected] == [faq.key for faq in faqs[:2]]
