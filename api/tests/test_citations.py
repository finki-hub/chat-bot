from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from app.llms.context import (
    _assign_citations,
    _chunk_candidate,
    _question_candidate,
    _render_blocks,
)
from app.llms.retrieval_result import CitationRefs, citation_key
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema


def _faq(name: str) -> QuestionSchema:
    return QuestionSchema(
        id=uuid4(),
        name=name,
        content=f"{name} одговор.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _chunk(title: str, index: int) -> ChunkSchema:
    return ChunkSchema(
        id=uuid4(),
        document_id=uuid4(),
        document_name="document",
        document_title=title,
        chunk_index=index,
        section=None,
        content=f"{title} дел {index}.",
    )


def test_chunks_of_one_document_share_a_citation_key() -> None:
    first = _chunk_candidate(_chunk("Правилник", 1)).retrieval_source
    second = _chunk_candidate(_chunk("Правилник", 7)).retrieval_source
    faq = _question_candidate(_faq("Упис")).retrieval_source

    assert citation_key(first) == citation_key(second)
    assert citation_key(faq) != citation_key(first)


def test_numbers_stay_stable_across_searches_of_one_answer() -> None:
    refs = CitationRefs()
    faq = _question_candidate(_faq("Упис")).retrieval_source
    doc = _chunk_candidate(_chunk("Правилник", 1)).retrieval_source

    assert refs.assign(faq) == 1
    assert refs.assign(doc) == 2
    # A later search returning the same FAQ first does not renumber it.
    assert refs.assign(faq) == 1
    assert refs.get(doc) == 2


def test_only_shown_sources_are_numbered_in_rank_order() -> None:
    shown = _question_candidate(_faq("Упис"))
    hidden = _question_candidate(_faq("Заверка"))
    doc = _chunk_candidate(_chunk("Правилник", 1))
    final = [shown, hidden, doc]
    refs = CitationRefs()

    cite = _assign_citations(
        final,
        (shown.retrieval_source, doc.retrieval_source),
        refs,
    )

    assert cite(shown.retrieval_source) == 1
    assert cite(hidden.retrieval_source) is None
    assert cite(doc.retrieval_source) == 2


def test_blocks_carry_their_citation_label() -> None:
    shown = _question_candidate(_faq("Упис"))
    hidden = _question_candidate(_faq("Заверка"))
    refs = CitationRefs()
    cite = _assign_citations([shown, hidden], (shown.retrieval_source,), refs)

    text = _render_blocks([shown, hidden], {}, cite)
    shown_block, hidden_block = text.split("\n\n---\n\n")

    assert shown_block.startswith("Ознака за цитат: [1]\n")
    assert "Ознака за цитат" not in hidden_block


def test_without_citations_blocks_are_unchanged() -> None:
    candidate = _question_candidate(_faq("Упис"))

    assert _render_blocks([candidate], {}) == candidate.context_text


def test_the_source_payload_carries_its_number() -> None:
    source = _question_candidate(_faq("Упис")).retrieval_source

    assert "ref" not in source.as_payload()
    assert replace(source, ref=3).as_payload().get("ref") == 3
