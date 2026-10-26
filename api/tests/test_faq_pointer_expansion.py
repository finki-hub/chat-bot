from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest
from asyncpg import PostgresError

from app.data.connection import Database
from app.data.questions import get_questions_by_names_query
from app.llms import context as context_module
from app.llms.context import (
    _Candidate,
    _chunk_candidate,
    _question_candidate,
    _with_pointed_faqs,
    get_retrieved_context_with_sources,
)
from app.llms.faq_evidence_constraints import FaqConstraints
from app.llms.models import Model
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from app.llms.retrieval_result import RetrievalSource
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema

_PROXY_TITLE = "Не можам да дојдам да ги донесам документите за упис. Може ли друго лице да ги донесе?"


def _faq(name: str, content: str, note: str | None = None) -> QuestionSchema:
    return QuestionSchema(
        id=uuid4(),
        name=name,
        content=content,
        links={},
        assistant_note=note,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _pointing_to(*titles: str) -> str:
    return " ".join(f"Види „{title}“." for title in titles)


def _database(
    monkeypatch: pytest.MonkeyPatch,
    questions: list[QuestionSchema],
) -> tuple[Database, AsyncMock]:
    lookup = AsyncMock(
        side_effect=lambda _db, names: [q for q in questions if q.name in names],
    )
    monkeypatch.setattr(context_module, "get_questions_by_names_query", lookup)
    return Database("postgresql://pointer-test"), lookup


def _expand(
    database: Database,
    final: list[_Candidate],
    *,
    sources: tuple[RetrievalSource, ...] | None = None,
    constraints: FaqConstraints | None = None,
    top_k: int = 10,
) -> tuple[list[_Candidate], tuple[RetrievalSource, ...]]:
    async def run() -> tuple[list[_Candidate], tuple[RetrievalSource, ...]]:
        return await _with_pointed_faqs(
            database,
            final,
            tuple(c.retrieval_source for c in final) if sources is None else sources,
            constraints,
            top_k=top_k,
        )

    return anyio.run(run)


def test_pointer_to_unselected_faq_is_added_after_the_pointing_faq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _faq(_PROXY_TITLE, "Да, со полномошно.", note="Провери го рокот.")
    big = _faq(
        "Упис на прв циклус",
        f"За доставување на документите од друго лице, види „{_PROXY_TITLE}“.",
    )
    other = _faq("Контакт", "Контакт информации.")
    database, lookup = _database(monkeypatch, [proxy])

    final, sources = _expand(
        database,
        [_question_candidate(big), _question_candidate(other)],
    )

    assert [c.key for c in final] == [f"Q:{big.id}", f"Q:{proxy.id}", f"Q:{other.id}"]
    assert final[1].context_text == _question_candidate(proxy).context_text
    assert "Провери го рокот." in final[1].context_text
    assert [s.title for s in sources] == [big.name, proxy.name, other.name]
    lookup.assert_awaited_once()


def test_pointer_to_selected_faq_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _faq(_PROXY_TITLE, "Да, со полномошно.")
    big = _faq("Упис на прв циклус", _pointing_to(_PROXY_TITLE))
    second = _faq("Упис на втор циклус", _pointing_to(_PROXY_TITLE))
    database, _ = _database(monkeypatch, [proxy])
    initial = [
        _question_candidate(big),
        _question_candidate(proxy),
        _question_candidate(second),
    ]

    final, sources = _expand(database, initial)

    assert final == initial
    assert sources == tuple(c.retrieval_source for c in initial)


def test_faq_pointed_to_twice_is_added_once_after_the_first_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _faq(_PROXY_TITLE, "Да, со полномошно.")
    big = _faq("Упис на прв циклус", _pointing_to(_PROXY_TITLE))
    second = _faq("Упис на втор циклус", _pointing_to(_PROXY_TITLE))
    database, _ = _database(monkeypatch, [proxy])

    final, _ = _expand(
        database,
        [_question_candidate(big), _question_candidate(second)],
    )

    assert [c.key for c in final] == [
        f"Q:{big.id}",
        f"Q:{proxy.id}",
        f"Q:{second.id}",
    ]


def test_quoted_text_that_is_not_a_faq_name_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = _faq(
        "Упис на прв циклус",
        "Пополнете го образецот „Пријава за упис“ и види „Непостоечко прашање“.",
    )
    database, lookup = _database(monkeypatch, [])
    initial = [_question_candidate(big)]

    final, sources = _expand(database, initial)

    assert final == initial
    assert sources == (initial[0].retrieval_source,)
    assert lookup.await_args is not None
    assert lookup.await_args.args[1] == ["Пријава за упис", "Непостоечко прашање"]


def test_expansion_is_capped_at_three_and_follows_one_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointed = [_faq(f"Факт {index}", _pointing_to("Втор скок")) for index in range(4)]
    second_hop = _faq("Втор скок", "Не треба да се додаде.")
    big = _faq("Преглед", _pointing_to(*(q.name for q in pointed)))
    database, lookup = _database(monkeypatch, [*pointed, second_hop])

    final, _ = _expand(database, [_question_candidate(big)])

    assert [c.key for c in final] == [
        f"Q:{big.id}",
        *(f"Q:{q.id}" for q in pointed[:3]),
    ]
    lookup.assert_awaited_once()


def test_pointed_faq_source_follows_visibility_and_source_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _faq(_PROXY_TITLE, "Да, со полномошно.")
    hidden_proxy = _faq("Скриен факт", "Не е видлив извор.")
    big = _faq("Упис на прв циклус", _pointing_to(_PROXY_TITLE))
    hidden = _faq("Слабо поврзано", _pointing_to("Скриен факт"))
    chunk = _chunk_candidate(
        ChunkSchema(
            id=uuid4(),
            document_id=uuid4(),
            document_name="document",
            document_title="Document",
            chunk_index=0,
            section=None,
            content="Документ.",
        ),
    )
    database, _ = _database(monkeypatch, [proxy, hidden_proxy])
    initial = [_question_candidate(big), chunk, _question_candidate(hidden)]

    final, sources = _expand(
        database,
        initial,
        sources=(initial[0].retrieval_source, chunk.retrieval_source),
        top_k=2,
    )

    assert [c.key for c in final] == [
        f"Q:{big.id}",
        f"Q:{proxy.id}",
        chunk.key,
        f"Q:{hidden.id}",
        f"Q:{hidden_proxy.id}",
    ]
    assert [s.title for s in sources] == [big.name, proxy.name]


def test_pointed_faq_respects_evidence_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    excluded = _faq("Исклучен факт", "Не важи за ова прашање.")
    big = _faq("Преглед", _pointing_to("Исклучен факт"))
    database, _ = _database(monkeypatch, [excluded])
    initial = [_question_candidate(big)]

    final, _ = _expand(
        database,
        initial,
        constraints=FaqConstraints(
            kind="exclusion",
            reason="entity_exclusion",
            office_titles=(),
            excluded_title="Исклучен факт",
        ),
    )

    assert final == initial


@pytest.mark.parametrize("error", [PostgresError("boom"), OSError("down")])
def test_lookup_failure_returns_unexpanded_context(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    big = _faq("Упис на прв циклус", _pointing_to(_PROXY_TITLE))
    monkeypatch.setattr(
        context_module,
        "get_questions_by_names_query",
        AsyncMock(side_effect=error),
    )
    initial = [_question_candidate(big)]

    final, sources = _expand(Database("postgresql://pointer-test"), initial)

    assert final == initial
    assert sources == (initial[0].retrieval_source,)
    assert "FAQ pointer lookup failed" in caplog.text


def test_names_lookup_is_one_exact_batched_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database("postgresql://pointer-test")
    fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(database, "fetch", fetch)

    anyio.run(get_questions_by_names_query, database, ["А", "Б"])

    assert fetch.await_args is not None
    sql, names = fetch.await_args.args
    assert "name = ANY($1::text[])" in sql
    assert names == ["А", "Б"]


def test_search_returns_the_pointed_faq_in_text_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _faq(_PROXY_TITLE, "Да, друго лице може со полномошно.")
    big = _faq(
        "Упис на прв циклус",
        f"За доставување на документите од друго лице, види „{_PROXY_TITLE}“.",
    )
    big = big.model_copy(update={"distance": 0.2})

    async def embed(*args, **kwargs):
        return [0.1]

    async def vector_search(*args, **kwargs):
        return [big], []

    def build_variants(*args, **kwargs):
        raw = QueryVariant(kind="raw", text="upis", is_document=False)
        return QueryVariantBundle(variants=(raw,), rerank_query="upis")

    class RerankResponse:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.9}]}

    async def rerank(_payload):
        return RerankResponse()

    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "build_query_variants", build_variants)
    monkeypatch.setattr(context_module, "_search_both", vector_search)
    monkeypatch.setattr(
        context_module,
        "get_matching_questions",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(context_module, "_post_rerank", rerank)
    _database(monkeypatch, [proxy])

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "upis",
            Model.BGE_M3_LOCAL,
        )

    result = anyio.run(run)

    assert proxy.content in result.text
    assert result.text.index(big.content) < result.text.index(proxy.content)
    assert [s.title for s in result.sources] == [big.name, proxy.name]
