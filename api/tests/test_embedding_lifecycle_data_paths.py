import anyio
import pytest

from app.data.connection import Database
from app.data.diplomas import (
    fetch_diploma_rows_for_fill,
    get_backtest_population,
    get_closest_diplomas,
    get_diplomas_without_embeddings,
)
from app.data.documents import fetch_chunk_rows_for_fill, get_closest_chunks
from app.data.professor_documents import (
    fetch_professor_document_rows_for_fill,
    get_closest_professor_documents,
)
from app.data.questions import (
    get_closest_questions,
    get_questions_without_embeddings_query,
)
from app.llms.models import BGE_M3_EMBEDDING_SPEC_VERSION, Model


class CapturingState:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str | int | float, ...]]] = []

    async def fetch(self, query, *args):
        self.calls.append((query, args))
        return []


def _database(state, monkeypatch) -> Database:
    database = Database("postgresql://lifecycle-test")
    monkeypatch.setattr(database, "fetch", state.fetch)
    return database


def _last_call(state):
    return state.calls[-1]


@pytest.mark.parametrize("model", [Model.BGE_M3, Model.BGE_M3_LOCAL])
def test_dirty_selectors_include_old_version_rows_for_both_bge_aliases(
    model: Model,
    monkeypatch,
) -> None:
    async def run() -> None:
        # Given: a BGE alias whose stored vector can have an old spec version.
        state = CapturingState()
        database = _database(state, monkeypatch)

        # When: each corpus asks for fill candidates or an unfilled population.
        await get_questions_without_embeddings_query(database, model)
        question_query, question_args = _last_call(state)
        await fetch_chunk_rows_for_fill(database, model, None, all_chunks=False)
        chunk_query, chunk_args = _last_call(state)
        await get_diplomas_without_embeddings(database, model)
        diploma_query, diploma_args = _last_call(state)
        await fetch_diploma_rows_for_fill(database, model)
        diploma_fill_query, diploma_fill_args = _last_call(state)
        await get_backtest_population(database, model)
        backtest_query, backtest_args = _last_call(state)
        await fetch_professor_document_rows_for_fill(database, model)
        professor_query, professor_args = _last_call(state)

        # Then: NULL vectors and old-version vectors are both dirty work.
        for query, args in (
            (question_query, question_args),
            (chunk_query, chunk_args),
            (diploma_query, diploma_args),
            (diploma_fill_query, diploma_fill_args),
            (professor_query, professor_args),
        ):
            assert "embedding_bge_m3 IS NULL" in query
            assert "embedding_bge_m3_version IS DISTINCT FROM" in query
            assert BGE_M3_EMBEDDING_SPEC_VERSION in args
        assert "embedding_bge_m3 IS NOT NULL" in backtest_query
        assert "embedding_bge_m3_version =" in backtest_query
        assert BGE_M3_EMBEDDING_SPEC_VERSION in backtest_args

    anyio.run(run)


@pytest.mark.parametrize("model", [Model.BGE_M3, Model.BGE_M3_LOCAL])
def test_retrieval_excludes_old_version_rows_for_both_bge_aliases(
    model: Model,
    monkeypatch,
) -> None:
    async def run() -> None:
        # Given: one of the aliases backed by the shared BGE-M3 column.
        state = CapturingState()
        database = _database(state, monkeypatch)
        query_vector = [0.0] * 1024

        # When: every corpus performs vector retrieval.
        await get_closest_questions(database, query_vector, model)
        question_query, question_args = _last_call(state)
        await get_closest_chunks(database, query_vector, model)
        chunk_query, chunk_args = _last_call(state)
        await get_closest_diplomas(database, query_vector, model)
        diploma_query, diploma_args = _last_call(state)
        await get_closest_professor_documents(database, query_vector, model)
        professor_query, professor_args = _last_call(state)

        # Then: exact current-version vectors are the only eligible BGE rows.
        for query, args in (
            (question_query, question_args),
            (chunk_query, chunk_args),
            (diploma_query, diploma_args),
            (professor_query, professor_args),
        ):
            assert "embedding_bge_m3 IS NOT NULL" in query
            assert "embedding_bge_m3_version =" in query
            assert BGE_M3_EMBEDDING_SPEC_VERSION in args

    anyio.run(run)


def test_non_bge_selectors_retain_the_existing_null_only_predicate(
    monkeypatch,
) -> None:
    async def run() -> None:
        # Given: an unrelated hosted embedding model.
        state = CapturingState()
        database = _database(state, monkeypatch)

        # When: its dirty selector and retrieval SQL are generated.
        await get_questions_without_embeddings_query(
            database,
            Model.TEXT_EMBEDDING_3_LARGE,
        )
        dirty_query, dirty_args = _last_call(state)
        await get_closest_questions(
            database,
            [0.0] * 3072,
            Model.TEXT_EMBEDDING_3_LARGE,
        )
        retrieval_query, retrieval_args = _last_call(state)

        # Then: no BGE version predicate or version argument leaks into non-BGE SQL.
        assert "embedding_text_embedding_3_large IS NULL" in dirty_query
        assert "embedding_bge_m3_version" not in dirty_query
        assert BGE_M3_EMBEDDING_SPEC_VERSION not in dirty_args
        assert "embedding_text_embedding_3_large IS NOT NULL" in retrieval_query
        assert "embedding_bge_m3_version" not in retrieval_query
        assert BGE_M3_EMBEDDING_SPEC_VERSION not in retrieval_args

    anyio.run(run)
