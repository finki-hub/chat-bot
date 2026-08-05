import anyio
import pytest

from app.data.connection import Database
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from tests.eval import run_eval


def test_evaluate_one_budgets_from_retained_query_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def return_raw_bundle(*args, **kwargs):
        raw = QueryVariant(kind="raw", text="query", is_document=False)
        return QueryVariantBundle(variants=(raw,), rerank_query="query")

    async def return_embedding(*args, **kwargs):
        return [0.1]

    async def return_no_results(*args, **kwargs):
        return []

    monkeypatch.setattr(run_eval, "build_query_variants", return_raw_bundle)
    monkeypatch.setattr(run_eval, "_embed_variant", return_embedding)
    monkeypatch.setattr(run_eval, "get_closest_questions", return_no_results)
    monkeypatch.setattr(run_eval, "get_closest_chunks", return_no_results)

    async def evaluate() -> run_eval.Result:
        return await run_eval.evaluate_one(
            Database("postgresql://unused"),
            run_eval.Example(
                id="raw-fallback",
                query="query",
                anchor={"type": "none"},
            ),
            embedding_model=Model.BGE_M3_LOCAL,
            qt_model=Model.GPT_5_4_MINI,
            initial_k=30,
            top_k=10,
            ideal_limit=60,
            transform_mode=QueryTransformMode.REWRITE_HYDE,
        )

    result = anyio.run(evaluate)

    assert result.effective_transform_mode == QueryTransformMode.RAW
    assert result.effective_initial_k == 30
    assert result.effective_per_query_k == 31
