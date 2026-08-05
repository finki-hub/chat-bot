import anyio
import pytest

from app.data.connection import Database
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from tests.eval import run_eval


class _BudgetProbeError(RuntimeError):
    pass


def test_evaluate_one_budgets_from_retained_query_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_modes: list[QueryTransformMode] = []

    async def return_raw_bundle(*args, **kwargs):
        raw = QueryVariant(kind="raw", text="query", is_document=False)
        return QueryVariantBundle(variants=(raw,), rerank_query="query")

    def capture_budget(mode: QueryTransformMode, requested_initial_k: int):
        seen_modes.append(mode)
        raise _BudgetProbeError

    monkeypatch.setattr(run_eval, "build_query_variants", return_raw_bundle)
    monkeypatch.setattr(run_eval, "retrieval_budget", capture_budget)

    async def evaluate() -> None:
        with pytest.raises(_BudgetProbeError):
            await run_eval.evaluate_one(
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

    anyio.run(evaluate)

    assert seen_modes == [QueryTransformMode.RAW]
