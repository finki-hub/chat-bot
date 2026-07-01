import pytest

from app.llms import retrieval_budget as retrieval_budget_module
from app.llms.query_modes import QueryTransformMode
from app.llms.retrieval_budget import (
    UnsupportedRetrievalBudgetModeError,
    retrieval_budget,
)


def test_retrieval_budget_preserves_raw_budget_when_requested_budget_is_default():
    budget = retrieval_budget(QueryTransformMode.RAW, 30)

    assert budget.initial_k == 30
    assert budget.per_query_k == 31


def test_retrieval_budget_expands_rewrite_hyde_budget_when_requested_budget_is_default():
    budget = retrieval_budget(QueryTransformMode.REWRITE_HYDE, 30)

    assert budget.initial_k == 60
    assert budget.per_query_k == 21


def test_retrieval_budget_keeps_larger_explicit_budget_for_rewrite_hyde():
    budget = retrieval_budget(QueryTransformMode.REWRITE_HYDE, 90)

    assert budget.initial_k == 90
    assert budget.per_query_k == 31


def test_retrieval_budget_expands_two_variant_modes_when_requested_budget_is_default():
    rewrite = retrieval_budget(QueryTransformMode.REWRITE, 30)
    hyde = retrieval_budget(QueryTransformMode.HYDE, 30)

    assert rewrite.initial_k == 40
    assert rewrite.per_query_k == 21
    assert hyde.initial_k == 40
    assert hyde.per_query_k == 21


def test_retrieval_budget_raises_explicit_error_when_mode_budget_is_missing(
    monkeypatch,
):
    monkeypatch.delitem(
        retrieval_budget_module._MIN_INITIAL_K_BY_MODE,  # noqa: SLF001 - fault-inject missing mapping
        QueryTransformMode.RAW,
    )

    with pytest.raises(UnsupportedRetrievalBudgetModeError, match="raw"):
        retrieval_budget(QueryTransformMode.RAW, 30)
