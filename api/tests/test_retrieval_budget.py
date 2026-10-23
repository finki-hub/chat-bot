from app.llms.retrieval_budget import retrieval_budget


def test_retrieval_budget_preserves_raw_budget_when_requested_budget_is_default():
    budget = retrieval_budget(30)

    assert budget.initial_k == 30
    assert budget.per_query_k == 31


def test_retrieval_budget_raises_small_requests_to_the_minimum():
    budget = retrieval_budget(5)

    assert budget.initial_k == 30
    assert budget.per_query_k == 31


def test_retrieval_budget_keeps_larger_explicit_budget():
    budget = retrieval_budget(90)

    assert budget.initial_k == 90
    assert budget.per_query_k == 91


def test_retrieval_budget_splits_initial_k_across_variants():
    budget = retrieval_budget(30, variant_count=2)

    assert budget.initial_k == 30
    assert budget.per_query_k == 16
