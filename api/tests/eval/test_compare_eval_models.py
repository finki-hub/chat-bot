import pytest

from .compare_eval import compare_runs
from .compare_eval_models import (
    EffectiveRetrievalConfig,
    EvalJsonError,
    JsonValue,
    parse_cases,
)


def _result(**effective_config: JsonValue) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "id": "case-1",
        "difficulty": "easy",
        "category": "test",
        "anchor": {"type": "Q"},
        "ann_ideal": True,
        "ann_prod": True,
        "final": True,
        "rank": 1,
    }
    result.update(effective_config)
    return result


def _run(
    results: list[JsonValue],
    *,
    mode: str = "rewrite_hyde",
) -> dict[str, JsonValue]:
    return {
        "config": {
            "query_transform_mode": mode,
            "initial_k": 40,
            "per_query_k": 40 if mode == "raw" else 10,
        },
        "results": results,
    }


def test_legacy_result_uses_run_config_when_compared_with_modern_result():
    legacy = _run([_result()])
    modern = _run(
        [
            _result(
                effective_transform_mode="rewrite_hyde",
                effective_initial_k=40,
                effective_per_query_k=10,
            ),
        ],
    )

    comparison = compare_runs(legacy, modern)

    assert comparison.incomparable_configs == []


def test_null_effective_config_is_incomparable():
    unavailable = _run(
        [
            _result(
                effective_transform_mode=None,
                effective_initial_k=None,
                effective_per_query_k=None,
            ),
        ],
    )

    comparison = compare_runs(_run([_result()]), unavailable)

    assert [case.current.id for case in comparison.incomparable_configs] == ["case-1"]


@pytest.mark.parametrize(
    "effective_config",
    [
        {"effective_transform_mode": "raw"},
        {
            "effective_transform_mode": "raw",
            "effective_initial_k": True,
            "effective_per_query_k": 40,
        },
        {
            "effective_transform_mode": "unknown",
            "effective_initial_k": 40,
            "effective_per_query_k": 40,
        },
    ],
)
def test_malformed_effective_config_is_rejected(
    effective_config: dict[str, JsonValue],
):
    with pytest.raises(EvalJsonError):
        compare_runs(_run([_result(**effective_config)]), _run([_result()]))


def test_duplicate_case_ids_are_rejected():
    duplicate = _run([_result(), _result()])

    with pytest.raises(EvalJsonError, match="duplicate case id: case-1"):
        compare_runs(duplicate, _run([_result()]))


def test_prior_requested_only_schema_derives_nominal_budget():
    prior: dict[str, JsonValue] = {
        "config": {
            "requested_query_transform_mode": "rewrite_hyde",
            "requested_initial_k": 30,
        },
        "results": [_result()],
    }
    modern: dict[str, JsonValue] = {
        "config": {
            "query_transform_mode": "rewrite_hyde",
            "initial_k": 60,
            "per_query_k": 21,
            "requested_query_transform_mode": "rewrite_hyde",
            "requested_initial_k": 30,
        },
        "results": [
            _result(
                effective_transform_mode="rewrite_hyde",
                effective_initial_k=60,
                effective_per_query_k=21,
            ),
        ],
    }

    prior_case = parse_cases(prior, "prior")["case-1"]
    comparison = compare_runs(prior, modern)

    assert prior_case.requested_config == EffectiveRetrievalConfig(
        transform_mode="rewrite_hyde",
        initial_k=60,
        per_query_k=21,
    )
    assert comparison.incomparable_configs == []


@pytest.mark.parametrize("missing_key", ["initial_k", "per_query_k"])
def test_partial_nominal_run_config_is_rejected(missing_key: str):
    malformed = _run([_result()])
    config = malformed["config"]
    assert isinstance(config, dict)
    del config[missing_key]

    with pytest.raises(EvalJsonError, match="requested retrieval configuration"):
        compare_runs(malformed, _run([_result()]))


@pytest.mark.parametrize("null_key", ["initial_k", "per_query_k"])
def test_null_nominal_run_budget_is_rejected(null_key: str):
    malformed = _run([_result()])
    config = malformed["config"]
    assert isinstance(config, dict)
    config[null_key] = None

    with pytest.raises(EvalJsonError):
        compare_runs(malformed, _run([_result()]))


@pytest.mark.parametrize(
    "effective_config",
    [
        {
            "effective_transform_mode": "raw",
            "effective_initial_k": 40,
            "effective_per_query_k": 10,
        },
        {
            "effective_transform_mode": "rewrite_hyde",
            "effective_initial_k": 41,
            "effective_per_query_k": 10,
        },
        {
            "effective_transform_mode": "rewrite_hyde",
            "effective_initial_k": 40,
            "effective_per_query_k": 11,
        },
    ],
)
def test_each_effective_config_field_can_make_case_incomparable(
    effective_config: dict[str, JsonValue],
):
    comparison = compare_runs(
        _run([_result()]),
        _run([_result(**effective_config)]),
    )

    assert [case.current.id for case in comparison.incomparable_configs] == ["case-1"]
