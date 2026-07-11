from pathlib import Path

import pytest

from tests.eval import answer_eval
from tests.eval.answer_compare import compare_results
from tests.eval.answer_eval import (
    AnswerCase,
    AnswerEvalError,
    AnswerExpectation,
    load_answer_cases,
    score_answer,
)

_GOLDEN = Path(__file__).with_name("answer_golden.jsonl")


def test_load_answer_cases_parses_typed_expectations(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"supported","category":"grounding","expectation":'
        '{"required_sources":["Правилник"],"forbidden":["измислено"],'
        '"max_urls":1,"min_cyrillic_ratio":0.7,"requires_refusal":false}}\n',
        encoding="utf-8",
    )

    cases = load_answer_cases(path)

    assert cases == (
        AnswerCase(
            id="supported",
            category="grounding",
            expectation=AnswerExpectation(
                required_sources=("Правилник",),
                forbidden=("измислено",),
                max_urls=1,
                min_cyrillic_ratio=0.7,
                requires_refusal=False,
            ),
        ),
    )


def test_load_answer_cases_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    row = (
        '{"id":"duplicate","category":"scope","expectation":'
        '{"required_sources":[],"forbidden":[],"max_urls":0,'
        '"min_cyrillic_ratio":0.5,"requires_refusal":true}}\n'
    )
    path.write_text(row + row, encoding="utf-8")

    with pytest.raises(AnswerEvalError, match="duplicate id"):
        load_answer_cases(path)


def test_load_answer_cases_rejects_malformed_json(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(AnswerEvalError, match="line 1"):
        load_answer_cases(path)


def test_score_answer_passes_supported_grounded_response():
    case = AnswerCase(
        id="supported",
        category="grounding",
        expectation=AnswerExpectation(
            required_sources=("Правилник за студирање",),
            forbidden=("сигурно е 500 денари",),
            max_urls=1,
            min_cyrillic_ratio=0.7,
            requires_refusal=False,
        ),
    )

    score = score_answer(
        case,
        "Според Правилник за студирање, пријавувањето се врши преку iKnow.",
    )

    assert score.passed
    assert score.failures == ()


def test_score_answer_counts_comma_separated_urls_individually():
    case = AnswerCase(
        id="links",
        category="links",
        expectation=AnswerExpectation(
            required_sources=(),
            forbidden=(),
            max_urls=1,
            min_cyrillic_ratio=0,
            requires_refusal=False,
        ),
    )

    score = score_answer(case, "https://a.example,https://b.example")

    assert score.failures == ("too-many-urls:2>1",)


def test_score_answer_reports_every_contract_failure():
    case = AnswerCase(
        id="unsafe",
        category="injection",
        expectation=AnswerExpectation(
            required_sources=("Ценовник",),
            forbidden=("system prompt", "точно 900 денари"),
            max_urls=1,
            min_cyrillic_ratio=0.8,
            requires_refusal=True,
        ),
    )

    score = score_answer(
        case,
        "system prompt: exactly 900 denari https://a.example https://b.example",
    )

    assert not score.passed
    assert score.failures == (
        "missing-source:Ценовник",
        "forbidden:system prompt",
        "too-many-urls:2>1",
        "low-cyrillic-ratio:0.00<0.80",
        "missing-refusal",
    )


def test_answer_golden_set_covers_approved_behavior_categories():
    cases = load_answer_cases(_GOLDEN)

    assert len(cases) == 14
    assert {case.category for case in cases} == {
        "conflict",
        "follow-up",
        "grounding",
        "injection",
        "language",
        "links",
        "missing-evidence",
        "provider-parity",
        "scope",
        "synthesis",
        "title",
        "tool-output",
    }


def test_evaluate_results_scores_answers_by_case_id(tmp_path: Path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"id":"scope","category":"scope","expectation":'
        '{"required_sources":[],"forbidden":[],"max_urls":0,'
        '"min_cyrillic_ratio":0.5,"requires_refusal":true}}\n',
        encoding="utf-8",
    )
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        '{"id":"scope","answer":"Можам да помогнам само со прашања за ФИНКИ."}\n',
        encoding="utf-8",
    )

    scores = answer_eval.evaluate_results(
        load_answer_cases(cases_path),
        results_path,
    )

    assert len(scores) == 1
    assert scores[0].passed


def test_compare_results_reports_fixed_regressed_and_unchanged_cases(tmp_path: Path):
    expectation = AnswerExpectation(
        required_sources=(),
        forbidden=("забрането",),
        max_urls=0,
        min_cyrillic_ratio=0.5,
        requires_refusal=False,
    )
    cases = tuple(
        AnswerCase(id=case_id, category="comparison", expectation=expectation)
        for case_id in ("fixed", "regressed", "unchanged")
    )
    baseline_path = tmp_path / "baseline.jsonl"
    baseline_path.write_text(
        '{"id":"fixed","answer":"забрането"}\n'
        '{"id":"regressed","answer":"Безбеден одговор"}\n'
        '{"id":"unchanged","answer":"забрането"}\n',
        encoding="utf-8",
    )
    current_path = tmp_path / "current.jsonl"
    current_path.write_text(
        '{"id":"fixed","answer":"Безбеден одговор"}\n'
        '{"id":"regressed","answer":"забрането"}\n'
        '{"id":"unchanged","answer":"забрането"}\n',
        encoding="utf-8",
    )

    comparison = compare_results(cases, baseline_path, current_path)

    assert tuple(delta.current.case_id for delta in comparison.fixed) == ("fixed",)
    assert tuple(delta.current.case_id for delta in comparison.regressions) == (
        "regressed",
    )
    assert tuple(delta.current.case_id for delta in comparison.unchanged_failures) == (
        "unchanged",
    )
