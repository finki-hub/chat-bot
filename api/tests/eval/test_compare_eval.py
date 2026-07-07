import json

from .compare_eval import compare_runs, main, render_report
from .run_eval import Example, final_context_hit


def _result(
    example_id: str,
    *,
    anchor_type: str,
    final: bool,
    ann_ideal: bool = True,
    ann_prod: bool = True,
    difficulty: str = "easy",
) -> dict:
    anchor: dict[str, int | str] = {"type": anchor_type}
    if anchor_type == "Q":
        anchor["name"] = f"question-{example_id}"
    if anchor_type == "C":
        anchor["document_name"] = f"document-{example_id}"
        anchor["chunk_index"] = 1

    return {
        "id": example_id,
        "difficulty": difficulty,
        "category": "test",
        "anchor": anchor,
        "ann_ideal": ann_ideal,
        "ann_prod": ann_prod,
        "final": final,
        "rank": 1 if final else None,
        "best_distance": 0.24,
    }


def _run(results: list[dict]) -> dict:
    return {
        "config": {
            "embedding_model": "BAAI/bge-m3",
            "query_transform_mode": "rewrite_hyde",
        },
        "results": results,
    }


def test_compare_runs_identifies_decision_cases_by_bucket():
    baseline = _run(
        [
            _result("faq-stable", anchor_type="Q", final=True),
            _result(
                "chunk-fixed",
                anchor_type="C",
                final=False,
                ann_ideal=False,
                ann_prod=False,
                difficulty="hard",
            ),
            _result("chunk-regressed", anchor_type="C", final=True, difficulty="hard"),
            _result(
                "abstain-clean",
                anchor_type="none",
                final=False,
                difficulty="abstain",
            ),
        ],
    )
    current = _run(
        [
            _result("faq-stable", anchor_type="Q", final=True),
            _result("chunk-fixed", anchor_type="C", final=True, difficulty="hard"),
            _result("chunk-regressed", anchor_type="C", final=False, difficulty="hard"),
            _result(
                "abstain-clean",
                anchor_type="none",
                final=True,
                difficulty="abstain",
            ),
        ],
    )

    comparison = compare_runs(baseline, current)

    assert [case.current.id for case in comparison.fixed] == ["chunk-fixed"]
    assert [case.current.id for case in comparison.new_regressions] == [
        "abstain-clean",
        "chunk-regressed",
    ]
    assert comparison.bucket_deltas["source=chunk"][0].final_rate == 0.5
    assert comparison.bucket_deltas["source=chunk"][1].final_rate == 0.5
    assert comparison.bucket_deltas["abstain"][1].final_rate == 1.0


def test_render_report_names_fixed_and_regressed_examples():
    comparison = compare_runs(
        _run([_result("was-missed", anchor_type="Q", final=False)]),
        _run([_result("was-missed", anchor_type="Q", final=True)]),
    )

    report = render_report(comparison)

    assert "Fixed cases" in report
    assert "was-missed" in report
    assert "Decision: PASS" in report


def test_cli_fails_when_new_regressions_exceed_budget(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(
        json.dumps(_run([_result("new-miss", anchor_type="C", final=True)])),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps(_run([_result("new-miss", anchor_type="C", final=False)])),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--max-regressions",
            "0",
        ],
    )

    assert exit_code == 1


def test_cli_rejects_current_run_that_omits_baseline_case(tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(
        json.dumps(_run([_result("omitted", anchor_type="Q", final=True)])),
        encoding="utf-8",
    )
    current_path.write_text(json.dumps(_run([])), encoding="utf-8")

    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
        ],
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "baseline-only=omitted" in captured.err


def test_cli_rejects_negative_regression_budget(capsys):
    try:
        main(["--baseline", "a.json", "--current", "b.json", "--max-regressions", "-1"])
    except SystemExit as exc:
        exit_code = exc.code
    else:
        exit_code = None

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "greater than or equal to 0" in captured.err


def test_final_context_hit_marks_abstain_topk_as_leak():
    example = Example(
        id="abstain-001",
        query="Надвор од опсег",
        anchor={"type": "none"},
        difficulty="abstain",
    )

    final, rank = final_context_hit(example, ("none",), [("Q", "some-source")])

    assert final is True
    assert rank == 1


def test_cli_reports_missing_eval_file_without_traceback(tmp_path, capsys):
    missing = tmp_path / "missing.json"

    exit_code = main(
        [
            "--baseline",
            str(missing),
            "--current",
            str(missing),
        ],
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "missing.json" in captured.err
