import json
from pathlib import Path

from .compare_eval import JsonValue, main


def _result(
    case_id: str,
    final: bool,
    effective_mode: str,
) -> dict[str, JsonValue]:
    per_query_k = 40 if effective_mode == "raw" else 10
    return {
        "id": case_id,
        "difficulty": "hard",
        "category": "test",
        "anchor": {"type": "C"},
        "ann_ideal": final,
        "ann_prod": final,
        "final": final,
        "rank": 1 if final else None,
        "effective_transform_mode": effective_mode,
        "effective_initial_k": 40,
        "effective_per_query_k": per_query_k,
    }


def _run(results: list[JsonValue], mode: str = "rewrite_hyde") -> dict[str, JsonValue]:
    return {
        "config": {
            "query_transform_mode": mode,
            "initial_k": 40,
            "per_query_k": 40 if mode == "raw" else 10,
        },
        "results": results,
    }


def _write(path: Path, run: dict[str, JsonValue]) -> None:
    path.write_text(json.dumps(run), encoding="utf-8")


def test_cli_rejects_fallback_from_requested_retrieval_config(tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write(baseline_path, _run([_result("fallback-miss", True, "rewrite_hyde")]))
    _write(current_path, _run([_result("fallback-miss", False, "raw")]))

    exit_code = main(
        ["--baseline", str(baseline_path), "--current", str(current_path)],
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "rewrite_hyde(initial_k=40, per_query_k=10)" in captured.out
    assert "raw(initial_k=40, per_query_k=40)" in captured.out
    assert "0 new regressions" in captured.out
    assert "Decision: INVALID" in captured.out


def test_cli_allows_intentional_comparison_between_requested_modes(tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write(baseline_path, _run([_result("mode-ab", True, "rewrite_hyde")]))
    _write(current_path, _run([_result("mode-ab", False, "raw")], "raw"))

    exit_code = main(
        ["--baseline", str(baseline_path), "--current", str(current_path)],
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "1 new regressions" in captured.out
    assert "0 incomparable configurations" in captured.out
    assert "Decision: FAIL" in captured.out


def test_cli_keeps_comparable_regressions_when_another_case_falls_back(
    tmp_path,
    capsys,
):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline = _run(
        [
            _result("genuine-regression", True, "rewrite_hyde"),
            _result("fallback", True, "rewrite_hyde"),
        ],
    )
    current = _run(
        [
            _result("genuine-regression", False, "rewrite_hyde"),
            _result("fallback", False, "raw"),
        ],
    )
    _write(baseline_path, baseline)
    _write(current_path, current)

    exit_code = main(
        ["--baseline", str(baseline_path), "--current", str(current_path)],
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "overall: 1/1 (100.0%) -> 0/1 (0.0%)" in captured.out
    assert "1 new regressions" in captured.out
    assert "1 incomparable configurations" in captured.out
