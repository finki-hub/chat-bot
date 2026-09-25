import json

import anyio
import pytest

from tests.eval import compare_eval, run_eval


@pytest.mark.parametrize(
    ("failed", "limit", "completed", "errors"),
    [
        ((), 0, 4, 0),
        (("faq-1", "faq-2", "abstain-1", "abstain-2"), 0, 0, 4),
        (("faq-1", "abstain-1"), 0, 2, 2),
        ((), 1, 1, 0),
    ],
)
def test_runner_execution_errors_are_not_quality_results(
    monkeypatch, tmp_path, capsys, *, failed, limit, completed, errors
):
    rows = [
        {
            "id": name,
            "query": name,
            "anchor": {"type": "none"}
            if name.startswith("abstain")
            else {"type": "Q", "name": "expected"},
            "difficulty": "easy",
            "cohort": "sample",
        }
        for name in ("faq-1", "faq-2", "abstain-1", "abstain-2")
    ]
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "result.json"
    calls = []

    class FakeDatabase:
        async def init(self):
            return None

        async def disconnect(self):
            return None

    async def close_http_client():
        return None

    async def embed(query, *args, **kwargs):
        calls.append(query)
        if query in failed:
            raise TimeoutError("credential=must-not-appear")
        return [0.1]

    async def empty(*args, **kwargs):
        return []

    async def unexpected_rerank(*args, **kwargs):
        pytest.fail("Empty candidates must not call the reranker")

    monkeypatch.setattr(run_eval, "Database", lambda dsn: FakeDatabase())
    monkeypatch.setattr(run_eval, "init_http_client", lambda: None)
    monkeypatch.setattr(run_eval, "close_http_client", close_http_client)
    monkeypatch.setattr(run_eval, "_embed_variant", embed)
    monkeypatch.setattr(run_eval, "get_closest_questions", empty)
    monkeypatch.setattr(run_eval, "get_closest_chunks", empty)
    monkeypatch.setattr(run_eval, "get_matching_questions", empty)
    monkeypatch.setattr(run_eval, "_post_rerank", unexpected_rerank)
    ns = run_eval.build_parser().parse_args(
        [
            "--golden",
            str(golden),
            "--json",
            str(output),
            "--transform-mode",
            "raw",
            "--concurrency",
            "1",
            "--limit",
            str(limit),
        ]
    )

    code = anyio.run(run_eval.main_async, ns)
    captured = capsys.readouterr()
    serialized = output.read_text(encoding="utf-8")
    payload = json.loads(serialized)

    assert code == (2 if errors else 0)
    assert len(calls) == completed + errors
    assert len(payload["results"]) == completed + errors
    assert (
        f"total={completed + errors} completed={completed} errors={errors}"
        in captured.out
    )
    assert "credential=must-not-appear" not in captured.out + captured.err + serialized
    for row in payload["results"]:
        is_error = row["id"] in failed
        assert row["execution_status"] == ("error" if is_error else "completed")
        assert row["error_type"] == ("TimeoutError" if is_error else None)
        assert row["effective_transform_mode"] == (None if is_error else "raw")
        if is_error:
            assert f"[ANN-MISS] {row['id']}" not in captured.out
    if completed == 0:
        assert "ANN recall" not in captured.out
        assert "retrieved a (false)" not in captured.out
    elif errors:
        assert "overall (n=1)" in captured.out
        assert "source=faq (n=1)" in captured.out
        assert "difficulty=easy (n=1)" in captured.out
        assert "cohort=sample (n=1)" in captured.out
        assert "source for 0/1" in captured.out
        assert "source for 0/2" not in captured.out


@pytest.mark.parametrize("effective", ["present", "absent", "null"])
@pytest.mark.parametrize("failed_side", ["baseline", "current"])
def test_compare_rejects_explicit_error_regardless_of_config_or_budget(
    tmp_path, capsys, effective, failed_side
):
    paths = {}
    for side in ("baseline", "current"):
        row = {
            "id": "case",
            "anchor": {"type": "none"},
            "ann_ideal": False,
            "ann_prod": False,
            "final": False,
            "rank": None,
            "execution_status": "error" if side == failed_side else "completed",
            "error_type": "TimeoutError" if side == failed_side else None,
        }
        if effective != "absent":
            row.update(
                effective_transform_mode="raw" if effective == "present" else None,
                effective_initial_k=30 if effective == "present" else None,
                effective_per_query_k=31 if effective == "present" else None,
            )
        path = tmp_path / f"{side}.json"
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "query_transform_mode": "raw",
                        "initial_k": 30,
                        "per_query_k": 31,
                    },
                    "results": [row],
                }
            ),
            encoding="utf-8",
        )
        paths[side] = str(path)

    assert (
        compare_eval.main(
            [
                "--baseline",
                paths["baseline"],
                "--current",
                paths["current"],
                "--max-regressions",
                "999",
            ]
        )
        == 2
    )
    assert "execution failed" in capsys.readouterr().err


def test_error_type_is_bounded_and_sanitized():
    unusual_error = type("Unsafe\nClass:" + "x" * 200, (Exception,), {})
    label = run_eval.execution_error_type(unusual_error("credential=secret"))
    assert len(label) <= 80
    assert label.isascii()
    assert all(char.isalnum() or char == "_" for char in label)
    assert "credential" not in label
