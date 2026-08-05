import argparse
import json
from pathlib import Path

import anyio
import pytest

from app.data.connection import Database
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from tests.eval import run_eval


def test_main_async_json_preserves_legacy_config_and_reports_effective_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    golden_path = tmp_path / "golden.jsonl"
    output_path = tmp_path / "results.json"
    golden_path.write_text(
        '{"id":"case","query":"query","anchor":{"type":"none"}}\n',
        encoding="utf-8",
    )

    class FakeDatabase:
        async def init(self) -> None:
            return None

        async def disconnect(self) -> None:
            return None

    async def close_http_client() -> None:
        return None

    async def evaluate_one(*args, **kwargs) -> run_eval.Result:
        return run_eval.Result(
            example=args[1],
            effective_transform_mode=QueryTransformMode.RAW,
            effective_initial_k=30,
            effective_per_query_k=31,
        )

    monkeypatch.setattr(run_eval, "Database", lambda dsn: FakeDatabase())
    monkeypatch.setattr(run_eval, "init_http_client", lambda: None)
    monkeypatch.setattr(run_eval, "close_http_client", close_http_client)
    monkeypatch.setattr(run_eval, "evaluate_one", evaluate_one)

    namespace = argparse.Namespace(
        concurrency=1,
        embedding_model=Model.BGE_M3_LOCAL.value,
        golden=str(golden_path),
        ideal_limit=60,
        initial_k=30,
        json=str(output_path),
        limit=None,
        no_transform=False,
        query_transform_model=Model.GPT_5_4_MINI.value,
        top_k=10,
        transform_mode=QueryTransformMode.REWRITE_HYDE,
    )

    exit_code = anyio.run(run_eval.main_async, namespace)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["config"] == {
        "embedding_model": Model.BGE_M3_LOCAL.value,
        "query_transform_model": Model.GPT_5_4_MINI.value,
        "query_transform_mode": QueryTransformMode.REWRITE_HYDE.value,
        "initial_k": 60,
        "per_query_k": 21,
        "ideal_limit": 60,
        "requested_query_transform_mode": QueryTransformMode.REWRITE_HYDE.value,
        "requested_initial_k": 30,
        "budget_scope": "per_example_effective_mode",
        "ideal_limit_floor": 60,
        "top_k": 10,
        "reranker_min_score": run_eval.settings.RERANKER_MIN_SCORE,
    }
    assert payload["results"][0]["effective_transform_mode"] == "raw"
    assert payload["results"][0]["effective_initial_k"] == 30
    assert payload["results"][0]["effective_per_query_k"] == 31


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
