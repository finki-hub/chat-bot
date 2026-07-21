import importlib.util
from pathlib import Path

import anyio
from fastapi.responses import StreamingResponse

from app.data.embedding_lifecycle import EmbeddingLifecycleCount
from app.data.embedding_lifecycle_sql import EmbeddingCorpus


def _loader_module():
    script_path = Path(__file__).parents[2] / "scripts" / "load_professor_documents.py"
    spec = importlib.util.spec_from_file_location(
        "professor_document_loader",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_reports_current_and_dirty_professor_document_counts(
    monkeypatch,
    capsys,
) -> None:
    class FakeDatabase:
        async def init(self) -> None:
            return None

        async def disconnect(self) -> None:
            return None

        async def fetchval(self, *_args) -> int:
            return 99

    async def response_body():
        yield 'data: {"status":"ok","total":1}\n\n'

    async def stream_fill(*_args):
        return StreamingResponse(response_body(), media_type="text/event-stream")

    async def lifecycle_counts(*_args):
        return (
            EmbeddingLifecycleCount(EmbeddingCorpus.QUESTION, ready=1, dirty=0),
            EmbeddingLifecycleCount(EmbeddingCorpus.CHUNK, ready=1, dirty=0),
            EmbeddingLifecycleCount(EmbeddingCorpus.DIPLOMA, ready=1, dirty=0),
            EmbeddingLifecycleCount(
                EmbeddingCorpus.PROFESSOR_DOCUMENT,
                ready=1,
                dirty=2,
            ),
        )

    async def upsert(*_args, **_kwargs):
        return "paper"

    async def close_client() -> None:
        return None

    def settings():
        return type("Settings", (), {"DATABASE_URL": ""})()

    loader = _loader_module()
    monkeypatch.setattr(loader, "Database", lambda **_kwargs: FakeDatabase())
    monkeypatch.setattr(
        loader,
        "Settings",
        settings,
    )
    monkeypatch.setattr(loader, "init_http_client", lambda: None)
    monkeypatch.setattr(loader, "close_http_client", close_client)
    monkeypatch.setattr(loader, "upsert_professor_document", upsert)
    monkeypatch.setattr(
        loader,
        "stream_fill_professor_document_embeddings",
        stream_fill,
    )
    monkeypatch.setattr(loader, "lifecycle_counts", lifecycle_counts, raising=False)

    assert anyio.run(loader.run, [{"title": "Paper"}]) == 0
    assert "VERIFY: current papers = 1 dirty papers = 2" in capsys.readouterr().out
