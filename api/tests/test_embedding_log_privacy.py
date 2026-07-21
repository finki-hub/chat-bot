import logging

import anyio
import pytest
from anyio.lowlevel import checkpoint

from app.llms import embedding_generation as generation_module
from app.llms import embeddings as embeddings_module
from app.llms.models import Model

_PRIVATE_PROMPT = "private prompt content"
_PRIVATE_PROVIDER_ERROR = (
    "provider rejected key=secret at https://private-provider.example/v1"
)


def test_embedding_logs_shape_without_raw_text(caplog, monkeypatch):
    async def dispatch(*args, **kwargs):
        await checkpoint()
        return [0.1]

    monkeypatch.setattr(generation_module, "_dispatch_embeddings", dispatch)
    caplog.set_level(logging.INFO, logger=generation_module.__name__)

    async def collect():
        return await embeddings_module.generate_embeddings(
            _PRIVATE_PROMPT,
            Model.BGE_M3,
        )

    result = anyio.run(collect)

    assert result == [0.1]
    assert _PRIVATE_PROMPT not in caplog.text
    assert "text_count=1" in caplog.text
    assert f"total_chars={len(_PRIVATE_PROMPT)}" in caplog.text


def test_embedding_failure_logs_only_safe_error_metadata(caplog, monkeypatch):
    async def fail_dispatch(*args, **kwargs):
        await checkpoint()
        raise RuntimeError(_PRIVATE_PROVIDER_ERROR)

    monkeypatch.setattr(generation_module, "_dispatch_embeddings", fail_dispatch)
    caplog.set_level(logging.ERROR, logger=generation_module.__name__)

    async def collect():
        return await embeddings_module.generate_embeddings(
            _PRIVATE_PROMPT,
            Model.BGE_M3,
        )

    with pytest.raises(RuntimeError, match="provider rejected"):
        anyio.run(collect)

    assert _PRIVATE_PROMPT not in caplog.text
    assert _PRIVATE_PROVIDER_ERROR not in caplog.text
    assert "error_type=RuntimeError" in caplog.text
