import logging

import anyio

from app.llms import context as context_module
from app.llms.context import _contextualize_query
from app.llms.models import Model


def test_contextualize_query_logs_lengths_without_raw_text(caplog, monkeypatch):
    async def fake_transform_query(*args, **kwargs):
        return "rewritten private query"

    monkeypatch.setattr(context_module, "transform_query", fake_transform_query)
    caplog.set_level(logging.INFO, logger="app.llms.context")

    result = anyio.run(
        _contextualize_query,
        "original private query",
        Model.CLAUDE_HAIKU_4_5,
        "private history",
    )

    assert result == "rewritten private query"
    assert "original private query" not in caplog.text
    assert "rewritten private query" not in caplog.text
    assert "query_len=" in caplog.text
    assert "condensed_len=" in caplog.text
