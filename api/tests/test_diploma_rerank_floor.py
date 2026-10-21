from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from app.data.connection import Database
from app.llms import diploma_retrieval
from app.llms.models import Model
from app.schemas.diplomas import DiplomaSchema


def _diploma(external_id: str) -> DiplomaSchema:
    now = datetime.now(UTC)
    return DiplomaSchema(
        id=uuid4(),
        external_id=external_id,
        title=f"Thesis {external_id}",
        description="",
        mentor="Mentor",
        member1=None,
        member2=None,
        status="defended",
        date_of_submission=date(2026, 1, 1),
        created_at=now,
        updated_at=now,
        distance=0.2,
    )


@pytest.mark.anyio
async def test_diploma_retrieval_uses_its_own_rerank_floor(monkeypatch):
    candidates = [_diploma("a"), _diploma("b")]

    async def closest(*_args, **_kwargs):
        return candidates

    async def rerank(_payload):
        return SimpleNamespace(
            json=lambda: {
                "reranked_documents": [
                    {"index": 0, "score": 0.5},
                    {"index": 1, "score": 0.05},
                ],
            },
        )

    monkeypatch.setattr(diploma_retrieval, "get_closest_diplomas", closest)
    monkeypatch.setattr(diploma_retrieval, "post_rerank", rerank)
    monkeypatch.setattr(diploma_retrieval.settings, "RERANKER_MIN_SCORE", 0.03)
    monkeypatch.setattr(diploma_retrieval.settings, "DIPLOMA_RERANKER_MIN_SCORE", 0.1)

    selected = await diploma_retrieval.retrieve_similar_diplomas(
        cast(Database, None),
        "query",
        Model.BGE_M3_LOCAL,
        initial_k=10,
        top_k=5,
        query_embedding=[0.0],
    )

    assert [d.external_id for d in selected] == ["a"]
