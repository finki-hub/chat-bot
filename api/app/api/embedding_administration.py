from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.data.connection import Database
from app.data.db import get_db
from app.data.embedding_lifecycle import (
    EmbeddingCorpus,
    EmbeddingLifecycleCount,
    lifecycle_counts,
    rebuild_embedding_lifecycle,
    wake_embedding_worker,
)
from app.schemas.embedding_administration import (
    EmbeddingCorpusCountResponse,
    EmbeddingFillDirtyResponse,
    EmbeddingHealthResponse,
    EmbeddingLifecycleCountsResponse,
    EmbeddingRebuildResponse,
)
from app.utils.auth import verify_api_key

DatabaseDependency = Annotated[Database, Depends(get_db)]

router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key)],
)


def _count_response(count: EmbeddingLifecycleCount) -> EmbeddingCorpusCountResponse:
    return EmbeddingCorpusCountResponse(ready=count.ready, dirty=count.dirty)


def _counts_response(
    counts: tuple[EmbeddingLifecycleCount, ...],
) -> EmbeddingLifecycleCountsResponse:
    by_corpus = {count.corpus: count for count in counts}
    return EmbeddingLifecycleCountsResponse(
        question=_count_response(by_corpus[EmbeddingCorpus.QUESTION]),
        chunk=_count_response(by_corpus[EmbeddingCorpus.CHUNK]),
        diploma=_count_response(by_corpus[EmbeddingCorpus.DIPLOMA]),
        professor_document=_count_response(
            by_corpus[EmbeddingCorpus.PROFESSOR_DOCUMENT],
        ),
    )


@router.get(
    "/health",
    response_model=EmbeddingHealthResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    operation_id="getEmbeddingLifecycleHealth",
)
async def health(db: DatabaseDependency) -> EmbeddingHealthResponse:
    return EmbeddingHealthResponse(counts=_counts_response(await lifecycle_counts(db)))


@router.post(
    "/fill-dirty",
    response_model=EmbeddingFillDirtyResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    operation_id="wakeEmbeddingLifecycleFill",
)
async def fill_dirty(db: DatabaseDependency) -> EmbeddingFillDirtyResponse:
    await wake_embedding_worker(db)
    return EmbeddingFillDirtyResponse(
        counts=_counts_response(await lifecycle_counts(db)),
    )


@router.post(
    "/rebuild",
    response_model=EmbeddingRebuildResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    operation_id="rebuildEmbeddingLifecycle",
)
async def rebuild(db: DatabaseDependency) -> EmbeddingRebuildResponse:
    return EmbeddingRebuildResponse(
        counts=_counts_response(await rebuild_embedding_lifecycle(db)),
    )
