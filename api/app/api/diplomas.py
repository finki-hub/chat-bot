import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.llms.embeddings import stream_fill_diploma_embeddings
from app.schemas.diplomas import FillDiplomaEmbeddingsSchema
from app.services.diploma_sync import DiplomaSyncError, sync_defended_diplomas
from app.utils.auth import verify_api_key

logger = logging.getLogger(__name__)

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/diplomas",
    tags=["Diplomas"],
    dependencies=[db_dep],
)


@router.post(
    "/sync",
    summary="Sync diplomas from the upstream Diplomas API",
    description=(
        "Fetch and validate the upstream diploma corpus, keep only defended defenses "
        "(status 'Одбрана'), and atomically write new or changed rows keyed on a "
        "content-hash external_id. Returns fetched / defended / inserted / updated / "
        "upserted counts. The embedding worker fills new embeddings."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Upstream Diplomas API error or invalid data",
        },
    },
    dependencies=[api_key_dep],
    operation_id="syncDiplomas",
)
async def sync_diplomas(db: Database = db_dep) -> dict[str, int]:
    try:
        result = await sync_defended_diplomas(db)
    except DiplomaSyncError as exc:
        logger.exception("Diploma sync failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return {
        "fetched": result.fetched,
        "defended": result.defended,
        "inserted": result.inserted,
        "updated": result.updated,
        "upserted": result.inserted + result.updated,
    }


@router.post(
    "/fill-embeddings",
    summary="Fill diploma embeddings with progress",
    description="Stream back per-diploma embedding progress as Server-Sent Events (SSE).",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    operation_id="fillDiplomaEmbeddings",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Unsupported model"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    dependencies=[api_key_dep],
)
async def fill_diploma_embeddings(
    payload: FillDiplomaEmbeddingsSchema,
    db: Database = db_dep,
) -> StreamingResponse:
    return stream_fill_diploma_embeddings(
        db,
        payload.embeddings_model,
        all_models=payload.all_models,
    )
