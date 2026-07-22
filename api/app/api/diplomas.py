import hashlib
import logging
from datetime import date, datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.data.diplomas import upsert_diploma
from app.llms.embeddings import stream_fill_diploma_embeddings
from app.schemas.diplomas import FillDiplomaEmbeddingsSchema
from app.utils.auth import verify_api_key
from app.utils.http_client import get_http_client
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

_DIPLOMAS_TIMEOUT = httpx.Timeout(timeout=60.0)

_DEFENDED_STATUS = "Одбрана"

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/diplomas",
    tags=["Diplomas"],
    dependencies=[db_dep],
)


def _parse_submission_date(value: object | None) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), "%d.%m.%Y").date()  # noqa: DTZ007
    except ValueError:
        return None


def _compute_external_id(record: dict) -> str:
    raw = (
        f"{record.get('fileId')}|{record.get('student')}"
        f"|{record.get('title')}|{record.get('dateOfSubmission')}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@router.post(
    "/sync",
    summary="Sync diplomas from the upstream Diplomas API",
    description=(
        "Fetch the upstream diploma corpus, keep only defended defenses "
        "(status 'Одбрана'), and UPSERT them keyed on a content-hash external_id. "
        "Returns fetched / defended / upserted counts. Embeddings are left unfilled "
        "— call /diplomas/fill-embeddings afterwards."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
        status.HTTP_502_BAD_GATEWAY: {"description": "Upstream Diplomas API error"},
    },
    dependencies=[api_key_dep],
    operation_id="syncDiplomas",
)
async def sync_diplomas(db: Database = db_dep) -> dict[str, int]:
    client = get_http_client()
    try:
        response = await client.get(
            settings.DIPLOMAS_API_URL,
            timeout=_DIPLOMAS_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.exception("Diplomas API returned an error status")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Diplomas API returned status {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        logger.exception("Connection error to Diplomas API")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to reach the Diplomas API",
        ) from exc

    records = response.json()
    fetched = len(records)
    defended = 0
    upserted = 0

    for record in records:
        if record.get("status") != _DEFENDED_STATUS:
            continue
        defended += 1

        result = await upsert_diploma(
            db,
            external_id=_compute_external_id(record),
            title=record["title"],
            description=record["description"],
            mentor=record["mentor"],
            member1=record.get("member1"),
            member2=record.get("member2"),
            status=record["status"],
            date_of_submission=_parse_submission_date(record.get("dateOfSubmission")),
        )
        if result is not None:
            upserted += 1

    logger.info(
        "Diploma sync complete: fetched=%d defended=%d upserted=%d",
        fetched,
        defended,
        upserted,
    )

    return {"fetched": fetched, "defended": defended, "upserted": upserted}


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
