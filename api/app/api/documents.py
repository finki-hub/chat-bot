import hashlib
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.data.documents import (
    delete_document_query,
    get_document_by_name_query,
    list_documents_query,
    replace_document_with_chunks,
)
from app.llms.chunking import chunk_markdown
from app.llms.embeddings import stream_fill_chunk_embeddings
from app.schemas.documents import (
    DocumentSchema,
    FillChunkEmbeddingsSchema,
    IngestDocumentSchema,
)
from app.utils.auth import verify_api_key

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    dependencies=[db_dep],
)


@router.get(
    "/list",
    summary="List all documents",
    description="Return all stored source-of-truth documents with their chunk counts.",
    status_code=status.HTTP_200_OK,
    operation_id="listDocuments",
)
async def list_documents(db: Database = db_dep) -> list[DocumentSchema]:
    return await list_documents_query(db)


@router.get(
    "/name/{name:path}",
    summary="Get a document by name",
    description="Return the matching document, or 404 if not found.",
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Document not found"}},
    operation_id="getDocumentByName",
)
async def get_document_by_name(name: str, db: Database = db_dep) -> DocumentSchema:
    decoded = urllib.parse.unquote(name)
    document = await get_document_by_name_query(db, decoded)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{decoded}' not found",
        )
    return document


@router.post(
    "/",
    summary="Ingest (or replace) a document",
    description=(
        "Chunk a Markdown document into the `chunk` table. Idempotent by name: an existing "
        "document with the same name is fully replaced (its chunks are removed). Embeddings "
        "are left unfilled — call /documents/fill afterwards. If the content is unchanged "
        "(same hash) the existing document is returned untouched unless `force=true`."
    ),
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_200_OK: {
            "description": "Content unchanged (same hash); existing document returned",
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Empty document / no chunks produced",
        },
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    dependencies=[api_key_dep],
    operation_id="ingestDocument",
)
async def ingest_document(
    payload: IngestDocumentSchema,
    response: Response,
    db: Database = db_dep,
    force: bool = Query(
        default=False,
        description="Re-chunk and replace even if the content hash is unchanged",
    ),
) -> DocumentSchema:
    source_hash = hashlib.sha256(payload.content.encode("utf-8")).hexdigest()

    existing = await get_document_by_name_query(db, payload.name)
    if existing and existing.source_hash == source_hash and not force:
        # Unchanged content: nothing was created, so report 200 instead of the 201 default.
        response.status_code = status.HTTP_200_OK
        return existing

    chunks = chunk_markdown(payload.content)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document produced no chunks (empty content?)",
        )

    return await replace_document_with_chunks(db, payload, source_hash, chunks)


@router.delete(
    "/{name:path}",
    summary="Delete a document",
    description="Delete the document and all of its chunks, returning the deleted record.",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Document not found"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    dependencies=[api_key_dep],
    operation_id="deleteDocument",
)
async def delete_document(name: str, db: Database = db_dep) -> DocumentSchema:
    decoded = urllib.parse.unquote(name)
    existing = await get_document_by_name_query(db, decoded)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{decoded}' not found",
        )
    await delete_document_query(db, decoded)
    return existing


@router.post(
    "/fill",
    summary="Fill chunk embeddings with progress",
    description="Stream back per-chunk embedding progress as Server-Sent Events (SSE).",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    operation_id="fillChunkEmbeddings",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Unsupported model"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing API Key"},
    },
    dependencies=[api_key_dep],
)
async def fill_chunk_embeddings(
    payload: FillChunkEmbeddingsSchema,
    db: Database = db_dep,
) -> StreamingResponse:
    return stream_fill_chunk_embeddings(
        db,
        payload.embeddings_model,
        documents=payload.documents,
        all_chunks=payload.all_chunks,
        all_models=payload.all_models,
    )
