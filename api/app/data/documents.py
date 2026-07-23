# mypy: disable-error-code="arg-type"

import json
from collections.abc import Sequence
from uuid import UUID

from asyncpg import Record

from app.data.connection import Database
from app.data.embedding_sql import (
    current_embedding_predicate,
    dirty_embedding_predicate,
    embedding_column_name,
    embedding_vector_sql,
)
from app.llms.chunking import Chunk
from app.llms.models import (
    MODEL_DISTANCE_THRESHOLDS,
    Model,
    is_bge_m3_lifecycle_model,
)
from app.schemas.documents import ChunkSchema, DocumentSchema, IngestDocumentSchema
from app.utils.database import embedding_to_pgvector


def _document_from_row(row: Record, chunk_count: int | None = None) -> DocumentSchema:
    return DocumentSchema(
        id=row["id"],
        name=row["name"],
        title=row["title"],
        source_type=row["source_type"],
        source_hash=row["source_hash"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        user_id=row["user_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        chunk_count=chunk_count,
    )


async def list_documents_query(db: Database) -> list[DocumentSchema]:
    query = """
    SELECT d.*, COUNT(c.id) AS chunk_count
    FROM document d
    LEFT JOIN chunk c ON c.document_id = d.id
    GROUP BY d.id
    ORDER BY d.name ASC
    """
    result = await db.fetch(query)
    return [_document_from_row(row, chunk_count=row["chunk_count"]) for row in result]


async def get_document_by_name_query(
    db: Database,
    name: str,
) -> DocumentSchema | None:
    query = """
    SELECT d.*, COUNT(c.id) AS chunk_count
    FROM document d
    LEFT JOIN chunk c ON c.document_id = d.id
    WHERE d.name = $1
    GROUP BY d.id
    """
    result = await db.fetchrow(query, name)
    if not result:
        return None
    return _document_from_row(result, chunk_count=result["chunk_count"])


async def delete_document_query(db: Database, name: str) -> None:
    # ON DELETE CASCADE removes the document's chunks.
    await db.execute("DELETE FROM document WHERE name = $1", name)


async def replace_document_with_chunks(
    db: Database,
    payload: IngestDocumentSchema,
    source_hash: str,
    chunks: Sequence[Chunk],
) -> DocumentSchema:
    """Atomically (re)create a document and its chunks by name; old chunks go via ON DELETE CASCADE."""
    async with db.transaction() as conn:
        await conn.execute("DELETE FROM document WHERE name = $1", payload.name)
        row = await conn.fetchrow(
            """
            INSERT INTO document (name, title, source_type, source_hash, metadata, user_id)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            RETURNING *
            """,
            payload.name,
            payload.title,
            payload.source_type,
            source_hash,
            json.dumps(payload.metadata) if payload.metadata else None,
            payload.user_id,
        )
        if row is None:
            msg = "INSERT INTO document ... RETURNING returned no row"
            raise RuntimeError(msg)
        if chunks:
            await conn.executemany(
                """
                INSERT INTO chunk (document_id, chunk_index, content, section)
                VALUES ($1, $2, $3, $4)
                """,
                [(row["id"], c.index, c.content, c.section) for c in chunks],
            )

    return _document_from_row(row, chunk_count=len(chunks))


async def get_closest_chunks(
    db: Database,
    embedded_query: list[float],
    model: Model,
    limit: int = 8,
    threshold: float | None = None,
) -> list[ChunkSchema]:
    """Vector search over the chunk table (mirrors get_closest_questions)."""
    embedding = embedding_vector_sql(model, embedded_query, table_alias="c")
    predicate = current_embedding_predicate(
        model,
        embedding.column_ref,
        version_parameter=4,
    )

    if threshold is None:
        threshold = MODEL_DISTANCE_THRESHOLDS.get(model, 0.5)

    sql = f"""
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.content,
        c.section,
        d.name AS document_name,
        d.title AS document_title,
        d.metadata->>'source_url' AS document_url,
        {embedding.distance_operand} <=> {embedding.query_operand} AS distance
    FROM chunk c
    JOIN document d ON d.id = c.document_id
    WHERE {predicate.sql}
        AND {embedding.distance_operand} <=> {embedding.query_operand} < $3
    ORDER BY distance
    LIMIT $2
    """  # noqa: S608

    result = await db.fetch(
        sql,
        embedding_to_pgvector(embedded_query),
        limit,
        threshold,
        *predicate.parameters,
    )

    return [
        ChunkSchema(
            id=row["id"],
            document_id=row["document_id"],
            document_name=row["document_name"],
            document_title=row["document_title"],
            document_url=row.get("document_url"),
            chunk_index=row["chunk_index"],
            section=row["section"],
            content=row["content"],
            distance=row.get("distance", None),
        )
        for row in result
    ]


async def get_chunks_window(
    db: Database,
    refs: Sequence[tuple[UUID, int]],
    model: Model,
    window: int = 1,
) -> list[ChunkSchema]:
    """Chunks within ±window (by chunk_index, same document) of the given center refs,
    including the centers themselves.

    Lets contiguous retrieved chunks be stitched back into a single passage so an answer
    that spans a chunk boundary — common for legal articles split mid-text — isn't
    truncated. Only positions that actually exist are returned.
    """
    wanted: set[tuple[UUID, int]] = {
        (doc_id, idx + delta)
        for doc_id, idx in refs
        for delta in range(-window, window + 1)
    }
    if not wanted:
        return []

    predicate = (
        current_embedding_predicate(
            model,
            "c.embedding_bge_m3",
            version_parameter=3,
        )
        if is_bge_m3_lifecycle_model(model)
        else None
    )
    window_sql = f"""
        SELECT c.id, c.document_id, c.chunk_index, c.content, c.section,
               d.name AS document_name, d.title AS document_title,
               d.metadata->>'source_url' AS document_url
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        WHERE c.document_id = ANY($1::uuid[]) AND c.chunk_index = ANY($2::int[])
        {f"AND {predicate.sql}" if predicate else ""}
        """  # noqa: S608
    rows = await db.fetch(
        window_sql,
        list({doc_id for doc_id, _ in wanted}),
        list({idx for _, idx in wanted}),
        *(() if predicate is None else predicate.parameters),
    )

    return [
        _window_chunk(row)
        for row in rows
        if (row["document_id"], row["chunk_index"]) in wanted
    ]


def _window_chunk(row: Record) -> ChunkSchema:
    return ChunkSchema(
        id=row["id"],
        document_id=row["document_id"],
        document_name=row["document_name"],
        document_title=row["document_title"],
        document_url=row.get("document_url"),
        chunk_index=row["chunk_index"],
        section=row["section"],
        content=row["content"],
        distance=None,
    )


async def fetch_chunk_rows_for_fill(
    db: Database,
    model: Model,
    documents: list[str] | None,
    *,
    all_chunks: bool,
) -> list[Record]:
    """Chunk rows (with document title) needing this model's embedding."""
    select = """
        SELECT c.id, c.content, c.section, c.embedding_revision, d.title AS document_title
        FROM chunk c
        JOIN document d ON d.id = c.document_id
    """
    model_column = embedding_column_name(model)
    if documents:
        placeholders = ",".join("$" + str(i + 1) for i in range(len(documents)))
        where = f"WHERE d.name IN ({placeholders})"
        if all_chunks:
            return await db.fetch(
                f"{select} {where} ORDER BY d.name, c.chunk_index",
                *documents,
            )
        predicate = dirty_embedding_predicate(
            model,
            f"c.{model_column}",
            version_parameter=len(documents) + 1,
        )
        return await db.fetch(
            f"{select} {where} AND {predicate.sql} ORDER BY d.name, c.chunk_index",
            *documents,
            *predicate.parameters,
        )

    if all_chunks:
        return await db.fetch(f"{select} ORDER BY d.name, c.chunk_index")
    predicate = dirty_embedding_predicate(model, f"c.{model_column}")
    return await db.fetch(
        f"{select} WHERE {predicate.sql} ORDER BY d.name, c.chunk_index",
        *predicate.parameters,
    )
