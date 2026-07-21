# mypy: disable-error-code="arg-type"

import json

from asyncpg import Record

from app.data.connection import Database
from app.data.embedding_sql import (
    current_embedding_predicate,
    dirty_embedding_predicate,
    embedding_column_name,
    embedding_vector_sql,
)
from app.llms.models import (
    MODEL_DISTANCE_THRESHOLDS,
    Model,
)
from app.utils.database import embedding_to_pgvector


async def upsert_professor_document(
    db: Database,
    external_id: str,
    title: str,
    abstract: str | None,
    year: int | None,
    topics: list[str],
    canonical_authors: list[str],
    sources: list[str],
) -> str:
    query = """
    INSERT INTO professor_document (
        external_id, title, abstract, year, topics, canonical_authors, sources
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (external_id) DO UPDATE SET
        title = EXCLUDED.title,
        abstract = EXCLUDED.abstract,
        year = EXCLUDED.year,
        topics = EXCLUDED.topics,
        canonical_authors = EXCLUDED.canonical_authors,
        sources = EXCLUDED.sources,
        updated_at = NOW()
    RETURNING external_id
    """
    result = await db.fetchval(
        query,
        external_id,
        title,
        abstract,
        year,
        json.dumps(topics, ensure_ascii=False),
        json.dumps(canonical_authors, ensure_ascii=False),
        json.dumps(sources, ensure_ascii=False),
    )
    return str(result)


async def fetch_professor_document_rows_for_fill(
    db: Database,
    model: Model,
) -> list[Record]:
    predicate = dirty_embedding_predicate(model, embedding_column_name(model))
    return await db.fetch(
        f"SELECT id, title, abstract, embedding_revision FROM professor_document WHERE {predicate.sql} ORDER BY external_id ASC",  # noqa: S608
        *predicate.parameters,
    )


async def get_all_paper_authors(db: Database) -> list[list[str]]:
    rows = await db.fetch("SELECT canonical_authors FROM professor_document")
    out: list[list[str]] = []
    for row in rows:
        authors = row["canonical_authors"]
        if isinstance(authors, str):
            authors = json.loads(authors)
        out.append(list(authors or []))
    return out


async def get_closest_professor_documents(
    db: Database,
    embedded_query: list[float],
    model: Model,
    limit: int = 50,
    threshold: float | None = None,
) -> list[Record]:
    embedding = embedding_vector_sql(model, embedded_query)
    predicate = current_embedding_predicate(
        model,
        embedding.column_ref,
        version_parameter=4,
    )
    if threshold is None:
        threshold = MODEL_DISTANCE_THRESHOLDS.get(model, 0.5)

    sql = f"""
    SELECT
        external_id,
        title,
        year,
        canonical_authors,
        {embedding.distance_operand} <=> {embedding.query_operand} AS distance
    FROM professor_document
    WHERE {predicate.sql}
        AND {embedding.distance_operand} <=> {embedding.query_operand} < $3
    ORDER BY distance
    LIMIT $2
    """  # noqa: S608

    return await db.fetch(
        sql,
        embedding_to_pgvector(embedded_query),
        limit,
        threshold,
        *predicate.parameters,
    )
