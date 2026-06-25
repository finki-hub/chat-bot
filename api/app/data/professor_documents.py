# mypy: disable-error-code="arg-type"

import json

from asyncpg import Record

from app.data.connection import Database
from app.llms.models import (
    HALFVEC_EMBEDDING_MODELS,
    MODEL_DISTANCE_THRESHOLDS,
    MODEL_EMBEDDINGS_COLUMNS,
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
    model_column: str,
) -> list[Record]:
    return await db.fetch(
        f"SELECT id, title, abstract FROM professor_document WHERE {model_column} IS NULL ORDER BY external_id ASC",  # noqa: S608
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
    embedding_column = MODEL_EMBEDDINGS_COLUMNS[model]
    if threshold is None:
        threshold = MODEL_DISTANCE_THRESHOLDS.get(model, 0.5)

    if model in HALFVEC_EMBEDDING_MODELS:
        dims = len(embedded_query)
        col_expr = f"{embedding_column}::halfvec({dims})"
        param_expr = f"$1::halfvec({dims})"
    else:
        col_expr = embedding_column
        param_expr = "$1"

    sql = f"""
    SELECT
        external_id,
        title,
        year,
        canonical_authors,
        {col_expr} <=> {param_expr} AS distance
    FROM professor_document
    WHERE {embedding_column} IS NOT NULL
        AND {col_expr} <=> {param_expr} < $3
    ORDER BY distance
    LIMIT $2
    """  # noqa: S608

    return await db.fetch(
        sql,
        embedding_to_pgvector(embedded_query),
        limit,
        threshold,
    )
