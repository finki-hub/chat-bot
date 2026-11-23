# mypy: disable-error-code="arg-type"

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from asyncpg import Record
from asyncpg.pool import PoolConnectionProxy

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
from app.schemas.diplomas import DiplomaSchema
from app.utils.database import embedding_to_pgvector


@dataclass(frozen=True, slots=True)
class DiplomaUpsert:
    external_id: str
    title: str
    description: str
    mentor: str
    member1: str | None
    member2: str | None
    status: str
    date_of_submission: date | None


async def count_diplomas(connection: PoolConnectionProxy[Record]) -> int:
    return int(await connection.fetchval("SELECT count(*) FROM diploma"))


async def upsert_changed_diplomas(
    connection: PoolConnectionProxy[Record],
    diplomas: Sequence[DiplomaUpsert],
) -> tuple[int, int]:
    """Write only new or changed rows and return (inserted, updated)."""
    query = """
    INSERT INTO diploma (
        external_id, title, description, mentor, member1, member2, status, date_of_submission
    )
    SELECT * FROM unnest(
        $1::text[], $2::text[], $3::text[], $4::text[],
        $5::text[], $6::text[], $7::text[], $8::date[]
    )
    ON CONFLICT (external_id) DO UPDATE SET
        title = EXCLUDED.title,
        description = EXCLUDED.description,
        mentor = EXCLUDED.mentor,
        member1 = EXCLUDED.member1,
        member2 = EXCLUDED.member2,
        status = EXCLUDED.status,
        date_of_submission = EXCLUDED.date_of_submission,
        updated_at = NOW()
    WHERE (
        diploma.title, diploma.description, diploma.mentor, diploma.member1,
        diploma.member2, diploma.status, diploma.date_of_submission
    ) IS DISTINCT FROM (
        EXCLUDED.title, EXCLUDED.description, EXCLUDED.mentor, EXCLUDED.member1,
        EXCLUDED.member2, EXCLUDED.status, EXCLUDED.date_of_submission
    )
    RETURNING xmax = 0 AS inserted
    """
    rows = await connection.fetch(
        query,
        [diploma.external_id for diploma in diplomas],
        [diploma.title for diploma in diplomas],
        [diploma.description for diploma in diplomas],
        [diploma.mentor for diploma in diplomas],
        [diploma.member1 for diploma in diplomas],
        [diploma.member2 for diploma in diplomas],
        [diploma.status for diploma in diplomas],
        [diploma.date_of_submission for diploma in diplomas],
    )
    inserted = sum(1 for row in rows if row["inserted"])
    return inserted, len(rows) - inserted


async def get_closest_diplomas(
    db: Database,
    embedded_query: list[float],
    model: Model,
    limit: int = 8,
    threshold: float | None = None,
    *,
    exclude_external_id: str | None = None,
    submitted_before: date | None = None,
) -> list[DiplomaSchema]:
    embedding = embedding_vector_sql(model, embedded_query)
    if threshold is None:
        threshold = MODEL_DISTANCE_THRESHOLDS.get(model, 0.5)

    args: list[object] = [
        embedding_to_pgvector(embedded_query),
        limit,
        threshold,
    ]
    filter_clauses: list[str] = []
    if exclude_external_id is not None:
        args.append(exclude_external_id)
        filter_clauses.append(f"AND external_id <> ${len(args)}")
    order_by = "distance"
    if submitted_before is not None:
        args.append(submitted_before)
        filter_clauses.append(f"AND date_of_submission < ${len(args)}")
        # An approximate index scan would filter too late and return too few rows.
        order_by = f"({embedding.distance_operand} <=> {embedding.query_operand}) + 0"
    exclude_clause = " ".join(filter_clauses)
    predicate = current_embedding_predicate(
        model,
        embedding.column_ref,
        version_parameter=len(args) + 1,
    )

    # ruff: ignore[S608] -- fragments come from closed model mappings
    sql = f"""
    SELECT
        id,
        external_id,
        title,
        description,
        mentor,
        member1,
        member2,
        status,
        date_of_submission,
        created_at,
        updated_at,
        {embedding.distance_operand} <=> {embedding.query_operand} AS distance
    FROM diploma
    WHERE {predicate.sql}
        AND status = 'Одбрана'
        AND {embedding.distance_operand} <=> {embedding.query_operand} < $3
        {exclude_clause}
    ORDER BY {order_by}
    LIMIT $2
    """

    args.extend(predicate.parameters)

    result = await db.fetch(sql, *args)

    return [
        DiplomaSchema(
            id=row["id"],
            external_id=row["external_id"],
            title=row["title"],
            description=row["description"],
            mentor=row["mentor"],
            member1=row["member1"],
            member2=row["member2"],
            status=row["status"],
            date_of_submission=row["date_of_submission"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            distance=row.get("distance", None),
        )
        for row in result
    ]


async def get_diplomas_without_embeddings(
    db: Database,
    model: Model,
) -> list[DiplomaSchema]:
    embedding_column = embedding_column_name(model)
    predicate = dirty_embedding_predicate(model, embedding_column)
    query = f"SELECT * FROM diploma WHERE {predicate.sql} ORDER BY external_id ASC"  # ruff: ignore[S608] -- predicate comes from a closed model mapping
    result = await db.fetch(query, *predicate.parameters)

    return [
        DiplomaSchema(
            id=row["id"],
            external_id=row["external_id"],
            title=row["title"],
            description=row["description"],
            mentor=row["mentor"],
            member1=row["member1"],
            member2=row["member2"],
            status=row["status"],
            date_of_submission=row["date_of_submission"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in result
    ]


async def fetch_diploma_rows_for_fill(
    db: Database,
    model: Model,
) -> list[Record]:
    predicate = dirty_embedding_predicate(model, embedding_column_name(model))
    return await db.fetch(
        f"SELECT id, title, description, embedding_revision FROM diploma WHERE {predicate.sql} ORDER BY external_id ASC",  # ruff: ignore[S608] -- predicate comes from a closed model mapping
        *predicate.parameters,
    )


async def get_defended_committees(db: Database) -> list[Record]:
    query = (
        "SELECT mentor, member1, member2, date_of_submission FROM diploma "
        "WHERE status = 'Одбрана' AND mentor IS NOT NULL"
    )
    return await db.fetch(query)


async def get_defended_external_ids(db: Database) -> list[str]:
    query = "SELECT external_id FROM diploma WHERE status = 'Одбрана' ORDER BY external_id ASC"
    result = await db.fetch(query)
    return [row["external_id"] for row in result]


async def get_backtest_population(
    db: Database,
    model: Model,
) -> list[Record]:
    embedding_column = embedding_column_name(model)
    predicate = current_embedding_predicate(
        model,
        embedding_column,
        version_parameter=1,
    )
    # ruff: ignore[S608] -- predicate comes from a closed model mapping
    query = f"""
    SELECT external_id, title, mentor, member1, member2, date_of_submission
    FROM diploma
    WHERE status = 'Одбрана'
        AND {predicate.sql}
        AND mentor IS NOT NULL
        AND member1 IS NOT NULL
        AND member2 IS NOT NULL
    ORDER BY external_id ASC
    """
    return await db.fetch(query, *predicate.parameters)
