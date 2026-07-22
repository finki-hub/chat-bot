# mypy: disable-error-code="arg-type"

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
from app.schemas.diplomas import DiplomaSchema
from app.utils.database import embedding_to_pgvector


async def upsert_diploma(
    db: Database,
    external_id: str,
    title: str,
    description: str,
    mentor: str,
    member1: str | None,
    member2: str | None,
    status: str,
    date_of_submission: object | None,
) -> DiplomaSchema | None:
    query = """
    INSERT INTO diploma (
        external_id, title, description, mentor, member1, member2, status, date_of_submission
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    ON CONFLICT (external_id) DO UPDATE SET
        title = EXCLUDED.title,
        description = EXCLUDED.description,
        mentor = EXCLUDED.mentor,
        member1 = EXCLUDED.member1,
        member2 = EXCLUDED.member2,
        status = EXCLUDED.status,
        date_of_submission = EXCLUDED.date_of_submission,
        updated_at = NOW()
    RETURNING *
    """
    result = await db.fetchrow(
        query,
        external_id,
        title,
        description,
        mentor,
        member1,
        member2,
        status,
        date_of_submission,
    )

    if result is None:
        return None

    return DiplomaSchema(
        id=result["id"],
        external_id=result["external_id"],
        title=result["title"],
        description=result["description"],
        mentor=result["mentor"],
        member1=result["member1"],
        member2=result["member2"],
        status=result["status"],
        date_of_submission=result["date_of_submission"],
        created_at=result["created_at"],
        updated_at=result["updated_at"],
    )


async def get_closest_diplomas(
    db: Database,
    embedded_query: list[float],
    model: Model,
    limit: int = 8,
    threshold: float | None = None,
    *,
    exclude_external_id: str | None = None,
) -> list[DiplomaSchema]:
    embedding = embedding_vector_sql(model, embedded_query)
    version_parameter = 5 if exclude_external_id is not None else 4
    predicate = current_embedding_predicate(
        model,
        embedding.column_ref,
        version_parameter=version_parameter,
    )

    if threshold is None:
        threshold = MODEL_DISTANCE_THRESHOLDS.get(model, 0.5)

    exclude_clause = ""
    if exclude_external_id is not None:
        exclude_clause = "AND external_id <> $4"

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
    ORDER BY distance
    LIMIT $2
    """  # noqa: S608

    args: list[object] = [
        embedding_to_pgvector(embedded_query),
        limit,
        threshold,
    ]
    if exclude_external_id is not None:
        args.append(exclude_external_id)
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
    query = f"SELECT * FROM diploma WHERE {predicate.sql} ORDER BY external_id ASC"  # noqa: S608
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
        f"SELECT id, title, description, embedding_revision FROM diploma WHERE {predicate.sql} ORDER BY external_id ASC",  # noqa: S608
        *predicate.parameters,
    )


async def get_defended_committees(db: Database) -> list[Record]:
    query = (
        "SELECT mentor, member1, member2 FROM diploma "
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
    query = f"""
    SELECT external_id, title, mentor, member1, member2
    FROM diploma
    WHERE status = 'Одбрана'
        AND {predicate.sql}
        AND mentor IS NOT NULL
        AND member1 IS NOT NULL
        AND member2 IS NOT NULL
    ORDER BY external_id ASC
    """  # noqa: S608
    return await db.fetch(query, *predicate.parameters)
