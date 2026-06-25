# mypy: disable-error-code="arg-type"

from asyncpg import Record

from app.data.connection import Database
from app.llms.models import (
    HALFVEC_EMBEDDING_MODELS,
    MODEL_DISTANCE_THRESHOLDS,
    MODEL_EMBEDDINGS_COLUMNS,
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
        {col_expr} <=> {param_expr} AS distance
    FROM diploma
    WHERE {embedding_column} IS NOT NULL
        AND status = 'Одбрана'
        AND {col_expr} <=> {param_expr} < $3
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
    embedding_column = MODEL_EMBEDDINGS_COLUMNS[model]
    query = f"SELECT * FROM diploma WHERE {embedding_column} IS NULL ORDER BY external_id ASC"  # noqa: S608
    result = await db.fetch(query)

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
    model_column: str,
) -> list[Record]:
    return await db.fetch(
        f"SELECT id, title, description FROM diploma WHERE {model_column} IS NULL ORDER BY external_id ASC",  # noqa: S608
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
    embedding_column = MODEL_EMBEDDINGS_COLUMNS[model]
    query = f"""
    SELECT external_id, title, mentor, member1, member2
    FROM diploma
    WHERE status = 'Одбрана'
        AND {embedding_column} IS NOT NULL
        AND mentor IS NOT NULL
        AND member1 IS NOT NULL
        AND member2 IS NOT NULL
    ORDER BY external_id ASC
    """  # noqa: S608
    return await db.fetch(query)
