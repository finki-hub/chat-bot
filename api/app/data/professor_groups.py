# mypy: disable-error-code="arg-type"

import json

from asyncpg import Record

from app.data.connection import Database


async def replace_professor_groups(
    db: Database,
    source: str,
    groups: list[tuple[int, int, int, list[str], int]],
) -> int:
    """Replace ALL groups for a source atomically.

    Groups are recomputed wholesale (community detection is not incremental), so the old
    rows for this source are dropped and the fresh set inserted in one transaction. Each
    tuple is (window_start, window_end, group_index, members, min_weight).
    """
    async with db.transaction() as conn:
        await conn.execute("DELETE FROM professor_group WHERE source = $1", source)
        for window_start, window_end, group_index, members, min_weight in groups:
            await conn.execute(
                "INSERT INTO professor_group "
                "(source, window_start, window_end, group_index, members, size, min_weight) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                source,
                window_start,
                window_end,
                group_index,
                json.dumps(members, ensure_ascii=False),
                len(members),
                min_weight,
            )
    return len(groups)


async def get_professor_groups(
    db: Database,
    *,
    source: str | None = None,
    year: int | None = None,
    professor: str | None = None,
) -> list[Record]:
    """`professor` uses JSONB array containment to find every window that professor's cohort appears in."""
    clauses: list[str] = []
    args: list[object] = []
    if source is not None:
        args.append(source)
        clauses.append(f"source = ${len(args)}")
    if year is not None:
        args.append(year)
        clauses.append(f"window_start <= ${len(args)}")
        args.append(year)
        clauses.append(f"window_end >= ${len(args)}")
    if professor is not None:
        args.append(json.dumps(professor, ensure_ascii=False))
        clauses.append(f"members @> ${len(args)}::jsonb")

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    # S608 suppressed below: the WHERE clauses are static literals; values are all params ($n).
    sql = (
        "SELECT source, window_start, window_end, group_index, members, size, min_weight "  # noqa: S608
        f"FROM professor_group{where} "
        "ORDER BY source, window_start, group_index"
    )
    return await db.fetch(sql, *args)
