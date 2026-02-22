from pydantic import BaseModel, HttpUrl

from app.data.connection import Database
from app.schemas.links import CreateLinkSchema, LinkSchema

_LINK_ALLOWED_COLUMNS: frozenset[str] = frozenset(
    {"name", "url", "description", "user_id"},
)


async def get_links_query(db: Database) -> list[LinkSchema]:
    query = "SELECT * FROM link ORDER BY name ASC"
    result = await db.fetch(query)

    return [LinkSchema(**row) for row in result]


async def get_link_names_query(db: Database) -> list[str]:
    query = "SELECT name FROM link ORDER BY name ASC"
    result = await db.fetch(query)

    return [str(row["name"]) for row in result]


async def get_link_by_name_query(db: Database, name: str) -> LinkSchema | None:
    query = "SELECT * FROM link WHERE name = $1"
    result = await db.fetchrow(query, name)

    if not result:
        return None

    return LinkSchema(**result)


async def create_link_query(
    db: Database,
    link: CreateLinkSchema,
) -> LinkSchema | None:
    query = """
    INSERT INTO link (name, url, description, user_id)
    VALUES ($1, $2, $3, $4)
    RETURNING *
    """
    result = await db.fetchrow(
        query,
        link.name,
        link.url,
        link.description,
        link.user_id,
    )

    if not result:
        return None

    return LinkSchema(**result)


async def update_link_query(
    db: Database,
    name: str,
    updates: dict,
) -> LinkSchema | None:
    if not updates:
        raise ValueError("No fields to update")

    invalid = set(updates) - _LINK_ALLOWED_COLUMNS

    if invalid:
        raise ValueError(f"Invalid column(s) for update: {invalid}")

    for key, value in tuple(updates.items()):
        if isinstance(value, BaseModel):
            updates[key] = value.model_dump()
        elif isinstance(value, HttpUrl):
            updates[key] = str(value)

    set_clauses = []
    values = []
    param_index = 1

    for key, value in updates.items():
        set_clauses.append(f"{key} = ${param_index}")
        values.append(value)
        param_index += 1

    values.append(name)
    where_placeholder = f"${param_index}"

    query = f"""
    UPDATE link
    SET {", ".join(set_clauses)}
    WHERE name = {where_placeholder}
    RETURNING *
    """  # noqa: S608

    result = await db.fetchrow(query, *values)
    if not result:
        return None

    return LinkSchema(**result)


async def delete_link_query(db: Database, name: str) -> None:
    query = "DELETE FROM link WHERE name = $1"
    await db.execute(query, name)


async def get_nth_link_query(db: Database, n: int) -> LinkSchema | None:
    query = "SELECT * FROM link ORDER BY name ASC LIMIT 1 OFFSET $1"
    result = await db.fetchrow(query, n)

    if not result:
        return None

    return LinkSchema(**result)
