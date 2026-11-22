import json
from typing import Literal

from app.data.connection import Database
from app.data.professor_groups import replace_professor_groups
from app.recommenders.groups import cooccurrence_edges, detect_groups

type GroupSource = Literal["defense", "coauthor"]
type GroupRow = tuple[int, int, int, list[str], int]

DEFAULT_WINDOW_YEARS = 3
DEFAULT_MIN_WEIGHT = 2


async def _defense_records(db: Database) -> list[tuple[list[str], int]]:
    rows = await db.fetch(
        "SELECT mentor, member1, member2, date_of_submission FROM diploma "
        "WHERE status = 'Одбрана' AND date_of_submission IS NOT NULL",
    )
    out: list[tuple[list[str], int]] = []
    for row in rows:
        people = [p for p in (row["mentor"], row["member1"], row["member2"]) if p]
        out.append((people, row["date_of_submission"].year))
    return out


async def _coauthor_records(db: Database) -> list[tuple[list[str], int]]:
    rows = await db.fetch(
        "SELECT canonical_authors, year FROM professor_document WHERE year IS NOT NULL",
    )
    out: list[tuple[list[str], int]] = []
    for row in rows:
        authors = row["canonical_authors"]
        if isinstance(authors, str):
            authors = json.loads(authors)
        out.append((list(authors or []), int(row["year"])))
    return out


def _windows(years: list[int], size: int) -> list[tuple[int, int]]:
    lo, hi = min(years), max(years)
    windows: list[tuple[int, int]] = []
    start = lo
    while start <= hi:
        windows.append((start, min(start + size - 1, hi)))
        start += size
    return windows


def groups_for_records(
    records: list[tuple[list[str], int]],
    window_years: int,
    min_weight: int,
) -> list[GroupRow]:
    if not records:
        return []
    years = [year for _, year in records]
    out: list[GroupRow] = []
    for window_start, window_end in _windows(years, window_years):
        in_window = [
            people for people, year in records if window_start <= year <= window_end
        ]
        edges = cooccurrence_edges(in_window)
        for group_index, members in enumerate(detect_groups(edges, min_weight)):
            out.append((window_start, window_end, group_index, members, min_weight))
    return out


async def refresh_professor_groups(
    db: Database,
    source: GroupSource,
    *,
    window_years: int = DEFAULT_WINDOW_YEARS,
    min_weight: int = DEFAULT_MIN_WEIGHT,
) -> tuple[int, list[GroupRow]]:
    """Recompute one source's groups and return its record count and groups."""
    records = (
        await _defense_records(db)
        if source == "defense"
        else await _coauthor_records(db)
    )
    groups = groups_for_records(records, window_years, min_weight)
    await replace_professor_groups(db, source, groups)
    return len(records), groups
