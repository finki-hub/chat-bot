"""Compute temporal staff groups and store them in professor_group.

"Staff groups by time": for each fixed time window, detect the cohorts of professors who
repeatedly worked together — community detection (connected components over the >=min-weight
co-occurrence graph) per window. Two sources:
  defense  -> committee co-occurrence (diploma mentor+members, by date_of_submission)
  coauthor -> paper co-authorship       (professor_document canonical_authors, by year)
Groups shift as the window moves, which is the "buddies change over time" view.

    cd api && uv run python ../scripts/compute_professor_groups.py --window-years 3 --min-weight 2
"""

# ruff: noqa: INP001 - standalone offline script; scripts/ is not an importable package

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

if (Path.cwd() / "app").is_dir() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from app.data.connection import Database
from app.data.professor_groups import replace_professor_groups
from app.recommenders.groups import cooccurrence_edges, detect_groups
from app.utils.settings import Settings


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
    """Non-overlapping [start, start+size-1] windows spanning the data, anchored at min year."""
    lo, hi = min(years), max(years)
    windows: list[tuple[int, int]] = []
    start = lo
    while start <= hi:
        windows.append((start, min(start + size - 1, hi)))
        start += size
    return windows


def _groups_for_source(
    records: list[tuple[list[str], int]],
    window_years: int,
    min_weight: int,
) -> list[tuple[int, int, int, list[str], int]]:
    years = [year for _, year in records]
    out: list[tuple[int, int, int, list[str], int]] = []
    for window_start, window_end in _windows(years, window_years):
        in_window = [
            people for people, year in records if window_start <= year <= window_end
        ]
        edges = cooccurrence_edges(in_window)
        for group_index, members in enumerate(detect_groups(edges, min_weight)):
            out.append((window_start, window_end, group_index, members, min_weight))
    return out


async def _run(ns: argparse.Namespace) -> int:
    sources = ["defense", "coauthor"] if ns.source == "both" else [ns.source]
    settings = Settings()
    db = Database(dsn=settings.DATABASE_URL)
    await db.init()
    try:
        for source in sources:
            records = (
                await _defense_records(db)
                if source == "defense"
                else await _coauthor_records(db)
            )
            groups = _groups_for_source(records, ns.window_years, ns.min_weight)
            await replace_professor_groups(db, source, groups)
            print(
                f"[{source}] records={len(records)} "
                f"window_years={ns.window_years} min_weight={ns.min_weight} "
                f"-> {len(groups)} groups",
            )
            for window_start, window_end, group_index, members, _ in groups:
                if group_index == 0:  # show the largest group per window
                    preview = ", ".join(members[:6]) + (
                        "  …" if len(members) > 6 else ""
                    )
                    print(
                        f"   {window_start}-{window_end} biggest ({len(members)}): {preview}",
                    )
    finally:
        await db.disconnect()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute temporal staff groups.")
    parser.add_argument("--window-years", type=int, default=3)
    parser.add_argument("--min-weight", type=int, default=2)
    parser.add_argument(
        "--source",
        choices=("defense", "coauthor", "both"),
        default="both",
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
