"""cd api && uv run python ../scripts/compute_professor_groups.py --window-years 3 --min-weight 2"""

# ruff: noqa: INP001 - standalone offline script; scripts/ is not an importable package

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if (Path.cwd() / "app").is_dir() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from app.data.connection import Database
from app.services.professor_group_refresh import (
    DEFAULT_MIN_WEIGHT,
    DEFAULT_WINDOW_YEARS,
    GroupSource,
    refresh_professor_groups,
)
from app.utils.settings import Settings


async def _run(ns: argparse.Namespace) -> int:
    sources: list[GroupSource] = (
        ["defense", "coauthor"] if ns.source == "both" else [ns.source]
    )
    settings = Settings()
    db = Database(dsn=settings.DATABASE_URL)
    await db.init()
    try:
        for source in sources:
            records, groups = await refresh_professor_groups(
                db,
                source,
                window_years=ns.window_years,
                min_weight=ns.min_weight,
            )
            print(
                f"[{source}] records={records} "
                f"window_years={ns.window_years} min_weight={ns.min_weight} "
                f"-> {len(groups)} groups",
            )
            for window_start, window_end, group_index, members, _ in groups:
                if group_index == 0:
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
    parser.add_argument("--window-years", type=int, default=DEFAULT_WINDOW_YEARS)
    parser.add_argument("--min-weight", type=int, default=DEFAULT_MIN_WEIGHT)
    parser.add_argument(
        "--source",
        choices=("defense", "coauthor", "both"),
        default="both",
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
