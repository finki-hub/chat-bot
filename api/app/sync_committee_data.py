"""Scheduled entrypoint that syncs defended theses and refreshes defense groups."""

# ruff: file-ignore[T201] - CLI output is the metadata-only sync report

from __future__ import annotations

import argparse
import asyncio
import sys

from app.data.connection import Database
from app.services.diploma_sync import DiplomaSyncError, sync_defended_diplomas
from app.services.professor_group_refresh import refresh_professor_groups
from app.utils.http_client import close_http_client, init_http_client
from app.utils.settings import Settings


async def run(*, accept_fewer: bool) -> int:
    init_http_client()
    db = Database(dsn=Settings().DATABASE_URL, min_size=1, max_size=2)
    try:
        await db.init()
        result = await sync_defended_diplomas(db, accept_fewer=accept_fewer)
        records, groups = await refresh_professor_groups(db, "defense")
    except DiplomaSyncError as exc:
        print(f"committee sync failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await db.disconnect()
        await close_http_client()

    print(
        f"fetched={result.fetched} defended={result.defended} "
        f"inserted={result.inserted} updated={result.updated} "
        f"group_records={records} groups={len(groups)}",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-fewer",
        action="store_true",
        help="allow an upstream list far shorter than the stored defenses",
    )
    namespace = parser.parse_args(argv)
    return asyncio.run(run(accept_fewer=namespace.accept_fewer))


if __name__ == "__main__":
    raise SystemExit(main())
