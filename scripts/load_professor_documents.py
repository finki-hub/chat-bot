"""cd api && uv run python ../scripts/load_professor_documents.py --json /path/to/dataset_B_union.json"""

# ruff: noqa: INP001 - standalone offline script; scripts/ is not an importable package

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

if (Path.cwd() / "app").is_dir() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.data.connection import Database
from app.data.embedding_lifecycle import lifecycle_counts
from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.data.professor_documents import upsert_professor_document
from app.llms.embeddings import stream_fill_professor_document_embeddings
from app.utils.http_client import close_http_client, init_http_client
from app.utils.settings import Settings

_WS = re.compile(r"\s+")


def _external_id(paper: dict) -> str:
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return doi
    norm = _WS.sub(" ", (paper.get("title") or "").lower()).strip()
    return "T:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _sse_data_lines(chunk: str | bytes | memoryview[int]) -> Iterator[str]:
    """Yield the JSON payload from each server-sent event line."""
    text = chunk if isinstance(chunk, str) else bytes(chunk).decode("utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("data:"):
            yield line[len("data:") :].strip()


async def _upsert_papers(db: Database, papers: list[dict]) -> int:
    """Persist papers with titles and return the number of upserts."""
    upserted = 0
    for paper in papers:
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        await upsert_professor_document(
            db,
            external_id=_external_id(paper),
            title=title,
            abstract=(paper.get("abstract") or "").strip() or None,
            year=paper.get("year"),
            topics=paper.get("topics") or [],
            canonical_authors=paper.get("canonicalAuthors") or [],
            sources=paper.get("sources") or [],
        )
        upserted += 1
    return upserted


async def _fill_embeddings(db: Database) -> None:
    """Fill embeddings and report progress from the server-sent event stream."""
    response = await stream_fill_professor_document_embeddings(
        db,
        DEFAULT_EMBEDDINGS_MODEL,
    )
    ok = err = total = 0
    async for chunk in response.body_iterator:
        for data in _sse_data_lines(chunk):
            event = json.loads(data)
            total = event.get("total", total)
            if event.get("status") == "ok":
                ok += 1
            else:
                err += 1
            done = ok + err
            if done % 500 == 0 or done == total:
                print(f"  fill {done}/{total}  ok={ok} err={err}")
    print(f"FILL DONE: ok={ok} err={err} total={total}")


async def _report_lifecycle_counts(db: Database) -> None:
    """Print the current and dirty professor-document counts."""
    professor_document_count = next(
        count
        for count in await lifecycle_counts(db)
        if count.corpus is EmbeddingCorpus.PROFESSOR_DOCUMENT
    )
    print(
        "VERIFY: current papers = "
        f"{professor_document_count.ready} dirty papers = "
        f"{professor_document_count.dirty}",
    )


async def run(papers: list[dict]) -> int:
    settings = Settings()
    init_http_client()
    db = Database(dsn=settings.DATABASE_URL)
    await db.init()
    try:
        print(f"UPSERT: {await _upsert_papers(db, papers)} papers")
        await _fill_embeddings(db)
        await _report_lifecycle_counts(db)
    finally:
        await db.disconnect()
        await close_http_client()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load + embed the professor paper corpus.",
    )
    parser.add_argument("--json", required=True, help="path to the union papers JSON")
    ns = parser.parse_args()
    json_path = Path(ns.json).resolve()
    if not json_path.is_file():
        parser.error(f"--json must point to an existing file: {json_path}")
    papers = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"loaded {len(papers)} papers from {json_path}")
    return asyncio.run(run(papers))


if __name__ == "__main__":
    raise SystemExit(main())
