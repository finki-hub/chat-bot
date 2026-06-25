"""Load the professor publication corpus into professor_document, then embed it.

The corpus is a JSON array of papers (union of OpenAlex + UKIM-repo, deduped, with canonical
FINKI author names) — built offline, not scraped live, so this is a script rather than an
endpoint. Each record: {doi, title, year, abstract, topics[], sources[], canonicalAuthors[]}.
external_id = the DOI when present, else a sha256 of the normalized title (the corpus is
already deduped on that identity).

    cd api && uv run python ../scripts/load_professor_documents.py --json /path/to/dataset_B_union.json
"""

# ruff: noqa: INP001 - standalone offline script; scripts/ is not an importable package

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

if (Path.cwd() / "app").is_dir() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.data.connection import Database
from app.data.professor_documents import upsert_professor_document
from app.llms.embeddings import stream_fill_professor_document_embeddings
from app.utils.http_client import close_http_client, init_http_client
from app.utils.settings import Settings

_WS = re.compile(r"[^a-z0-9]+")


def _external_id(paper: dict) -> str:
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return doi
    norm = _WS.sub(" ", (paper.get("title") or "").lower()).strip()
    return "T:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()


async def _run(papers: list[dict]) -> int:
    settings = Settings()
    init_http_client()
    db = Database(dsn=settings.DATABASE_URL)
    await db.init()
    try:
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
        print(f"UPSERT: {upserted} papers")

        resp = await stream_fill_professor_document_embeddings(
            db,
            DEFAULT_EMBEDDINGS_MODEL,
        )
        ok = err = total = 0
        async for chunk in resp.body_iterator:
            text = chunk if isinstance(chunk, str) else bytes(chunk).decode("utf-8")
            line = text.strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[len("data:") :].strip())
            total = ev.get("total", total)
            if ev.get("status") == "ok":
                ok += 1
            else:
                err += 1
            done = ok + err
            if done % 500 == 0 or done == total:
                print(f"  fill {done}/{total}  ok={ok} err={err}")
        print(f"FILL DONE: ok={ok} err={err} total={total}")

        embedded = await db.fetchval(
            "SELECT COUNT(*) FROM professor_document WHERE embedding_bge_m3 IS NOT NULL",
        )
        print(f"VERIFY: embedded papers = {embedded}")
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
    return asyncio.run(_run(papers))


if __name__ == "__main__":
    raise SystemExit(main())
