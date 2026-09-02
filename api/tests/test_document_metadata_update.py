import hashlib
import json
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import anyio

from app.data.connection import Database
from app.data.documents import update_document_metadata
from app.schemas.documents import DocumentSchema, IngestDocumentSchema


class NoMatchDatabase:
    def __init__(self) -> None:
        self.query: str | None = None
        self.args: tuple[object, ...] = ()

    async def fetchrow(self, query: str, *args: object) -> None:
        self.query = query
        self.args = args


def test_metadata_update_uses_observed_identity_as_compare_and_set_guard() -> None:
    content = "# Document"
    observed = DocumentSchema(
        id=uuid4(),
        name="document",
        title="Document",
        source_hash=hashlib.sha256(content.encode()).hexdigest(),
        metadata={"source_file": "document.pdf"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        chunk_count=2,
    )
    payload = IngestDocumentSchema(
        name="document",
        title="Document",
        content=content,
        metadata={"authority_url": "https://www.finki.ukim.mk/document"},
    )
    db = NoMatchDatabase()

    async def run() -> None:
        result = await update_document_metadata(
            cast("Database", cast("object", db)), observed, payload
        )

        assert result is None

    anyio.run(run)

    assert db.query is not None
    assert "WHERE id = $1 AND source_hash = $2" in db.query
    assert db.args[:2] == (observed.id, observed.source_hash)
    assert isinstance(db.args[2], str)
    assert json.loads(db.args[2]) == payload.metadata
