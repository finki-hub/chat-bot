import hashlib
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import anyio
import pytest
from fastapi import Response, status
from pydantic import ValidationError

from app.api import documents as documents_api
from app.data.connection import Database
from app.schemas import document_sources
from app.schemas.document_sources import resolve_document_source_url
from app.schemas.documents import DocumentSchema, IngestDocumentSchema


def test_source_file_derives_encoded_raw_document_url() -> None:
    url = resolve_document_source_url(None, "акт.pdf")

    assert str(url) == (
        "https://raw.githubusercontent.com/finki-hub/documents/main/raw/"
        "%D0%B0%D0%BA%D1%82.pdf"
    )


def test_explicit_source_url_takes_precedence() -> None:
    url = resolve_document_source_url(
        "https://www.finki.ukim.mk/documents/akt.pdf",
        "акт.pdf",
    )

    assert str(url) == "https://www.finki.ukim.mk/documents/akt.pdf"


def test_explicit_source_url_ignores_malformed_source_file() -> None:
    url = resolve_document_source_url(
        "https://www.finki.ukim.mk/documents/akt.pdf",
        "a" * 3_000,
    )

    assert str(url) == "https://www.finki.ukim.mk/documents/akt.pdf"


def test_invalid_legacy_source_url_falls_back_to_source_file() -> None:
    url = resolve_document_source_url("javascript:alert(1)", "akt.pdf")

    assert str(url) == (
        "https://raw.githubusercontent.com/finki-hub/documents/main/raw/akt.pdf"
    )


def test_invalid_legacy_source_url_without_source_file_is_omitted() -> None:
    assert resolve_document_source_url("javascript:alert(1)", None) is None


def test_authority_url_is_primary_and_raw_sources_follow() -> None:
    urls = document_sources.resolve_document_source_urls(
        authority_url="https://www.finki.ukim.mk/documents/rulebook",
        source_url=(
            "https://raw.githubusercontent.com/finki-hub/documents/main/raw/"
            "rulebook.pdf"
        ),
        source_files=("rulebook.pdf", "amendment.pdf"),
    )

    assert tuple(map(str, urls)) == (
        "https://www.finki.ukim.mk/documents/rulebook",
        "https://raw.githubusercontent.com/finki-hub/documents/main/raw/rulebook.pdf",
        "https://raw.githubusercontent.com/finki-hub/documents/main/raw/amendment.pdf",
    )


def test_invalid_authority_url_falls_back_to_legacy_source_url() -> None:
    urls = document_sources.resolve_document_source_urls(
        authority_url="javascript:alert(1)",
        source_url="https://www.finki.ukim.mk/documents/rulebook.pdf",
        source_files=(),
    )

    assert tuple(map(str, urls)) == (
        "https://www.finki.ukim.mk/documents/rulebook.pdf",
    )


@pytest.mark.parametrize(
    "source_file",
    [
        "a" * 3_000,
        ".",
        "..",
        "../README.md",
        "directory/document.pdf",
        "directory\\document.pdf",
        "document\n.pdf",
    ],
)
def test_malformed_source_file_is_omitted(source_file: str) -> None:
    assert resolve_document_source_url("javascript:alert(1)", source_file) is None


@pytest.mark.parametrize(
    "source_url",
    [
        "javascript:alert(1)",
        "http://www.finki.ukim.mk/document.pdf",
        "https://user:password@example.com/document.pdf",
        "https://www.finki.ukim.mk/document.pdf?token=secret",
        "https://www.finki.ukim.mk/document.pdf#section",
    ],
)
@pytest.mark.parametrize("metadata_key", ["authority_url", "source_url"])
def test_ingest_rejects_non_public_source_url(
    source_url: str,
    metadata_key: str,
) -> None:
    with pytest.raises(ValidationError):
        _ = IngestDocumentSchema(
            name="document",
            title="Document",
            content="# Document",
            metadata={metadata_key: source_url},
        )


def test_ingest_normalizes_public_source_url() -> None:
    payload = IngestDocumentSchema(
        name="document",
        title="Document",
        content="# Document",
        metadata={
            "authority_url": "https://WWW.FINKI.UKIM.MK",
            "source_file": "document.pdf",
            "source_url": "https://www.finki.ukim.mk/document.pdf",
        },
    )

    assert payload.metadata == {
        "authority_url": "https://www.finki.ukim.mk/",
        "source_file": "document.pdf",
        "source_url": "https://www.finki.ukim.mk/document.pdf",
    }


def test_unchanged_ingest_updates_metadata_without_rechunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "# Document"
    existing = DocumentSchema(
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
        metadata={
            "authority_url": "https://www.finki.ukim.mk/documents/document",
            "source_file": "document.pdf",
        },
    )

    async def get_existing(*_args: object) -> DocumentSchema:
        return existing

    async def update_metadata(
        _db: object,
        update: IngestDocumentSchema,
        chunk_count: int | None,
    ) -> DocumentSchema:
        assert chunk_count == 2
        return existing.model_copy(update={"metadata": update.metadata})

    monkeypatch.setattr(documents_api, "get_document_by_name_query", get_existing)
    monkeypatch.setattr(documents_api, "update_document_metadata", update_metadata)
    monkeypatch.setattr(
        documents_api,
        "chunk_markdown",
        lambda _content: pytest.fail("unchanged content must not be rechunked"),
    )
    response = Response()

    async def run() -> None:
        result = await documents_api.ingest_document(
            payload,
            response,
            db=cast("Database", object()),
        )
        assert result.metadata == payload.metadata
        assert response.status_code == status.HTTP_200_OK

    anyio.run(run)
