import pytest
from pydantic import ValidationError

from app.schemas.document_sources import resolve_document_source_url
from app.schemas.documents import IngestDocumentSchema


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
def test_ingest_rejects_non_public_source_url(source_url: str) -> None:
    with pytest.raises(ValidationError):
        _ = IngestDocumentSchema(
            name="document",
            title="Document",
            content="# Document",
            metadata={"source_url": source_url},
        )


def test_ingest_normalizes_public_source_url() -> None:
    payload = IngestDocumentSchema(
        name="document",
        title="Document",
        content="# Document",
        metadata={
            "source_file": "document.pdf",
            "source_url": "https://www.finki.ukim.mk/document.pdf",
        },
    )

    assert payload.metadata == {
        "source_file": "document.pdf",
        "source_url": "https://www.finki.ukim.mk/document.pdf",
    }
