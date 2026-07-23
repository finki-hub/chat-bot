from typing import Final
from urllib.parse import quote

from pydantic import HttpUrl, ValidationError

DOCUMENTS_RAW_BASE_URL: Final = (
    "https://raw.githubusercontent.com/finki-hub/documents/main/raw/"
)


class InvalidDocumentSourceUrlError(ValueError):
    def __init__(self) -> None:
        super().__init__(
            "source_url must be an HTTPS URL without credentials, query, or fragment",
        )


def parse_document_source_url(value: str) -> HttpUrl:
    url = HttpUrl(value)
    if (
        url.scheme != "https"
        or url.username is not None
        or url.password is not None
        or url.query is not None
        or url.fragment is not None
    ):
        raise InvalidDocumentSourceUrlError
    return url


def resolve_document_source_url(
    source_url: str | None,
    source_file: str | None,
) -> HttpUrl | None:
    fallback = (
        parse_document_source_url(
            f"{DOCUMENTS_RAW_BASE_URL}{quote(source_file, safe='')}",
        )
        if source_file
        else None
    )
    if source_url:
        try:
            return parse_document_source_url(source_url)
        except InvalidDocumentSourceUrlError, ValidationError:
            return fallback
    return fallback
