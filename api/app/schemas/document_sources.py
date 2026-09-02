from collections.abc import Sequence
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


def _parse_optional_source_url(value: str | None) -> HttpUrl | None:
    if value is None:
        return None
    try:
        return parse_document_source_url(value)
    except InvalidDocumentSourceUrlError, ValidationError:
        return None


def _raw_source_url(source_file: str | None) -> HttpUrl | None:
    if (
        not source_file
        or source_file in {".", ".."}
        or "/" in source_file
        or "\\" in source_file
        or any(
            ord(character) < 32 or ord(character) == 127 for character in source_file
        )
    ):
        return None
    return _parse_optional_source_url(
        f"{DOCUMENTS_RAW_BASE_URL}{quote(source_file, safe='')}",
    )


def resolve_document_source_url(
    source_url: str | None,
    source_file: str | None,
) -> HttpUrl | None:
    explicit_url = _parse_optional_source_url(source_url)
    if explicit_url is not None:
        return explicit_url
    return _raw_source_url(source_file)


def resolve_document_source_urls(
    authority_url: str | None,
    source_url: str | None,
    source_files: Sequence[str],
) -> tuple[HttpUrl, ...]:
    candidates = (
        _parse_optional_source_url(authority_url),
        _parse_optional_source_url(source_url),
        *(_raw_source_url(source_file) for source_file in source_files),
    )
    resolved: list[HttpUrl] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None or str(candidate) in seen:
            continue
        resolved.append(candidate)
        seen.add(str(candidate))
    return tuple(resolved)
