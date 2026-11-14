import re
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict


class RetrievalSourceLinkPayload(TypedDict):
    label: str
    url: str


class RetrievalSourcePayload(TypedDict):
    id: str
    kind: Literal["chunk", "faq"]
    title: str
    authority_url: NotRequired[str]
    chunk_index: NotRequired[int]
    current_status: NotRequired[str]
    document_date: NotRequired[str]
    last_verified: NotRequired[str]
    links: NotRequired[list[RetrievalSourceLinkPayload]]
    ref: NotRequired[int]
    section: NotRequired[str]
    snippet: NotRequired[str]


@dataclass(frozen=True, slots=True)
class RetrievalSourceLink:
    label: str
    url: str

    def as_payload(self) -> RetrievalSourceLinkPayload:
        return {"label": self.label, "url": self.url}


@dataclass(frozen=True, slots=True)
class RetrievalSource:
    id: str
    kind: Literal["chunk", "faq"]
    title: str
    authority_url: str | None = None
    chunk_index: int | None = None
    current_status: str | None = None
    document_date: str | None = None
    last_verified: str | None = None
    links: tuple[RetrievalSourceLink, ...] = ()
    section: str | None = None
    snippet: str = ""
    # Model-only guidance: deliberately left out of as_payload.
    assistant_note: str | None = None
    # Citation number the model may write as `[N]` (web answers only).
    ref: int | None = None

    def as_payload(self) -> RetrievalSourcePayload:
        payload: RetrievalSourcePayload = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
        }
        if self.chunk_index is not None:
            payload["chunk_index"] = self.chunk_index
        if self.authority_url is not None:
            payload["authority_url"] = self.authority_url
        if self.current_status is not None:
            payload["current_status"] = self.current_status
        if self.document_date is not None:
            payload["document_date"] = self.document_date
        if self.last_verified is not None:
            payload["last_verified"] = self.last_verified
        if self.links:
            payload["links"] = [link.as_payload() for link in self.links]
        if self.ref is not None:
            payload["ref"] = self.ref
        if self.section:
            payload["section"] = self.section
        if self.snippet:
            payload["snippet"] = self.snippet
        return payload


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    text: str
    sources: tuple[RetrievalSource, ...] = ()

    def sources_payload(self) -> list[RetrievalSourcePayload]:
        return [source.as_payload() for source in self.sources]


def visible_sources(
    scored_sources: list[tuple[RetrievalSource, float | None]],
    *,
    source_score_floor: float,
) -> tuple[RetrievalSource, ...]:
    return tuple(
        source
        for source, score in scored_sources
        if score is not None and score >= source_score_floor
    )


def citation_key(source: RetrievalSource) -> str:
    """One citable unit per FAQ entry or per document, matching how clients group
    source cards: every chunk of a document shares its number."""
    return f"faq:{source.id}" if source.kind == "faq" else f"chunk:{source.title}"


class CitationRefs:
    """Citation numbers for one answer, stable across its searches: a source keeps
    the number it got the first time any search returned it."""

    def __init__(self) -> None:
        self._refs: dict[str, int] = {}

    def assign(self, source: RetrievalSource) -> int:
        return self._refs.setdefault(citation_key(source), len(self._refs) + 1)

    def get(self, source: RetrievalSource) -> int | None:
        return self._refs.get(citation_key(source))


_CITATION_MARKERS = re.compile(r" ?(?:\[\d{1,3}\])+")


def strip_citation_markers(text: str) -> str:
    """Drop `[N]` markers from an earlier answer: each answer numbers its own
    sources from 1, so old markers in the history would point at the wrong ones."""
    return _CITATION_MARKERS.sub("", text)
