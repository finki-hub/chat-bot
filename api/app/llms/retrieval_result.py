from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict


class RetrievalSourceLinkPayload(TypedDict):
    label: str
    url: str


class RetrievalSourcePayload(TypedDict):
    id: str
    kind: Literal["chunk", "faq"]
    title: str
    chunk_index: NotRequired[int]
    links: NotRequired[list[RetrievalSourceLinkPayload]]
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
    chunk_index: int | None = None
    links: tuple[RetrievalSourceLink, ...] = ()
    section: str | None = None
    snippet: str = ""

    def as_payload(self) -> RetrievalSourcePayload:
        payload: RetrievalSourcePayload = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
        }
        if self.chunk_index is not None:
            payload["chunk_index"] = self.chunk_index
        if self.links:
            payload["links"] = [link.as_payload() for link in self.links]
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
