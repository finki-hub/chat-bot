"""
Member-aware Markdown chunking for source-of-truth documents.

Sizes are in characters, calibrated to multilingual-e5-large (~4.36 chars/token median
on Macedonian Cyrillic, ~3.9 on digit-dense text). The 1650-char hard cap is ~415-420
tokens worst case, leaving headroom for the "Наслов:/Содржина:" wrapper and the e5
"passage:" prefix under e5's 512-token window (also the reranker's window).
"""

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

TARGET_CHARS = 1300
HARD_CHARS = 1650
OVERLAP_CHARS = 150

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MEMBER_HEAD_RE = re.compile(r"^#{1,6}\s*(Член\s+\d+.*)$", re.MULTILINE)
_MEMBER_SPLIT_RE = re.compile(r"(?=^#{1,6}\s*Член\s+\d+)", re.MULTILINE)
_HEADING_SPLIT_RE = re.compile(r"(?=^#{1,6}\s+\S)", re.MULTILINE)
_HEAD_RE = re.compile(r"^#{1,6}\s+(.*)$")

PREAMBLE_LABEL = "Преамбула"


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    section: str | None


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        separators=_SEPARATORS,
        keep_separator=True,
    )


def _split_members(md: str) -> list[tuple[str | None, str]]:
    parts = _MEMBER_SPLIT_RE.split(md)
    units: list[tuple[str | None, str]] = []
    preamble = parts[0].strip()
    if preamble:
        units.append((PREAMBLE_LABEL, preamble))
    for part in parts[1:]:
        lines = part.splitlines()
        header = lines[0].lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        # No header fall-back: an empty body makes chunk_markdown skip this member
        # instead of emitting a chunk whose only text is the label (kept in `section`).
        units.append((header, body))
    return units


def _split_headings(md: str) -> list[tuple[str | None, str]]:
    parts = [p for p in _HEADING_SPLIT_RE.split(md) if p.strip()]
    if not parts:
        return [(None, md)]
    units: list[tuple[str | None, str]] = []
    for part in parts:
        lines = part.splitlines()
        head_match = _HEAD_RE.match(lines[0])
        if head_match:
            section: str | None = head_match.group(1).strip()
            body = "\n".join(lines[1:]).strip()
        else:
            section = None
            body = part.strip()
        units.append((section, body))
    return units


def chunk_markdown(markdown: str) -> list[Chunk]:
    md = _COMMENT_RE.sub("", markdown).strip()
    if not md:
        return []

    units = _split_members(md) if _MEMBER_HEAD_RE.search(md) else _split_headings(md)

    splitter = _splitter()
    chunks: list[Chunk] = []
    index = 0
    for section, raw_body in units:
        body = raw_body.strip()
        if not body:
            continue
        pieces = [body] if len(body) <= HARD_CHARS else splitter.split_text(body)
        for raw_piece in pieces:
            piece = raw_piece.strip()
            if piece:
                chunks.append(Chunk(index=index, content=piece, section=section))
                index += 1
    return chunks
