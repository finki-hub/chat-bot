from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

TARGET_CHARS = 1300
HARD_CHARS = 1650
OVERLAP_CHARS = 150

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

PREAMBLE_LABEL = "Преамбула"


@dataclass(frozen=True, slots=True)
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


def _without_html_comments(markdown: str) -> str:
    parts: list[str] = []
    cursor = 0
    while True:
        start = markdown.find("<!--", cursor)
        if start == -1:
            parts.append(markdown[cursor:])
            break
        end = markdown.find("-->", start + 4)
        if end == -1:
            parts.append(markdown[cursor:])
            break
        parts.append(markdown[cursor:start])
        cursor = end + 3
    return "".join(parts)


def _heading(line: str, *, require_space: bool) -> str | None:
    marker_length = len(line) - len(line.lstrip("#"))
    if not 1 <= marker_length <= 6:
        return None

    remainder = line[marker_length:]
    if not remainder or (require_space and not remainder[0].isspace()):
        return None
    title = remainder.strip()
    return title or None


def _member_heading(line: str) -> str | None:
    title = _heading(line, require_space=False)
    if title is None or not title.startswith("Член"):
        return None

    suffix = title[len("Член") :]
    if not suffix or not suffix[0].isspace():
        return None
    number_and_rest = suffix.lstrip()
    digit_count = 0
    while (
        digit_count < len(number_and_rest) and number_and_rest[digit_count].isdecimal()
    ):
        digit_count += 1
    return title if digit_count > 0 else None


def _split_members(lines: list[str]) -> list[tuple[str | None, str]]:
    units: list[tuple[str | None, str]] = []
    section = PREAMBLE_LABEL
    body_lines: list[str] = []
    found_member = False

    for line in lines:
        header = _member_heading(line)
        if header is None:
            body_lines.append(line)
            continue

        body = "\n".join(body_lines).strip()
        if found_member or body:
            units.append((section, body))
        section = header
        body_lines = []
        found_member = True

    units.append((section, "\n".join(body_lines).strip()))
    return units


def _split_headings(md: str, lines: list[str]) -> list[tuple[str | None, str]]:
    units: list[tuple[str | None, str]] = []
    section: str | None = None
    body_lines: list[str] = []
    found_heading = False

    for line in lines:
        header = _heading(line, require_space=True)
        if header is None:
            body_lines.append(line)
            continue

        body = "\n".join(body_lines).strip()
        if found_heading or body:
            units.append((section, body))
        section = header
        body_lines = []
        found_heading = True

    if not found_heading:
        return [(None, md)]
    units.append((section, "\n".join(body_lines).strip()))
    return units


def chunk_markdown(markdown: str) -> list[Chunk]:
    md = _without_html_comments(markdown).strip()
    if not md:
        return []

    lines = md.splitlines()
    has_member_heading = any(_member_heading(line) for line in lines)
    units = _split_members(lines) if has_member_heading else _split_headings(md, lines)

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
