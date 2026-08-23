from time import perf_counter

import pytest

from app.llms.chunking import chunk_markdown


def _sections_and_content(markdown: str) -> list[tuple[str | None, str]]:
    return [(chunk.section, chunk.content) for chunk in chunk_markdown(markdown)]


def test_unclosed_comment_markers_complete_within_conservative_bound() -> None:
    start = perf_counter()

    chunk_markdown("<!--" * 25_000)

    assert perf_counter() - start < 2.0


def test_comment_removal_preserves_surrounding_markdown() -> None:
    assert _sections_and_content("before<!-- hidden\ntext -->after") == [
        (None, "beforeafter"),
    ]


def test_unclosed_comment_is_preserved() -> None:
    assert _sections_and_content("before<!-- unfinished") == [
        (None, "before<!-- unfinished"),
    ]


def test_heading_with_long_whitespace_prefix_is_parsed() -> None:
    title = " " * 100_000 + "Title"

    assert _sections_and_content(f"#{title}\nbody") == [
        ("Title", "body"),
    ]


def test_heading_label_is_trimmed() -> None:
    assert _sections_and_content("# Title   \nbody") == [("Title", "body")]


def test_blank_heading_is_preserved_as_content() -> None:
    assert _sections_and_content("#   \nbody") == [(None, "#   \nbody")]


def test_heading_marker_without_title_is_preserved_as_content() -> None:
    assert _sections_and_content("#\nЧлен 1\nbody") == [
        (None, "#\nЧлен 1\nbody"),
    ]


def test_member_heading_with_long_number_is_parsed() -> None:
    member_number = "0" * 100_000

    assert _sections_and_content(f"# Член {member_number}\nbody") == [
        (f"Член {member_number}", "body"),
    ]


@pytest.mark.parametrize("level", range(1, 7))
def test_member_heading_is_parsed_at_every_supported_level(level: int) -> None:
    assert _sections_and_content(f"{'#' * level} Член 12\nbody") == [
        ("Член 12", "body"),
    ]


def test_member_heading_label_is_trimmed() -> None:
    assert _sections_and_content("# Член 12   \nbody") == [("Член 12", "body")]


def test_member_document_preserves_preamble_and_skips_empty_sections() -> None:
    assert _sections_and_content("intro\n# Член 1\n# Член 2\nbody") == [
        ("Преамбула", "intro"),
        ("Член 2", "body"),
    ]
