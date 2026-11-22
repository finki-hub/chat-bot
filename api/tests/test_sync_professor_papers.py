from typing import Any

import httpx
import pytest

from app.sync_professor_papers import (
    Profile,
    StoredPaper,
    _request_json,
    _RetryableRequestError,
    abstract_from_index,
    build_refresh_plan,
    credited_staff,
    title_external_id,
    unsupported_credits,
)

FINKI = "Faculty of Computer Science and Engineering, Skopje"
PROFILES = {
    "A1": Profile("A1", "Ана Долга Презиме"),
    "A2": Profile("A2", "Марко Петров"),
    "A3": Profile("A3", "Елена Мешана", "E. Meshana"),
}
ALIASES = {"Ана Долга": "Ана Долга Презиме"}


def _authorship(
    author: str,
    *,
    affiliation: str = FINKI,
    written: str = "Author",
) -> dict[str, Any]:
    return {
        "author": {"id": f"https://openalex.org/{author}"},
        "raw_affiliation_strings": [affiliation],
        "raw_author_name": written,
    }


def _work(number: int, *authorships: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/W{number}",
        "doi": f"https://doi.org/10.1000/{number}",
        "title": f"Paper {number}",
        "publication_year": 2026,
        "type": "article",
        "is_retracted": False,
        "authorships": list(authorships),
        "abstract_inverted_index": {"Short": [0], "abstract": [1]},
        "topics": [{"display_name": "Machine learning"}],
    } | fields


def test_credit_requires_a_finki_affiliation_on_that_authorship() -> None:
    work = _work(
        1,
        _authorship("A1"),
        _authorship("A2", affiliation="Boston University"),
    )

    assert credited_staff(work, PROFILES) == ("Ана Долга Презиме",)


def test_mixed_profile_counts_only_its_written_name() -> None:
    kept = _work(1, _authorship("A3", written=" e.  meshana "))
    other = _work(2, _authorship("A3", written="Elias Meshana"))

    assert credited_staff(kept, PROFILES) == ("Елена Мешана",)
    assert credited_staff(other, PROFILES) == ()


def test_plan_adds_new_papers_with_doi_key_and_abstract() -> None:
    plan = build_refresh_plan([_work(1, _authorship("A2"))], PROFILES, [], ALIASES)

    assert len(plan.new_papers) == 1
    paper = plan.new_papers[0]
    assert paper.external_id == "10.1000/1"
    assert paper.abstract == "Short abstract"
    assert paper.topics == ("Machine learning",)
    assert paper.sources == ("openalex",)


def test_plan_keeps_existing_credits_and_does_not_duplicate_aliases() -> None:
    stored = [
        StoredPaper(
            "10.1000/1", "Paper 1", ("Ана Долга", "Друг Автор"), ("ukim_repo",)
        ),
    ]
    work = _work(1, _authorship("A1"), _authorship("A2"))

    plan = build_refresh_plan([work], PROFILES, stored, ALIASES)

    assert plan.new_papers == ()
    assert plan.updates[0].added == ("Марко Петров",)
    assert plan.updates[0].canonical_authors == (
        "Ана Долга",
        "Друг Автор",
        "Марко Петров",
    )
    assert plan.updates[0].sources == ("ukim_repo", "openalex")


def test_plan_matches_stored_papers_by_title_and_skips_unchanged() -> None:
    stored = [
        StoredPaper(
            title_external_id("Paper 1"),
            "paper  1",
            ("Марко Петров",),
            ("openalex",),
        ),
    ]

    plan = build_refresh_plan(
        [_work(1, _authorship("A2"))],
        PROFILES,
        stored,
        ALIASES,
    )

    assert plan.new_papers == ()
    assert plan.updates == ()


def test_plan_prefers_published_version_over_preprint() -> None:
    preprint = _work(
        1,
        _authorship("A1"),
        type="preprint",
        doi="https://doi.org/10.1000/pre",
        title="Same title",
    )
    published = _work(
        2,
        _authorship("A2"),
        doi="https://doi.org/10.1000/pub",
        title="Same  title",
    )

    plan = build_refresh_plan([preprint, published], PROFILES, [], ALIASES)

    assert [paper.external_id for paper in plan.new_papers] == ["10.1000/pub"]
    assert plan.new_papers[0].canonical_authors == (
        "Ана Долга Презиме",
        "Марко Петров",
    )


def test_plan_ignores_retracted_and_non_paper_works() -> None:
    works = [
        _work(1, _authorship("A1"), is_retracted=True),
        _work(2, _authorship("A1"), type="dissertation"),
        _work(3, _authorship("A1"), title=" "),
    ]

    plan = build_refresh_plan(works, PROFILES, [], ALIASES)

    assert plan.new_papers == ()


def test_fingerprint_changes_with_the_plan() -> None:
    first = build_refresh_plan([_work(1, _authorship("A1"))], PROFILES, [], ALIASES)
    same = build_refresh_plan([_work(1, _authorship("A1"))], PROFILES, [], ALIASES)
    other = build_refresh_plan([_work(2, _authorship("A1"))], PROFILES, [], ALIASES)

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != other.fingerprint


@pytest.mark.anyio
async def test_request_errors_do_not_expose_the_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed {request.url}", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(_RetryableRequestError) as raised:
            await _request_json(
                client,
                "https://api.openalex.org/works",
                {"api_key": "secret-key"},
                "api.openalex.org",
            )

    assert "secret-key" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_abstract_is_rebuilt_in_position_order() -> None:
    assert abstract_from_index({"world": [1], "hello": [0, 2]}) == "hello world hello"
    assert abstract_from_index(None) is None


def test_unsupported_credits_report_mapped_staff_missing_from_the_work() -> None:
    stored = [
        StoredPaper("10.1000/1", "Paper 1", ("Ана Долга", "Надворешен"), ("openalex",)),
    ]

    findings = unsupported_credits(
        [_work(1, _authorship("A2"))],
        PROFILES,
        stored,
        ALIASES,
    )

    assert [finding["staff"] for finding in findings] == ["Ана Долга Презиме"]
