"""Refresh professor papers from reviewed OpenAlex profiles; a dry run by default."""

# ruff: file-ignore[T201] - CLI output is the metadata-only refresh report

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import httpx
from asyncpg import Record
from asyncpg.pool import PoolConnectionProxy

from app.data.connection import Database
from app.utils.settings import Settings

type Work = Mapping[str, Any]

_OPENALEX_WORKS_URL: Final = "https://api.openalex.org/works"
_WORK_FIELDS: Final = (
    "id,doi,title,publication_year,type,is_retracted,authorships,"
    "abstract_inverted_index,topics"
)
_AUTHOR_BATCH: Final = 40
_PAGE_SIZE: Final = 200
_MAX_ATTEMPTS: Final = 5
_RETRY_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
_REFRESH_LOCK_KEY: Final = 7_241_806_134
_OPENALEX_SOURCE: Final = "openalex"
_PAPER_TYPES: Final = frozenset(
    {
        "article",
        "book",
        "book-chapter",
        "conference-paper",
        "data-paper",
        "preprint",
        "review",
    },
)
_FINKI_AFFILIATION: Final = re.compile(
    r"computer science (and|&) engineering|finki|fcse|"
    r"informatics and computer engineering|"
    r"information sciences and computer engineering|"
    r"информатички науки",
    re.IGNORECASE,
)
_WHITESPACE: Final = re.compile(r"\s+")


class PaperRefreshError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Profile:
    openalex_id: str
    staff_name: str
    written_name: str | None = None


@dataclass(frozen=True, slots=True)
class StoredPaper:
    external_id: str
    title: str
    canonical_authors: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NewPaper:
    external_id: str
    title: str
    abstract: str | None
    year: int | None
    topics: tuple[str, ...]
    canonical_authors: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CreditUpdate:
    external_id: str
    title: str
    added: tuple[str, ...]
    canonical_authors: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RefreshPlan:
    new_papers: tuple[NewPaper, ...]
    updates: tuple[CreditUpdate, ...]

    @property
    def fingerprint(self) -> str:
        payload = {
            "new": [
                [paper.external_id, paper.title, list(paper.canonical_authors)]
                for paper in self.new_papers
            ],
            "updates": [
                [update.external_id, list(update.canonical_authors)]
                for update in self.updates
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_doi(value: str | None) -> str:
    doi = (value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/"):
        doi = doi.removeprefix(prefix)
    return doi


def title_external_id(title: str) -> str:
    normalized = _WHITESPACE.sub(" ", title.lower()).strip()
    return "T:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def abstract_from_index(index: Mapping[str, Sequence[int]] | None) -> str | None:
    if not index:
        return None
    positions = sorted(
        (position, word) for word, places in index.items() for position in places
    )
    return " ".join(word for _, word in positions) or None


def _author_id(authorship: Mapping[str, Any]) -> str:
    author = authorship.get("author") or {}
    return str(author.get("id") or "").rsplit("/", 1)[-1]


def _written_key(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip().casefold()


def _same_written_name(left: str | None, right: str) -> bool:
    return left is not None and _written_key(left) == _written_key(right)


def credited_staff(work: Work, profiles: Mapping[str, Profile]) -> tuple[str, ...]:
    """Return staff credited by a FINKI-affiliated authorship on this work."""
    names: set[str] = set()
    for authorship in work.get("authorships") or []:
        profile = profiles.get(_author_id(authorship))
        if profile is None:
            continue
        affiliations = " ".join(authorship.get("raw_affiliation_strings") or [])
        if not _FINKI_AFFILIATION.search(affiliations):
            continue
        if profile.written_name is not None and not _same_written_name(
            authorship.get("raw_author_name"),
            profile.written_name,
        ):
            continue
        names.add(profile.staff_name)
    return tuple(sorted(names))


def _is_paper(work: Work) -> bool:
    return (
        bool((work.get("title") or "").strip())
        and work.get("type") in _PAPER_TYPES
        and not work.get("is_retracted")
    )


def _merge_names(
    stored: Iterable[str],
    credited: Iterable[str],
    aliases: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = list(stored)
    people = {aliases.get(name, name) for name in names}
    added = [name for name in credited if aliases.get(name, name) not in people]
    return tuple(names + added), tuple(added)


def _with_openalex(sources: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*sources, _OPENALEX_SOURCE)))


def _new_paper(work: Work, names: tuple[str, ...]) -> NewPaper:
    title = str(work["title"]).strip()
    return NewPaper(
        external_id=normalize_doi(work.get("doi")) or title_external_id(title),
        title=title,
        abstract=abstract_from_index(work.get("abstract_inverted_index")),
        year=work.get("publication_year"),
        topics=tuple(
            str(topic["display_name"])
            for topic in work.get("topics") or []
            if topic.get("display_name")
        ),
        canonical_authors=names,
        sources=(_OPENALEX_SOURCE,),
    )


def build_refresh_plan(
    works: Iterable[Work],
    profiles: Mapping[str, Profile],
    stored: Iterable[StoredPaper],
    aliases: Mapping[str, str],
) -> RefreshPlan:
    """Plan additive changes; stored papers and credits are never removed."""
    stored_by_id = {paper.external_id.lower(): paper for paper in stored}
    stored_by_title = {
        title_external_id(paper.title): paper for paper in stored_by_id.values()
    }
    credits_for_stored: defaultdict[str, set[str]] = defaultdict(set)
    new_by_title: dict[str, NewPaper] = {}

    ordered = sorted(
        (work for work in works if _is_paper(work)),
        key=lambda work: (work.get("type") == "preprint", str(work.get("id"))),
    )
    for work in ordered:
        names = credited_staff(work, profiles)
        if not names:
            continue
        title_key = title_external_id(str(work["title"]))
        match = stored_by_id.get(normalize_doi(work.get("doi"))) or stored_by_title.get(
            title_key,
        )
        if match is not None:
            credits_for_stored[match.external_id].update(names)
        elif title_key in new_by_title:
            current = new_by_title[title_key]
            merged = tuple(sorted({*current.canonical_authors, *names}))
            new_by_title[title_key] = NewPaper(
                current.external_id,
                current.title,
                current.abstract,
                current.year,
                current.topics,
                merged,
                current.sources,
            )
        else:
            new_by_title[title_key] = _new_paper(work, names)

    updates: list[CreditUpdate] = []
    for external_id, credited in credits_for_stored.items():
        paper = stored_by_id[external_id.lower()]
        canonical, added = _merge_names(
            paper.canonical_authors,
            sorted(credited),
            aliases,
        )
        if added:
            updates.append(
                CreditUpdate(
                    external_id=paper.external_id,
                    title=paper.title,
                    added=added,
                    canonical_authors=canonical,
                    sources=_with_openalex(paper.sources),
                ),
            )

    return RefreshPlan(
        new_papers=tuple(
            sorted(new_by_title.values(), key=lambda paper: paper.external_id),
        ),
        updates=tuple(sorted(updates, key=lambda update: update.external_id)),
    )


def unsupported_credits(
    works: Iterable[Work],
    profiles: Mapping[str, Profile],
    stored: Iterable[StoredPaper],
    aliases: Mapping[str, str],
) -> list[dict[str, str]]:
    """List stored credits for mapped staff whose profiles are not on the work."""
    stored_by_doi = {paper.external_id.lower(): paper for paper in stored}
    mapped_staff = {profile.staff_name for profile in profiles.values()}
    findings: list[dict[str, str]] = []
    for work in works:
        paper = stored_by_doi.get(normalize_doi(work.get("doi")))
        if paper is None:
            continue
        on_work = {
            profiles[author].staff_name
            for author in map(_author_id, work.get("authorships") or [])
            if author in profiles
        }
        for name in paper.canonical_authors:
            person = aliases.get(name, name)
            if person in mapped_staff and person not in on_work:
                findings.append(
                    {
                        "external_id": paper.external_id,
                        "title": paper.title,
                        "staff": person,
                    },
                )
    return findings


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    params: Mapping[str, str | int] | None = None,
) -> Any:  # ruff: ignore[ANN401] -- callers read known keys from external JSON
    label = httpx.URL(url).host
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await client.get(url, params=params)
        except httpx.TransportError:
            # Chained errors could carry the request URL and its API key.
            if attempt == _MAX_ATTEMPTS - 1:
                msg = f"Could not reach {label}"
                raise PaperRefreshError(msg) from None
        else:
            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError:
                    msg = f"{label} returned invalid JSON"
                    raise PaperRefreshError(msg) from None
            if (
                response.status_code not in _RETRY_STATUSES
                or attempt == _MAX_ATTEMPTS - 1
            ):
                msg = f"{label} returned status {response.status_code}"
                raise PaperRefreshError(msg)
        await asyncio.sleep(2 + attempt * 3)
    msg = f"Could not read {label}"
    raise PaperRefreshError(msg)


async def load_directus_identity(
    client: httpx.AsyncClient,
    settings: Settings,
) -> tuple[dict[str, Profile], dict[str, str]]:
    staff_url = settings.DIRECTUS_STAFF_URL
    profiles_url = staff_url.removesuffix("staff") + "staff_openalex_profiles"
    staff = (await _get_json(client, staff_url, {"limit": -1, "fields": "id,name"}))[
        "data"
    ]
    names = {row["id"]: row["name"].strip() for row in staff}
    links = (
        await _get_json(
            client,
            f"{settings.DIRECTUS_THESIS_STAFF_URL}?limit=-1"
            "&aggregate[count]=id&groupBy[]=name&groupBy[]=staff",
        )
    )["data"]
    resolved: defaultdict[str, set[str]] = defaultdict(set)
    for link in links:
        if link.get("staff") in names:
            resolved[link["name"].strip()].add(names[link["staff"]])
    aliases = {
        written: next(iter(people))
        for written, people in resolved.items()
        if len(people) == 1 and written not in people
    }
    rows = (
        await _get_json(
            client,
            profiles_url,
            {"limit": -1, "fields": "openalex_id,staff,written_name"},
        )
    )["data"]
    if not rows:
        msg = "Directus returned no OpenAlex profiles"
        raise PaperRefreshError(msg)
    profiles = {
        row["openalex_id"]: Profile(
            row["openalex_id"],
            names[row["staff"]],
            row.get("written_name"),
        )
        for row in rows
    }
    return profiles, aliases


async def fetch_openalex_works(
    client: httpx.AsyncClient,
    profile_ids: Sequence[str],
    api_key: str,
) -> list[dict[str, Any]]:
    works: dict[str, dict[str, Any]] = {}
    for start in range(0, len(profile_ids), _AUTHOR_BATCH):
        batch = "|".join(profile_ids[start : start + _AUTHOR_BATCH])
        cursor: str | None = "*"
        while cursor:
            page = await _get_json(
                client,
                _OPENALEX_WORKS_URL,
                {
                    "filter": f"authorships.author.id:{batch}",
                    "select": _WORK_FIELDS,
                    "per_page": _PAGE_SIZE,
                    "cursor": cursor,
                    "api_key": api_key,
                },
            )
            for work in page["results"]:
                works[work["id"]] = work
            cursor = page["meta"].get("next_cursor") if page["results"] else None
    return list(works.values())


def _json_list(value: str | Sequence[str] | None) -> tuple[str, ...]:
    decoded: Sequence[str] = (
        json.loads(value) if isinstance(value, str) else value or ()
    )
    return tuple(str(item) for item in decoded)


def _stored_paper(row: Record) -> StoredPaper:
    return StoredPaper(
        external_id=row["external_id"],
        title=row["title"],
        canonical_authors=_json_list(row["canonical_authors"]),
        sources=_json_list(row["sources"]),
    )


_STORED_SQL: Final = (
    "SELECT external_id, title, canonical_authors, sources FROM professor_document"
)


async def apply_refresh_plan(
    connection: PoolConnectionProxy[Record],
    plan: RefreshPlan,
) -> None:
    await connection.executemany(
        "INSERT INTO professor_document "
        "(external_id, title, abstract, year, topics, canonical_authors, sources) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb)",
        [
            (
                paper.external_id,
                paper.title,
                paper.abstract,
                paper.year,
                json.dumps(paper.topics, ensure_ascii=False),
                json.dumps(paper.canonical_authors, ensure_ascii=False),
                json.dumps(paper.sources, ensure_ascii=False),
            )
            for paper in plan.new_papers
        ],
    )
    await connection.executemany(
        "UPDATE professor_document SET canonical_authors = $2::jsonb, "
        "sources = $3::jsonb, updated_at = NOW() WHERE external_id = $1",
        [
            (
                update.external_id,
                json.dumps(update.canonical_authors, ensure_ascii=False),
                json.dumps(update.sources, ensure_ascii=False),
            )
            for update in plan.updates
        ],
    )


async def refresh_papers(
    db: Database,
    works: Sequence[Work],
    profiles: Mapping[str, Profile],
    aliases: Mapping[str, str],
    *,
    expected_plan: str | None,
) -> tuple[RefreshPlan, list[dict[str, str]]]:
    """Plan, and write only when the current plan matches the approved one."""
    if expected_plan is None:
        stored = [_stored_paper(row) for row in await db.fetch(_STORED_SQL)]
        plan = build_refresh_plan(works, profiles, stored, aliases)
        return plan, unsupported_credits(works, profiles, stored, aliases)

    async with db.transaction() as connection:
        await connection.execute(
            "SELECT pg_advisory_xact_lock($1)",
            _REFRESH_LOCK_KEY,
        )
        stored = [_stored_paper(row) for row in await connection.fetch(_STORED_SQL)]
        plan = build_refresh_plan(works, profiles, stored, aliases)
        if plan.fingerprint != expected_plan:
            msg = (
                "The refresh plan changed since it was reviewed: "
                f"expected {expected_plan}, got {plan.fingerprint}"
            )
            raise PaperRefreshError(msg)
        await apply_refresh_plan(connection, plan)
    return plan, unsupported_credits(works, profiles, stored, aliases)


def _report(
    plan: RefreshPlan,
    unsupported: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "plan": plan.fingerprint,
        "new_papers": [asdict(paper) for paper in plan.new_papers],
        "credit_updates": [asdict(update) for update in plan.updates],
        "unsupported_existing_credits": unsupported,
    }


async def run(
    *,
    expected_plan: str | None,
    report_path: Path | None,
) -> int:
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if not api_key:
        print("OPENALEX_API_KEY is required", file=sys.stderr)
        return 1

    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings()
    db = Database(dsn=settings.DATABASE_URL, min_size=1, max_size=2)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            profiles, aliases = await load_directus_identity(client, settings)
            works = await fetch_openalex_works(client, sorted(profiles), api_key)
        await db.init()
        plan, unsupported = await refresh_papers(
            db,
            works,
            profiles,
            aliases,
            expected_plan=expected_plan,
        )
    except PaperRefreshError as exc:
        print(f"paper refresh failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await db.disconnect()

    if report_path is not None:
        await asyncio.to_thread(
            report_path.write_text,
            json.dumps(_report(plan, unsupported), ensure_ascii=False),
            encoding="utf-8",
        )
    print(
        f"mode={'applied' if expected_plan else 'dry-run'} profiles={len(profiles)} "
        f"works={len(works)} new_papers={len(plan.new_papers)} "
        f"updated_papers={len(plan.updates)} "
        f"credits_added={sum(len(update.added) for update in plan.updates)} "
        f"unsupported_existing_credits={len(unsupported)} plan={plan.fingerprint}",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        metavar="PLAN",
        help="write the plan with this fingerprint from a reviewed dry run",
    )
    parser.add_argument("--report", type=Path, help="write the full plan as JSON")
    namespace = parser.parse_args(argv)
    return asyncio.run(
        run(expected_plan=namespace.apply, report_path=namespace.report),
    )


if __name__ == "__main__":
    raise SystemExit(main())
