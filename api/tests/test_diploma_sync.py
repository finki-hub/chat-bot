import hashlib
from datetime import date

import httpx
import pytest

from app import sync_committee_data
from app.services import diploma_sync
from app.services.diploma_sync import (
    DiplomaSyncResult,
    DiplomaUpstreamError,
    InvalidDiplomaDataError,
    parse_defended_diplomas,
)


def _record(**overrides: object) -> dict[str, object]:
    return {
        "fileId": "file",
        "student": "123 - Student",
        "title": "Title",
        "description": "Abstract",
        "mentor": "Mentor",
        "member1": "One",
        "member2": "Two",
        "status": "Одбрана",
        "dateOfSubmission": "21.10.2026",
    } | overrides


def test_parse_keeps_defended_rows_and_the_existing_key() -> None:
    fetched, diplomas = parse_defended_diplomas(
        [_record(), _record(status="Пријава", title="Pending")],
    )

    assert fetched == 2
    assert len(diplomas) == 1
    assert diplomas[0].date_of_submission == date(2026, 10, 21)
    assert (
        diplomas[0].external_id
        == hashlib.sha256(b"file|123 - Student|Title|21.10.2026").hexdigest()
    )


def test_parse_collapses_duplicate_upstream_rows() -> None:
    _, diplomas = parse_defended_diplomas([_record(), _record()])

    assert len(diplomas) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        None,
        ["not a record"],
        [_record(status="Пријава")],
        [_record(title=" ")],
        [_record(mentor=None)],
        [_record(description=None)],
        [_record(member1=7)],
        [_record(dateOfSubmission="2026-10-21")],
        [_record(), _record(title=None)],
    ],
)
def test_parse_refuses_the_whole_invalid_payload(payload: object) -> None:
    with pytest.raises(InvalidDiplomaDataError):
        parse_defended_diplomas(payload)


class _FakeClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response

    async def get(self, url: str, **_kwargs: object) -> httpx.Response:
        self.response.request = httpx.Request("GET", url)
        return self.response


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(503), DiplomaUpstreamError),
        (httpx.Response(200, content=b"not json"), InvalidDiplomaDataError),
    ],
)
async def test_sync_refuses_bad_upstream_before_touching_the_database(
    monkeypatch,
    response: httpx.Response,
    error: type[Exception],
) -> None:
    monkeypatch.setattr(
        diploma_sync,
        "get_http_client",
        lambda: _FakeClient(response),
    )

    with pytest.raises(error):
        await diploma_sync.sync_defended_diplomas(None)  # type: ignore[arg-type]


class _FakeDatabase:
    def __init__(self, **_kwargs: object) -> None:
        self.disconnected = False

    async def init(self) -> None:
        return None

    async def disconnect(self) -> None:
        self.disconnected = True


def _patch_job(monkeypatch) -> None:
    monkeypatch.setattr(sync_committee_data, "Database", _FakeDatabase)
    monkeypatch.setattr(sync_committee_data, "init_http_client", lambda: None)

    async def close() -> None:
        return None

    monkeypatch.setattr(sync_committee_data, "close_http_client", close)


def test_job_reports_success_and_passes_accept_fewer(monkeypatch, capsys) -> None:
    _patch_job(monkeypatch)
    calls: list[bool] = []

    async def sync(_db: object, *, accept_fewer: bool) -> DiplomaSyncResult:
        calls.append(accept_fewer)
        return DiplomaSyncResult(10, 8, 1, 2)

    async def groups(_db: object, source: str) -> tuple[int, list[object]]:
        assert source == "defense"
        return 8, [object()]

    monkeypatch.setattr(sync_committee_data, "sync_defended_diplomas", sync)
    monkeypatch.setattr(sync_committee_data, "refresh_professor_groups", groups)

    assert sync_committee_data.main(["--accept-fewer"]) == 0
    assert calls == [True]
    assert "inserted=1 updated=2 group_records=8 groups=1" in capsys.readouterr().out


def test_job_fails_without_refreshing_groups(monkeypatch, capsys) -> None:
    _patch_job(monkeypatch)

    async def sync(_db: object, *, accept_fewer: bool) -> DiplomaSyncResult:
        raise InvalidDiplomaDataError("bad list")

    async def groups(_db: object, source: str) -> tuple[int, list[object]]:
        raise AssertionError("groups must not refresh after a failed sync")

    monkeypatch.setattr(sync_committee_data, "sync_defended_diplomas", sync)
    monkeypatch.setattr(sync_committee_data, "refresh_professor_groups", groups)

    assert sync_committee_data.main([]) == 1
    assert "bad list" in capsys.readouterr().err
