import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

import app.corpus_sync as sync
from app.data.connection import Database
from app.llms.chunking import chunk_markdown

SOURCE_COMMIT = "a" * 40
MANIFEST_HASH = "b" * 64


def _entry(name: str = "legal/a", content: str = "# A") -> sync.CorpusEntry:
    metadata = {
        "authority_url": "https://www.finki.ukim.mk/authority",
        "source_url": "https://www.finki.ukim.mk/source",
        "source_path": "processed/a.md",
        "source_class": "official_legal",
        "source_commit": SOURCE_COMMIT,
        "manifest_hash": MANIFEST_HASH,
    }
    stable = {
        key: value
        for key, value in metadata.items()
        if key not in {"source_commit", "manifest_hash"}
    }
    metadata_hash = hashlib.sha256(
        json.dumps(
            stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode(),
    ).hexdigest()
    return sync.CorpusEntry(
        name=name,
        title=name.rsplit("/", maxsplit=1)[-1],
        content=content,
        metadata=metadata,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        metadata_sha256=metadata_hash,
    )


def _bundle(*entries: sync.CorpusEntry) -> sync.CorpusBundle:
    return sync.CorpusBundle(
        SOURCE_COMMIT,
        MANIFEST_HASH,
        entries,
        raw_bundle_sha256=MANIFEST_HASH,
    )


def _bundle_with_website() -> sync.CorpusBundle:
    return _bundle(_entry(), _entry("website/home"))


def _document(entry: sync.CorpusEntry) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        name=entry.name,
        title=entry.title,
        source_hash=entry.content_sha256,
        metadata=entry.metadata,
        chunk_count=len(chunk_markdown(entry.content)),
    )


def _db() -> Database:
    return cast("Database", object())


class _ReplaceDatabase:
    @asynccontextmanager
    async def table_lock(self, _table):
        yield self


def _json_entry(entry: sync.CorpusEntry) -> dict[str, object]:
    return {
        "name": entry.name,
        "title": entry.title,
        "content": entry.content,
        "metadata": entry.metadata,
        "content_sha256": entry.content_sha256,
        "metadata_sha256": entry.metadata_sha256,
    }


def _bare_document(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        name=name,
        title=name,
        source_hash=None,
        metadata=None,
        chunk_count=0,
    )


def _proven_legacy(entry: sync.CorpusEntry) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        name=entry.name.removeprefix("legal/"),
        title=entry.title,
        source_hash=entry.content_sha256,
        metadata=entry.metadata.copy(),
    )


@asynccontextmanager
async def _no_lock(_db):
    yield


def test_empty_malformed_and_duplicate_bundles_are_rejected() -> None:
    with pytest.raises(sync.BundleValidationError):
        sync.load_bundle('{"source_commit":"x"')
    with pytest.raises(sync.BundleValidationError):
        sync.load_bundle(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_revision": SOURCE_COMMIT,
                    "manifest_sha256": MANIFEST_HASH,
                    "entries": [],
                },
            ),
        )
    entry = _entry()
    payload = {
        "schema_version": 1,
        "source_revision": SOURCE_COMMIT,
        "manifest_sha256": MANIFEST_HASH,
        "entries": [_json_entry(entry)] * 2,
    }
    with pytest.raises(sync.BundleValidationError):
        sync.load_bundle(json.dumps(payload))
    bad_hash = _json_entry(entry)
    bad_hash["content_sha256"] = "0" * 64
    with pytest.raises(sync.BundleValidationError, match="content hash"):
        sync.load_bundle(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_revision": SOURCE_COMMIT,
                    "manifest_sha256": MANIFEST_HASH,
                    "entries": [bad_hash],
                },
            ),
        )


def test_bundle_rejects_unowned_namespace() -> None:
    entry = _entry("faq/not-owned")
    with pytest.raises(sync.BundleValidationError):
        sync.validate_bundle(_bundle(entry))


def test_release_bundle_rejects_worktree_revision() -> None:
    with pytest.raises(sync.BundleValidationError):
        sync.validate_bundle(
            sync.CorpusBundle(
                "WORKTREE-" + "a" * 64,
                MANIFEST_HASH,
                (_entry(),),
            ),
        )


def test_raw_bundle_sha_is_verified_independently(tmp_path) -> None:
    entry = _entry()
    payload = json.dumps(
        {
            "schema_version": 1,
            "source_revision": SOURCE_COMMIT,
            "manifest_sha256": MANIFEST_HASH,
            "entries": [_json_entry(entry)],
        },
        separators=(",", ":"),
    ).encode()
    path = tmp_path / "release.json"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    bundle = sync.load_bundle_path(path, expected_bundle_sha256=expected)
    assert bundle.raw_bundle_sha256 == expected
    with pytest.raises(sync.BundleValidationError, match="bundle SHA-256"):
        sync.load_bundle_path(path, expected_bundle_sha256="0" * 64)


@pytest.mark.anyio
async def test_apply_requires_the_verified_raw_bundle_sha() -> None:
    entry = _entry()
    plan = sync.SyncPlan((entry,), (), (), (), frozenset({entry.name}))
    with pytest.raises(sync.BundleValidationError, match="bundle SHA-256"):
        await sync.apply_plan(
            _db(),
            _bundle(entry),
            plan,
            apply=True,
            expected_source_commit=SOURCE_COMMIT,
            expected_bundle_sha256="0" * 64,
        )


@pytest.mark.anyio
async def test_replace_all_deletes_everything_only_after_preconditions_and_reimports(
    monkeypatch,
) -> None:
    bundle = _bundle(
        _entry(content="A paragraph"),
        _entry("website/home", content="Website paragraph"),
    )
    old_documents = [_bare_document("faq/keep"), _bare_document("legal/old")]
    final_documents = [_document(entry) for entry in bundle.entries]
    inventories = iter((old_documents, old_documents, final_documents))
    events: list[str] = []

    async def list_documents(_db):
        return next(inventories)

    async def delete_all(_db):
        events.append("delete-all")
        return len(old_documents)

    async def upload(_db, entry):
        events.append(f"import:{entry.name}")

    async def embeddings(_db, _names, _timeout):
        events.append("embeddings")
        total = sum(len(chunk_markdown(entry.content)) for entry in bundle.entries)
        return sync.EmbeddingVerification(2, total, total, 0)

    async def smoke(_db, _bundle):
        events.append("smoke")

    async def verify(_db, _names):
        total = sum(len(chunk_markdown(entry.content)) for entry in bundle.entries)
        return sync.EmbeddingVerification(2, total, total, 0)

    monkeypatch.setattr(sync, "list_documents_query", list_documents)
    monkeypatch.setattr(sync, "delete_all_documents_query", delete_all)
    monkeypatch.setattr(sync, "upsert_document_from_entry", upload)
    monkeypatch.setattr(sync, "wait_for_current_embeddings", embeddings)
    monkeypatch.setattr(sync, "embedding_verification", verify)
    monkeypatch.setattr(sync, "retrieval_citation_smoke", smoke)
    monkeypatch.setattr(sync, "single_writer_lock", _no_lock)
    result = await sync.synchronize(
        cast(Database, _ReplaceDatabase()),
        bundle,
        apply=True,
        replace_all=True,
        expected_source_commit=SOURCE_COMMIT,
        expected_bundle_sha256=MANIFEST_HASH,
    )

    assert result.added == ("legal/a", "website/home")
    assert result.deleted == ("faq/keep", "legal/old")
    assert events == [
        "delete-all",
        "import:legal/a",
        "import:website/home",
        "embeddings",
        "smoke",
    ]


@pytest.mark.anyio
async def test_replace_all_refuses_delete_when_inventory_precondition_changes(
    monkeypatch,
) -> None:
    bundle = _bundle_with_website()
    initial = [_bare_document("faq/keep")]
    changed = [_bare_document("faq/changed")]
    inventories = iter((initial, changed))
    deleted = False

    async def list_documents(_db):
        return next(inventories)

    async def delete_all(_db):
        nonlocal deleted
        deleted = True
        return 1

    monkeypatch.setattr(sync, "list_documents_query", list_documents)
    monkeypatch.setattr(sync, "delete_all_documents_query", delete_all)
    monkeypatch.setattr(sync, "single_writer_lock", _no_lock)
    with pytest.raises(sync.ConcurrentInventoryError):
        await sync.synchronize(
            cast(Database, _ReplaceDatabase()),
            bundle,
            apply=True,
            replace_all=True,
            expected_source_commit=SOURCE_COMMIT,
            expected_bundle_sha256=MANIFEST_HASH,
        )
    assert deleted is False


@pytest.mark.anyio
async def test_replace_all_rechecks_and_deletes_inside_table_lock(monkeypatch) -> None:
    bundle = _bundle_with_website()
    initial = [_bare_document("faq/keep")]
    changed = [_bare_document("faq/changed")]
    inventories = iter((initial, changed))
    events: list[str] = []

    class LockedDatabase:
        @asynccontextmanager
        async def table_lock(self, table):
            events.append(f"lock-enter:{table}")
            try:
                yield self
            finally:
                events.append("lock-exit")

    async def list_documents(_db):
        return next(inventories)

    async def delete_all(_db):
        events.append("delete-all")
        return 1

    monkeypatch.setattr(sync, "list_documents_query", list_documents)
    monkeypatch.setattr(sync, "delete_all_documents_query", delete_all)
    monkeypatch.setattr(sync, "single_writer_lock", _no_lock)
    with pytest.raises(sync.ConcurrentInventoryError):
        await sync.synchronize(
            cast(Database, LockedDatabase()),
            bundle,
            apply=True,
            replace_all=True,
            expected_source_commit=SOURCE_COMMIT,
            expected_bundle_sha256=MANIFEST_HASH,
        )
    assert events == ["lock-enter:document", "lock-exit"]


@pytest.mark.anyio
async def test_apply_requires_matching_release_pins_before_lock(monkeypatch) -> None:
    entry = _entry()
    plan = sync.SyncPlan((entry,), (), (), (), frozenset({entry.name}))
    locked = False

    @asynccontextmanager
    async def lock(_db):
        nonlocal locked
        locked = True
        yield

    monkeypatch.setattr(sync, "single_writer_lock", lock)
    with pytest.raises(sync.BundleValidationError, match="pins"):
        await sync.apply_plan(_db(), _bundle(entry), plan, apply=True)
    assert locked is False


@pytest.mark.anyio
async def test_plan_is_idempotent_and_deletes_only_owned_names(monkeypatch) -> None:
    desired = _entry()
    live = [
        _document(desired),
        _bare_document("legal/obsolete"),
        _bare_document("faq/keep"),
    ]

    async def list_documents(_db):
        return live[:2]

    monkeypatch.setattr(sync, "list_documents_query", list_documents)
    plan = await sync.build_plan(_db(), _bundle(desired))
    assert plan.add == ()
    assert plan.update == ()
    assert plan.unchanged == (desired.name,)
    assert plan.delete == ("legal/obsolete",)


@pytest.mark.anyio
async def test_plan_recognizes_legacy_legal_names_and_preserves_unknowns(
    monkeypatch,
) -> None:
    desired = _entry()
    collision = _entry("legal/collision")
    live = [
        _document(desired),
        _document(collision),
        _proven_legacy(desired),
        _bare_document("collision"),
        _bare_document("old-unrecognized"),
        _bare_document("faq/keep"),
    ]

    async def list_documents(_db):
        return live

    monkeypatch.setattr(sync, "list_documents_query", list_documents)
    plan = await sync.build_plan(_db(), _bundle(desired, collision))
    assert plan.delete == ("a",)
    assert plan.legacy_delete == ("a",)
    assert plan.preserved == ("collision", "faq/keep", "old-unrecognized")


@pytest.mark.anyio
async def test_exact_mirror_rejects_a_recognized_legacy_name(monkeypatch) -> None:
    desired = _entry()

    async def list_documents(_db):
        return [_document(desired), _proven_legacy(desired)]

    monkeypatch.setattr(sync, "list_documents_query", list_documents)
    with pytest.raises(RuntimeError, match="legacy"):
        await sync.assert_exact_target_metadata(_db(), _bundle(desired))


@pytest.mark.anyio
async def test_single_writer_uses_advisory_lock() -> None:
    seen: list[int] = []

    @asynccontextmanager
    async def advisory_lock(key):
        seen.append(key)
        yield True

    db = SimpleNamespace(try_advisory_lock=advisory_lock)
    async with sync.single_writer_lock(cast("Database", db)):
        pass
    assert seen == [sync.SYNC_ADVISORY_LOCK_KEY]


@pytest.mark.anyio
async def test_concurrent_apply_is_rejected_before_plan_inventory_read(
    monkeypatch,
) -> None:
    @asynccontextmanager
    async def busy_lock(_db):
        raise sync.ConcurrentSyncRunError("busy")
        yield

    async def unexpected_inventory(_db):
        raise AssertionError("apply plan inventory was built before the lock")

    monkeypatch.setattr(sync, "single_writer_lock", busy_lock)
    monkeypatch.setattr(sync, "list_documents_query", unexpected_inventory)
    with pytest.raises(sync.ConcurrentSyncRunError):
        await sync.synchronize(
            _db(),
            _bundle(_entry()),
            apply=True,
            max_deletions=0,
            expected_source_commit=SOURCE_COMMIT,
            expected_bundle_sha256=MANIFEST_HASH,
        )


@pytest.mark.anyio
async def test_dry_run_reports_deletions_without_enforcing_cap() -> None:
    entry = _entry()
    plan = sync.SyncPlan(
        (entry,),
        (),
        (),
        ("legal/one", "website/two"),
        frozenset({entry.name}),
        initial_inventory=(
            sync.InventoryEntry("legal/one", uuid4(), None, "", None),
            sync.InventoryEntry("website/two", uuid4(), None, "", None),
        ),
    )
    result = await sync.apply_plan(
        _db(),
        _bundle(entry),
        plan,
        apply=False,
        max_deletions=0,
    )
    assert result.deleted == ()
    assert result.planned_delete == plan.delete


@pytest.mark.anyio
async def test_apply_rechecks_inventory_and_uses_compare_and_set(monkeypatch) -> None:
    entry = _entry()
    obsolete = _bare_document("legal/obsolete")
    plan = sync.SyncPlan(
        (entry,),
        (),
        (),
        ("legal/obsolete",),
        frozenset({entry.name}),
        initial_inventory=(
            sync.InventoryEntry(
                obsolete.name,
                obsolete.id,
                obsolete.source_hash,
                obsolete.title,
                obsolete.metadata,
            ),
        ),
    )
    current = [_document(entry), obsolete]
    after_delete = [_document(entry)]
    inventories = iter((current, current, after_delete))
    deleted: list[str] = []

    async def list_documents(_db):
        return next(inventories)

    async def upload(_db, _entry):
        return None

    async def embeddings(_db, _names, _timeout):
        return sync.EmbeddingVerification(1, 1, 1, 0)

    async def smoke(_db, _bundle):
        return None

    async def delete(_db, name, **_kwargs):
        deleted.append(name)
        return True

    monkeypatch.setattr(sync, "list_documents_query", list_documents)

    async def planned(_db, _bundle):
        return plan

    monkeypatch.setattr(sync, "build_plan", planned)
    monkeypatch.setattr(sync, "upsert_document_from_entry", upload)
    monkeypatch.setattr(sync, "wait_for_current_embeddings", embeddings)
    monkeypatch.setattr(sync, "retrieval_citation_smoke", smoke)
    monkeypatch.setattr(sync, "delete_document_compare_and_set_query", delete)
    monkeypatch.setattr(sync, "single_writer_lock", _no_lock)
    result = await sync.apply_plan(
        _db(),
        _bundle(entry),
        plan,
        apply=True,
        max_deletions=1,
        expected_source_commit=SOURCE_COMMIT,
        expected_bundle_sha256=MANIFEST_HASH,
    )
    assert result.deleted == ("legal/obsolete",)
    assert deleted == ["legal/obsolete"]


@pytest.mark.anyio
async def test_apply_fails_closed_when_planned_delete_changes(monkeypatch) -> None:
    entry = _entry()
    obsolete = _bare_document("legal/obsolete")
    changed = _bare_document("legal/obsolete")
    changed.id = obsolete.id
    changed.source_hash = "changed"
    plan = sync.SyncPlan(
        (entry,),
        (),
        (),
        ("legal/obsolete",),
        frozenset({entry.name}),
        initial_inventory=(
            sync.InventoryEntry(
                obsolete.name,
                obsolete.id,
                obsolete.source_hash,
                obsolete.title,
                obsolete.metadata,
            ),
        ),
    )
    deleted = False

    async def list_documents(_db):
        return [_document(entry), changed]

    async def delete(_db, _name, **_kwargs):
        nonlocal deleted
        deleted = True
        return True

    async def upload(_db, _entry):
        return None

    async def embeddings(_db, _names, _timeout):
        return sync.EmbeddingVerification(1, 1, 1, 0)

    monkeypatch.setattr(sync, "list_documents_query", list_documents)

    async def planned(_db, _bundle):
        return plan

    monkeypatch.setattr(sync, "build_plan", planned)
    monkeypatch.setattr(sync, "upsert_document_from_entry", upload)
    monkeypatch.setattr(sync, "wait_for_current_embeddings", embeddings)
    monkeypatch.setattr(sync, "single_writer_lock", _no_lock)
    monkeypatch.setattr(sync, "delete_document_compare_and_set_query", delete)
    with pytest.raises(sync.ConcurrentInventoryError):
        await sync.apply_plan(
            _db(),
            _bundle(entry),
            plan,
            apply=True,
            max_deletions=1,
            expected_source_commit=SOURCE_COMMIT,
            expected_bundle_sha256=MANIFEST_HASH,
        )
    assert deleted is False


@pytest.mark.anyio
async def test_apply_never_deletes_when_upload_or_embedding_fails(monkeypatch) -> None:
    entry = _entry()
    bundle = _bundle(entry)
    obsolete = _bare_document("legal/obsolete")
    plan = sync.SyncPlan(
        (entry,),
        (),
        (),
        ("legal/obsolete",),
        frozenset({entry.name}),
        initial_inventory=(
            sync.InventoryEntry(
                obsolete.name,
                obsolete.id,
                obsolete.source_hash,
                obsolete.title,
                obsolete.metadata,
            ),
        ),
    )
    deleted: list[str] = []

    async def delete(_db, name, *_args):
        deleted.append(name)

    monkeypatch.setattr(sync, "delete_document_compare_and_set_query", delete)
    monkeypatch.setattr(sync, "single_writer_lock", _no_lock)

    async def planned(_db, _bundle):
        return plan

    monkeypatch.setattr(sync, "build_plan", planned)

    async def upload_failure(_db, _entry):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(sync, "upsert_document_from_entry", upload_failure)
    with pytest.raises(RuntimeError, match="upload failed"):
        await sync.apply_plan(
            _db(),
            bundle,
            plan,
            apply=True,
            max_deletions=1,
            expected_source_commit=SOURCE_COMMIT,
            expected_bundle_sha256=MANIFEST_HASH,
        )
    assert deleted == []

    async def upload(_db, _entry):
        return None

    async def embedding_failure(_db, _names, _timeout):
        raise sync.EmbeddingNotReadyError(
            frozenset({entry.name}),
            sync.EmbeddingVerification(1, 1, 0, 1),
        )

    monkeypatch.setattr(sync, "upsert_document_from_entry", upload)
    monkeypatch.setattr(sync, "wait_for_current_embeddings", embedding_failure)
    with pytest.raises(sync.EmbeddingNotReadyError):
        await sync.apply_plan(
            _db(),
            bundle,
            plan,
            apply=True,
            max_deletions=1,
            expected_source_commit=SOURCE_COMMIT,
            expected_bundle_sha256=MANIFEST_HASH,
        )
    assert deleted == []


@pytest.mark.anyio
async def test_deletion_cap_is_checked_before_upload(monkeypatch) -> None:
    entry = _entry()
    plan = sync.SyncPlan(
        (entry,),
        (),
        (),
        ("legal/one", "website/two"),
        frozenset({entry.name}),
        initial_inventory=(
            sync.InventoryEntry("legal/one", uuid4(), None, "", None),
            sync.InventoryEntry("website/two", uuid4(), None, "", None),
        ),
    )
    uploaded = False

    async def upload(_db, _entry):
        nonlocal uploaded
        uploaded = True

    monkeypatch.setattr(sync, "upsert_document_from_entry", upload)

    async def planned(_db, _bundle):
        return plan

    monkeypatch.setattr(sync, "build_plan", planned)
    monkeypatch.setattr(sync, "single_writer_lock", _no_lock)
    with pytest.raises(sync.DeletionCapExceededError):
        await sync.apply_plan(
            _db(),
            _bundle(entry),
            plan,
            apply=True,
            max_deletions=1,
            expected_source_commit=SOURCE_COMMIT,
            expected_bundle_sha256=MANIFEST_HASH,
        )
    assert uploaded is False


@pytest.mark.anyio
async def test_wait_requires_all_target_chunks_to_be_current(monkeypatch) -> None:
    responses = iter(
        [
            sync.EmbeddingVerification(1, 2, 1, 1),
            sync.EmbeddingVerification(1, 2, 2, 0),
        ],
    )

    async def verification(_db, _names):
        return next(responses)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(sync, "embedding_verification", verification)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    result = await sync.wait_for_current_embeddings(_db(), frozenset({"legal/a"}), 1)
    assert result.ready_chunks == result.total_chunks == 2
    assert result.dirty_chunks == 0


@pytest.mark.anyio
async def test_wait_requires_every_document_to_have_a_chunk(monkeypatch) -> None:
    responses = iter(
        [
            sync.EmbeddingVerification(
                2,
                1,
                1,
                0,
                document_status=(
                    sync.DocumentEmbeddingVerification("legal/a", 1, 1, 0),
                    sync.DocumentEmbeddingVerification("website/home", 0, 0, 0),
                ),
            ),
        ],
    )

    async def verification(_db, _names):
        return next(responses)

    monkeypatch.setattr(sync, "embedding_verification", verification)
    with pytest.raises(sync.EmbeddingNotReadyError):
        await sync.wait_for_current_embeddings(
            _db(),
            frozenset({"legal/a", "website/home"}),
            0,
        )


@pytest.mark.anyio
async def test_retrieval_smoke_fails_closed_for_a_bad_citation(monkeypatch) -> None:
    async def embeddings(_texts, _model):
        return [[0.0] * 1024]

    async def chunks(_db, _vector, _model, *, limit):
        return [
            SimpleNamespace(
                document_name="legal/a",
                document_title="a",
                document_authority_url="https://www.finki.ukim.mk/authority",
            ),
        ]

    monkeypatch.setattr(sync, "generate_embeddings", embeddings)
    monkeypatch.setattr(sync, "get_closest_chunks", chunks)
    with pytest.raises(sync.RetrievalSmokeError):
        await sync.retrieval_citation_smoke(_db(), _bundle_with_website())
