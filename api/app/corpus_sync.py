"""Guarded synchronization of a validated, pinned RAG corpus bundle.

This module deliberately uses the existing document ingestion and embedding
worker lifecycle.  It never calls the streaming ``/documents/fill`` path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import cast
from urllib.parse import urlsplit
from uuid import UUID

from app.data.connection import Database
from app.data.documents import (
    delete_all_documents_query,
    delete_document_compare_and_set_query,
    get_closest_chunks,
    list_documents_query,
    upsert_document_query,
)
from app.llms.chunking import chunk_markdown
from app.llms.embedding_generation import generate_embeddings
from app.llms.models import BGE_M3_EMBEDDING_SPEC_VERSION, Model
from app.schemas.documents import DocumentSchema, IngestDocumentSchema

OWNED_PREFIXES: tuple[str, ...] = ("legal/", "website/")
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_DELETIONS = 0
RELEASE_SCHEMA_VERSION = 1
SYNC_ADVISORY_LOCK_KEY = 0x46494E4B52414753
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_OWNED_NAME = re.compile(r"(?:legal|website)/[^/]+\Z")


class BundleValidationError(ValueError):
    """Raised when a release bundle is malformed or unsafe to apply."""


class EmbeddingNotReadyError(RuntimeError):
    """Raised when the worker did not make every target chunk current in time."""

    def __init__(
        self, names: frozenset[str], verification: EmbeddingVerification
    ) -> None:
        self.names = names
        self.verification = verification
        super().__init__(
            "BGE-M3 embeddings are not ready for "
            f"{len(names)} target documents "
            f"(ready={verification.ready_chunks}/{verification.total_chunks}, "
            f"dirty={verification.dirty_chunks})",
        )


class DeletionCapExceededError(RuntimeError):
    """Raised when a plan would remove more owned documents than approved."""


class ConcurrentInventoryError(RuntimeError):
    """Raised when live inventory changed after the plan was created."""


class ConcurrentSyncRunError(RuntimeError):
    """Raised when another synchronizer already owns the run lock."""


class RetrievalSmokeError(RuntimeError):
    """Raised when a legal or website retrieval citation smoke check fails."""


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    name: str
    title: str
    content: str
    metadata: dict[str, str]
    content_sha256: str
    metadata_sha256: str


@dataclass(frozen=True, slots=True)
class CorpusBundle:
    source_commit: str
    manifest_sha256: str
    entries: tuple[CorpusEntry, ...]
    raw_bundle_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    name: str
    document_id: UUID
    source_hash: str | None
    title: str
    metadata: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class SyncPlan:
    add: tuple[CorpusEntry, ...]
    update: tuple[CorpusEntry, ...]
    unchanged: tuple[str, ...]
    delete: tuple[str, ...]
    expected_names: frozenset[str]
    initial_inventory: tuple[InventoryEntry, ...] = ()
    legacy_delete: tuple[str, ...] = ()
    preserved: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EmbeddingVerification:
    target_documents: int
    total_chunks: int
    ready_chunks: int
    dirty_chunks: int
    document_status: tuple[DocumentEmbeddingVerification, ...] = ()
    missing_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class DocumentEmbeddingVerification:
    name: str
    total_chunks: int
    ready_chunks: int
    dirty_chunks: int


@dataclass(frozen=True, slots=True)
class SyncResult:
    added: tuple[str, ...]
    updated: tuple[str, ...]
    unchanged: tuple[str, ...]
    planned_delete: tuple[str, ...]
    deleted: tuple[str, ...]
    dry_run: bool
    legacy_deleted: tuple[str, ...] = ()
    preserved: tuple[str, ...] = ()

    @property
    def add(self) -> tuple[str, ...]:
        return self.added

    @property
    def update(self) -> tuple[str, ...]:
        return self.updated

    @property
    def delete(self) -> tuple[str, ...]:
        return self.planned_delete

    @classmethod
    def from_plan(
        cls,
        plan: SyncPlan,
        *,
        dry_run: bool,
        deleted: tuple[str, ...] = (),
    ) -> SyncResult:
        return cls(
            added=tuple(entry.name for entry in plan.add),
            updated=tuple(entry.name for entry in plan.update),
            unchanged=plan.unchanged,
            planned_delete=plan.delete,
            deleted=deleted,
            dry_run=dry_run,
            legacy_deleted=tuple(
                name for name in deleted if name in plan.legacy_delete
            ),
            preserved=plan.preserved,
        )


def _invalid(message: str) -> BundleValidationError:
    return BundleValidationError(f"invalid RAG corpus bundle: {message}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata_sha256(metadata: Mapping[str, str]) -> str:
    stable = {
        key: value
        for key, value in metadata.items()
        if key not in {"source_commit", "manifest_hash"}
    }
    encoded = json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return _sha256_text(encoded)


def _is_public_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _entry_from_json(raw: object, index: int) -> CorpusEntry:
    if not isinstance(raw, Mapping):
        raise _invalid(f"entry {index} must be an object")
    required = {
        "name",
        "title",
        "content",
        "metadata",
        "content_sha256",
        "metadata_sha256",
    }
    if set(raw) != required:
        raise _invalid(f"entry {index} has unexpected or missing fields")
    values = {field: raw[field] for field in required}
    if not all(
        isinstance(values[field], str) for field in ("name", "title", "content")
    ):
        raise _invalid(f"entry {index} name, title, and content must be strings")
    name = values["name"]
    title = values["title"]
    content = values["content"]
    if not name or not title or not content.strip():
        raise _invalid(f"entry {index} has an empty name, title, or content")
    if not _OWNED_NAME.fullmatch(name) or ".." in name or "\\" in name:
        raise _invalid(f"entry {index} has an unsafe or unowned name")
    raw_metadata = values["metadata"]
    if not isinstance(raw_metadata, Mapping) or not raw_metadata:
        raise _invalid(f"entry {index} metadata must be a non-empty object")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise _invalid(f"entry {index} metadata must contain only strings")
    metadata = dict(raw_metadata)
    for key in ("authority_url", "source_url"):
        if not _is_public_url(metadata.get(key, "")):
            raise _invalid(f"entry {index} metadata has no public {key}")
    content_hash = values["content_sha256"]
    metadata_hash = values["metadata_sha256"]
    if not isinstance(content_hash, str) or not _SHA256.fullmatch(content_hash):
        raise _invalid(f"entry {index} has an invalid content hash")
    if not isinstance(metadata_hash, str) or not _SHA256.fullmatch(metadata_hash):
        raise _invalid(f"entry {index} has an invalid metadata hash")
    if _sha256_text(content) != content_hash:
        raise _invalid(f"entry {name!r} content hash does not match payload")
    if _metadata_sha256(metadata) != metadata_hash:
        raise _invalid(f"entry {name!r} metadata hash does not match payload")
    return CorpusEntry(name, title, content, metadata, content_hash, metadata_hash)


def validate_bundle(bundle: CorpusBundle) -> CorpusBundle:
    """Validate a bundle object before it is used for planning or mutation."""
    if not _SOURCE_COMMIT.fullmatch(bundle.source_commit):
        raise _invalid("source_commit is not a pinned commit identifier")
    if not _SHA256.fullmatch(bundle.manifest_sha256):
        raise _invalid("manifest_sha256 is not a lowercase SHA-256 digest")
    if bundle.raw_bundle_sha256 is not None and not _SHA256.fullmatch(
        bundle.raw_bundle_sha256
    ):
        raise _invalid("raw_bundle_sha256 is not a lowercase SHA-256 digest")
    if not bundle.entries:
        raise _invalid("entries must not be empty")
    names = [entry.name for entry in bundle.entries]
    if len(names) != len(set(names)):
        raise _invalid("entry names must be unique")
    for index, entry in enumerate(bundle.entries):
        _entry_from_json(
            {
                "name": entry.name,
                "title": entry.title,
                "content": entry.content,
                "metadata": entry.metadata,
                "content_sha256": entry.content_sha256,
                "metadata_sha256": entry.metadata_sha256,
            },
            index,
        )
        if entry.metadata.get("source_commit") != bundle.source_commit:
            raise _invalid(f"entry {entry.name!r} source commit differs from bundle")
        if entry.metadata.get("manifest_hash") != bundle.manifest_sha256:
            raise _invalid(f"entry {entry.name!r} manifest hash differs from bundle")
    return bundle


def load_bundle(text: str, *, raw_bundle_sha256: str | None = None) -> CorpusBundle:
    """Parse and validate the JSON export produced by the source repository."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _invalid(f"JSON parse failed: {exc.msg}") from exc
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "source_revision",
        "manifest_sha256",
        "entries",
    }:
        raise _invalid(
            "bundle must contain only schema_version, source_revision, "
            "manifest_sha256, and entries"
        )
    schema_version = raw["schema_version"]
    source_commit = raw["source_revision"]
    manifest_sha256 = raw["manifest_sha256"]
    entries_raw = raw["entries"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != RELEASE_SCHEMA_VERSION
    ):
        raise _invalid(f"unsupported schema_version {schema_version!r}")
    if not isinstance(source_commit, str) or not isinstance(manifest_sha256, str):
        raise _invalid("release revision and manifest hash must be strings")
    if source_commit.startswith("WORKTREE-"):
        raise _invalid("release bundles must use a pinned source revision")
    if not isinstance(entries_raw, list):
        raise _invalid("entries must be an array")
    entries = tuple(
        _entry_from_json(entry, index) for index, entry in enumerate(entries_raw)
    )
    bundle = CorpusBundle(
        source_commit,
        manifest_sha256,
        entries,
        raw_bundle_sha256=raw_bundle_sha256 or _sha256_text(text),
    )
    return validate_bundle(bundle)


def load_bundle_path(
    path: str | Path,
    *,
    expected_bundle_sha256: str | None = None,
) -> CorpusBundle:
    try:
        payload = Path(path).read_bytes()
    except (OSError, UnicodeDecodeError) as exc:
        raise BundleValidationError(f"cannot read bundle: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if expected_bundle_sha256 is not None:
        if not _SHA256.fullmatch(expected_bundle_sha256):
            raise _invalid("expected bundle SHA-256 is not a lowercase digest")
        if actual != expected_bundle_sha256:
            raise BundleValidationError(
                "bundle SHA-256 does not match the approved expectation"
            )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleValidationError(f"cannot read bundle: {exc}") from exc
    return load_bundle(text, raw_bundle_sha256=actual)


def _owned(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in OWNED_PREFIXES)


def _inventory_entry(document: DocumentSchema) -> InventoryEntry:
    return InventoryEntry(
        name=document.name,
        document_id=document.id,
        source_hash=document.source_hash,
        title=document.title,
        metadata=document.metadata,
    )


def _inventory_snapshot(
    documents: Sequence[DocumentSchema],
) -> tuple[InventoryEntry, ...]:
    return tuple(
        sorted(
            (_inventory_entry(document) for document in documents),
            key=lambda item: item.name,
        )
    )


def _legacy_names(
    documents: Sequence[InventoryEntry], desired: Mapping[str, CorpusEntry]
) -> frozenset[str]:
    legal_entries = {
        name.removeprefix("legal/"): entry
        for name, entry in desired.items()
        if name.startswith("legal/")
    }
    proven: set[str] = set()
    for document in documents:
        entry = legal_entries.get(document.name)
        if entry is None or "/" in document.name:
            continue
        stored = document.metadata or {}
        provenance_keys = (
            "source_path",
            "source_url",
            "authority_url",
            "source_class",
        )
        if document.source_hash == entry.content_sha256 and all(
            stored.get(key) == entry.metadata.get(key) for key in provenance_keys
        ):
            proven.add(document.name)
    return frozenset(proven)


async def build_plan(db: Database, bundle: CorpusBundle) -> SyncPlan:
    validate_bundle(bundle)
    live = await list_documents_query(db)
    desired = {entry.name: entry for entry in bundle.entries}
    inventory = _inventory_snapshot(live)
    live_by_name = {document.name: document for document in inventory}
    additions: list[CorpusEntry] = []
    updates: list[CorpusEntry] = []
    unchanged: list[str] = []
    for name in sorted(desired):
        entry = desired[name]
        document = live_by_name.get(name)
        if document is None:
            additions.append(entry)
            continue
        if (
            document.source_hash != entry.content_sha256
            or document.title != entry.title
            or (document.metadata or {}) != entry.metadata
        ):
            updates.append(entry)
        else:
            unchanged.append(name)
    legacy = _legacy_names(inventory, desired)
    deletions = tuple(
        sorted(
            name
            for name in live_by_name
            if (name not in desired and (_owned(name) or name in legacy))
        )
    )
    preserved = tuple(
        sorted(
            name
            for name in live_by_name
            if name not in desired and name not in deletions
        )
    )
    return SyncPlan(
        add=tuple(additions),
        update=tuple(updates),
        unchanged=tuple(unchanged),
        delete=deletions,
        expected_names=frozenset(desired),
        initial_inventory=inventory,
        legacy_delete=tuple(sorted(legacy)),
        preserved=preserved,
    )


def _assert_plan_matches_bundle(plan: SyncPlan, bundle: CorpusBundle) -> None:
    desired = {entry.name for entry in bundle.entries}
    planned = {entry.name for entry in plan.add + plan.update} | set(plan.unchanged)
    recognized_legacy = _legacy_names(
        plan.initial_inventory, {entry.name: entry for entry in bundle.entries}
    )
    non_legacy_delete = set(plan.delete) - set(plan.legacy_delete)
    if (
        plan.expected_names != desired
        or planned != desired
        or set(plan.delete) & desired
        or set(plan.legacy_delete) != recognized_legacy
        or not non_legacy_delete.issubset(
            {item.name for item in plan.initial_inventory if _owned(item.name)}
        )
        or not set(plan.legacy_delete).issubset(plan.delete)
        or not set(plan.legacy_delete).isdisjoint(desired)
    ):
        raise RuntimeError("sync plan does not match the validated corpus bundle")


def _replace_all_plan(plan: SyncPlan, bundle: CorpusBundle) -> SyncPlan:
    """Turn the reviewed inventory into an explicit destructive replacement plan."""
    return SyncPlan(
        add=bundle.entries,
        update=(),
        unchanged=(),
        delete=tuple(item.name for item in plan.initial_inventory),
        expected_names=frozenset(entry.name for entry in bundle.entries),
        initial_inventory=plan.initial_inventory,
        preserved=(),
    )


async def _assert_replace_all_preconditions(plan: SyncPlan, db: Database) -> None:
    current = _inventory_snapshot(await list_documents_query(db))
    if current != plan.initial_inventory:
        raise ConcurrentInventoryError(
            "live inventory changed since the replace-all plan"
        )


async def assert_exact_replacement(db: Database, bundle: CorpusBundle) -> None:
    """Verify names, document chunk counts, and current embeddings after replacement."""
    live = await list_documents_query(db)
    desired = {entry.name: entry for entry in bundle.entries}
    if {document.name for document in live} != set(desired):
        raise RuntimeError("replace-all final document names differ from the bundle")
    expected_counts = {
        entry.name: len(chunk_markdown(entry.content)) for entry in bundle.entries
    }
    for item in live:
        if item.title != desired[item.name].title:
            raise RuntimeError(
                f"replace-all title verification failed for {item.name!r}"
            )
        if item.source_hash != desired[item.name].content_sha256:
            raise RuntimeError(
                f"replace-all source verification failed for {item.name!r}"
            )
        if (item.metadata or {}) != desired[item.name].metadata:
            raise RuntimeError(
                f"replace-all metadata verification failed for {item.name!r}"
            )
    counts_by_name = {document.name: document for document in live}
    if any(
        counts_by_name[name].chunk_count != count
        for name, count in expected_counts.items()
    ):
        raise RuntimeError("replace-all chunk counts differ from the bundle")
    verification = await embedding_verification(db, frozenset(desired))
    if verification.total_chunks != sum(
        expected_counts.values()
    ) or not _embeddings_ready(verification, frozenset(desired)):
        raise RuntimeError("replace-all embeddings are not current for every chunk")


async def upsert_document_from_entry(db: Database, entry: CorpusEntry) -> None:
    payload = IngestDocumentSchema(
        name=entry.name,
        title=entry.title,
        content=entry.content,
        source_type="markdown",
        metadata=entry.metadata,
    )
    await upsert_document_query(db, payload)


async def embedding_verification(
    db: Database,
    names: frozenset[str],
) -> EmbeddingVerification:
    if not names:
        return EmbeddingVerification(0, 0, 0, 0)
    rows = await db.fetch(
        """
        SELECT d.name,
            COUNT(c.id) AS total_chunks,
            COUNT(c.id) FILTER (
                WHERE c.embedding_bge_m3 IS NOT NULL
                  AND c.embedding_bge_m3_version = $2
            ) AS ready_chunks,
            COUNT(c.id) FILTER (
                WHERE c.embedding_bge_m3 IS NULL
                   OR c.embedding_bge_m3_version IS DISTINCT FROM $2
            ) AS dirty_chunks
        FROM document d
        LEFT JOIN chunk c ON c.document_id = d.id
        WHERE d.name = ANY($1::text[])
        GROUP BY d.name
        """,
        sorted(names),
        BGE_M3_EMBEDDING_SPEC_VERSION,
    )
    status = tuple(
        DocumentEmbeddingVerification(
            name=str(row["name"]),
            total_chunks=int(row["total_chunks"]),
            ready_chunks=int(row["ready_chunks"]),
            dirty_chunks=int(row["dirty_chunks"]),
        )
        for row in rows
    )
    missing_names = names - {item.name for item in status}
    return EmbeddingVerification(
        target_documents=len(status),
        total_chunks=sum(item.total_chunks for item in status),
        ready_chunks=sum(item.ready_chunks for item in status),
        dirty_chunks=sum(item.dirty_chunks for item in status),
        document_status=status,
        missing_names=frozenset(missing_names),
    )


def _embeddings_ready(
    verification: EmbeddingVerification,
    names: frozenset[str],
) -> bool:
    if (
        verification.target_documents != len(names)
        or verification.missing_names
        or verification.ready_chunks != verification.total_chunks
        or verification.dirty_chunks != 0
    ):
        return False
    if not verification.document_status:
        return verification.total_chunks > 0 or not names
    return all(
        item.total_chunks >= 1
        and item.ready_chunks == item.total_chunks
        and item.dirty_chunks == 0
        for item in verification.document_status
    )


async def wait_for_current_embeddings(
    db: Database,
    names: frozenset[str],
    timeout_seconds: int,
) -> EmbeddingVerification:
    """Poll the worker-owned lifecycle until every target chunk is current."""
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must not be negative")
    deadline = monotonic() + timeout_seconds
    last = await embedding_verification(db, names)
    if _embeddings_ready(last, names):
        return last
    while monotonic() < deadline:
        await asyncio.sleep(5)
        last = await embedding_verification(db, names)
        if _embeddings_ready(last, names):
            return last
    raise EmbeddingNotReadyError(names, last)


async def assert_exact_target_metadata(
    db: Database,
    bundle: CorpusBundle,
    *,
    allowed_extra_names: frozenset[str] = frozenset(),
) -> None:
    """Verify the exact owned mirror while preserving unrecognized documents."""
    live = await list_documents_query(db)
    inventory = _inventory_snapshot(live)
    by_name = {document.name: document for document in inventory}
    desired = {entry.name: entry for entry in bundle.entries}
    legacy = _legacy_names(inventory, desired)
    if legacy - allowed_extra_names:
        raise RuntimeError("exact mirror contains recognized legacy document names")
    expected_owned = set(desired) | {
        name for name in allowed_extra_names if _owned(name)
    }
    actual_owned = {name for name in by_name if _owned(name)}
    if actual_owned != expected_owned:
        raise RuntimeError("owned document set differs from the sync target")
    for name, entry in desired.items():
        document = by_name[name]
        if (
            document.title != entry.title
            or document.source_hash != entry.content_sha256
            or (document.metadata or {}) != entry.metadata
        ):
            raise RuntimeError(f"metadata verification failed for {name!r}")


async def _assert_predeletion_inventory(
    db: Database,
    bundle: CorpusBundle,
    plan: SyncPlan,
) -> None:
    """Re-read all inventory and reject stale or newly introduced delete targets."""
    current = _inventory_snapshot(await list_documents_query(db))
    initial_by_name = {item.name: item for item in plan.initial_inventory}
    current_by_name = {item.name: item for item in current}
    desired = {entry.name for entry in bundle.entries}
    expected_names = desired | (set(initial_by_name) - desired)
    if set(current_by_name) != expected_names:
        raise ConcurrentInventoryError("live inventory changed since the sync plan")
    for name, observed in initial_by_name.items():
        if name not in desired and current_by_name[name] != observed:
            raise ConcurrentInventoryError(f"planned inventory row changed: {name}")
    if set(plan.delete) & desired or not set(plan.delete).issubset(current_by_name):
        raise ConcurrentInventoryError(
            "planned deletions overlap desired names or disappeared"
        )
    await assert_exact_target_metadata(
        db,
        bundle,
        allowed_extra_names=frozenset(plan.delete),
    )


@asynccontextmanager
async def single_writer_lock(db: Database) -> AsyncGenerator[None]:
    """Acquire the synchronizer lock without waiting for another run."""
    async with db.try_advisory_lock(SYNC_ADVISORY_LOCK_KEY) as acquired:
        if not acquired:
            raise ConcurrentSyncRunError(
                "another corpus synchronization is in progress"
            )
        yield


async def retrieval_citation_smoke(
    db: Database,
    bundle: CorpusBundle,
) -> None:
    """Require one legal and one website result with the expected citation URL."""
    entries = {entry.name: entry for entry in bundle.entries}
    selected: list[CorpusEntry] = []
    for prefix in ("legal/", "website/"):
        entry = next(
            (entry for entry in entries.values() if entry.name.startswith(prefix)),
            None,
        )
        if entry is None:
            raise RetrievalSmokeError("bundle lacks legal and website smoke samples")
        selected.append(entry)
    for entry in selected:
        try:
            vectors = await generate_embeddings([entry.title], Model.BGE_M3_LOCAL)
        except Exception as exc:
            raise RetrievalSmokeError(
                f"retrieval smoke failed for {entry.name}"
            ) from exc
        if len(vectors) != 1:
            raise RetrievalSmokeError(
                f"retrieval embedding cardinality mismatch for {entry.name}"
            )
        try:
            chunks = await get_closest_chunks(
                db,
                vectors[0],
                Model.BGE_M3_LOCAL,
                limit=8,
            )
        except Exception as exc:
            raise RetrievalSmokeError(
                f"retrieval smoke failed for {entry.name}"
            ) from exc
        authority_url = entry.metadata["authority_url"]
        if not any(
            chunk.document_name == entry.name
            and chunk.document_title == entry.title
            and str(chunk.document_authority_url) == authority_url
            for chunk in chunks
        ):
            raise RetrievalSmokeError(f"retrieval citation mismatch for {entry.name}")


type RetrievalSmokeCheck = Callable[[Database, CorpusBundle], Awaitable[None]]


async def apply_plan(
    db: Database,
    bundle: CorpusBundle,
    plan: SyncPlan,
    *,
    apply: bool = False,
    max_deletions: int = DEFAULT_MAX_DELETIONS,
    timeout_seconds: int = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    retrieval_check: RetrievalSmokeCheck | None = None,
    expected_source_commit: str | None = None,
    expected_bundle_sha256: str | None = None,
) -> SyncResult:
    """Apply a previously reviewed plan, with dry-run as the safe default."""
    validate_bundle(bundle)
    if not apply:
        _assert_plan_matches_bundle(plan, bundle)
        return SyncResult.from_plan(plan, dry_run=True)
    if (
        expected_source_commit is None
        or expected_bundle_sha256 is None
        or expected_source_commit != bundle.source_commit
        or expected_bundle_sha256 != bundle.raw_bundle_sha256
    ):
        raise BundleValidationError(
            "apply requires matching source commit and raw bundle SHA-256 pins"
        )
    async with single_writer_lock(db):
        plan = await build_plan(db, bundle)
        _assert_plan_matches_bundle(plan, bundle)
        if max_deletions < 0:
            raise ValueError("max_deletions must not be negative")
        if len(plan.delete) > max_deletions:
            raise DeletionCapExceededError(
                f"plan has {len(plan.delete)} deletions; approved cap is {max_deletions}",
            )
        return await _apply_plan_locked(
            db,
            bundle,
            plan,
            timeout_seconds=timeout_seconds,
            retrieval_check=retrieval_check,
        )


async def _apply_plan_locked(
    db: Database,
    bundle: CorpusBundle,
    plan: SyncPlan,
    *,
    timeout_seconds: int,
    retrieval_check: RetrievalSmokeCheck | None,
) -> SyncResult:
    for entry in plan.add + plan.update:
        await upsert_document_from_entry(db, entry)
    await wait_for_current_embeddings(db, plan.expected_names, timeout_seconds)
    await _assert_predeletion_inventory(db, bundle, plan)
    await (retrieval_check or retrieval_citation_smoke)(db, bundle)
    deleted: list[str] = []
    initial_by_name = {item.name: item for item in plan.initial_inventory}
    for name in plan.delete:
        observed = initial_by_name.get(name)
        if observed is None:
            raise ConcurrentInventoryError(
                f"no precondition for planned deletion: {name}"
            )
        deleted_ok = await delete_document_compare_and_set_query(
            db,
            name,
            document_id=observed.document_id,
            source_hash=observed.source_hash,
            title=observed.title,
            metadata=observed.metadata,
        )
        if not deleted_ok:
            raise ConcurrentInventoryError(f"document changed during deletion: {name}")
        deleted.append(name)
    await assert_exact_target_metadata(db, bundle)
    return SyncResult.from_plan(plan, dry_run=False, deleted=tuple(deleted))


async def synchronize(
    db: Database,
    bundle: CorpusBundle,
    *,
    apply: bool = False,
    max_deletions: int = DEFAULT_MAX_DELETIONS,
    timeout_seconds: int = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    retrieval_check: RetrievalSmokeCheck | None = None,
    expected_source_commit: str | None = None,
    expected_bundle_sha256: str | None = None,
    plan_report: Callable[[SyncPlan], None] | None = None,
    replace_all: bool = False,
) -> SyncResult:
    """Build an apply plan only after acquiring the non-blocking run lock."""
    validate_bundle(bundle)
    if not apply:
        if replace_all:
            raise ValueError("replace_all requires apply=True")
        plan = await build_plan(db, bundle)
        return SyncResult.from_plan(plan, dry_run=True)
    if (
        expected_source_commit is None
        or expected_bundle_sha256 is None
        or expected_source_commit != bundle.source_commit
        or expected_bundle_sha256 != bundle.raw_bundle_sha256
    ):
        raise BundleValidationError(
            "apply requires matching source commit and raw bundle SHA-256 pins"
        )
    if max_deletions < 0:
        raise ValueError("max_deletions must not be negative")
    async with single_writer_lock(db):
        plan = await build_plan(db, bundle)
        if replace_all:
            plan = _replace_all_plan(plan, bundle)
        else:
            _assert_plan_matches_bundle(plan, bundle)
        if plan_report is not None:
            plan_report(plan)
        if not replace_all and len(plan.delete) > max_deletions:
            raise DeletionCapExceededError(
                f"plan has {len(plan.delete)} deletions; approved cap is {max_deletions}",
            )
        if replace_all:
            async with db.table_lock("document") as locked_db:
                locked_database = cast(Database, locked_db)
                await _assert_replace_all_preconditions(plan, locked_database)
                deleted_count = await delete_all_documents_query(locked_database)
                if deleted_count != len(plan.delete):
                    raise ConcurrentInventoryError(
                        "replace-all deleted a different number of documents than planned"
                    )
            for entry in bundle.entries:
                await upsert_document_from_entry(db, entry)
            await wait_for_current_embeddings(
                db,
                plan.expected_names,
                timeout_seconds,
            )
            await (retrieval_check or retrieval_citation_smoke)(db, bundle)
            await assert_exact_replacement(db, bundle)
            return SyncResult.from_plan(plan, dry_run=False, deleted=plan.delete)
        return await _apply_plan_locked(
            db,
            bundle,
            plan,
            timeout_seconds=timeout_seconds,
            retrieval_check=retrieval_check,
        )


__all__ = [
    "BundleValidationError",
    "ConcurrentInventoryError",
    "ConcurrentSyncRunError",
    "CorpusBundle",
    "CorpusEntry",
    "DeletionCapExceededError",
    "DocumentEmbeddingVerification",
    "EmbeddingNotReadyError",
    "EmbeddingVerification",
    "InventoryEntry",
    "RetrievalSmokeError",
    "SyncPlan",
    "SyncResult",
    "apply_plan",
    "assert_exact_replacement",
    "assert_exact_target_metadata",
    "build_plan",
    "embedding_verification",
    "load_bundle",
    "load_bundle_path",
    "retrieval_citation_smoke",
    "single_writer_lock",
    "synchronize",
    "upsert_document_from_entry",
    "validate_bundle",
    "wait_for_current_embeddings",
]
