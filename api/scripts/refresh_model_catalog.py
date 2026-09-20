"""Refresh the bundled model catalog from models.dev without adding remote models."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from app.llms.model_catalog import _merge_metadata
from app.llms.model_catalog_policy import MODEL_CATALOG
from app.llms.model_catalog_remote import CatalogMetadataError, parse_models_dev
from app.llms.model_catalog_types import DisplayMetadata, SnapshotCatalog, SnapshotEntry

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "app" / "llms" / "models_snapshot.json"
PROVENANCE_PATH = ROOT / "app" / "llms" / "models_snapshot_provenance.json"
SOURCE = "https://models.dev/api.json"
VERIFIED_ID_SOURCES = {
    "gpt-6-astra": "https://developers.openai.com/api/docs/models/gpt-6-astra",
    "gemini-3.8-flash": "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",
    "claude-fable-5-1": "https://platform.claude.com/docs/en/models/fable-5-1/overview",
    "openrouter:deepseek/deepseek-v4.1-flash": "https://openrouter.ai/api/v1/models",
    "openrouter:qwen/qwen3.8-max-0902": "https://openrouter.ai/api/v1/models",
}


def _serialize_snapshot(snapshot: SnapshotCatalog) -> bytes:
    entries = [
        json.dumps(
            entry.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for entry in snapshot.models
    ]
    text = '{\n  "models": [\n'
    text += ",\n".join(f"    {entry}" for entry in entries)
    text += "\n  ]\n}\n"
    return text.encode("utf-8")


def build_refresh_outputs(
    payload: bytes,
    snapshot_payload: bytes,
    *,
    refreshed_at: str,
) -> tuple[bytes, bytes]:
    """Build and validate both refresh files without touching the filesystem."""
    remote = parse_models_dev(payload)
    expected_models = tuple(policy.model for policy in MODEL_CATALOG)
    missing = tuple(model.value for model in expected_models if model not in remote)
    if missing:
        raise CatalogMetadataError(
            reason="models.dev metadata coverage is incomplete: " + ", ".join(missing),
        )

    current = SnapshotCatalog.model_validate_json(snapshot_payload)
    current_by_id = {entry.id: entry for entry in current.models}
    if len(current_by_id) != len(current.models):
        raise CatalogMetadataError(
            reason="bundled model snapshot contains duplicate IDs"
        )

    entries: list[SnapshotEntry] = []
    for policy in MODEL_CATALOG:
        remote_metadata = remote[policy.model]
        previous = current_by_id.get(policy.model.value)
        if previous is None:
            merged = remote_metadata
        else:
            previous_metadata = DisplayMetadata.model_validate(
                previous.model_dump(exclude={"id"}),
            )
            merged = _merge_metadata(previous_metadata, remote_metadata)
        entries.append(SnapshotEntry(id=policy.model.value, **merged.model_dump()))

    snapshot = SnapshotCatalog(models=tuple(entries))
    snapshot_bytes = _serialize_snapshot(snapshot)
    SnapshotCatalog.model_validate_json(snapshot_bytes)
    provenance = {
        "source": SOURCE,
        "refreshed_at": refreshed_at,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "remote_coverage": {"complete": True, "missing": []},
        "verified_id_sources": VERIFIED_ID_SOURCES,
        "entitlement_verified": False,
    }
    provenance_bytes = (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
    loaded_provenance = json.loads(provenance_bytes)
    if loaded_provenance["sha256"] != hashlib.sha256(payload).hexdigest():
        raise CatalogMetadataError(reason="provenance source hash mismatch")
    if (
        loaded_provenance["snapshot_sha256"]
        != hashlib.sha256(snapshot_bytes).hexdigest()
    ):
        raise CatalogMetadataError(reason="provenance snapshot hash mismatch")
    return snapshot_bytes, provenance_bytes


def _stage(path: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _restore(path: Path, original: bytes | None) -> None:
    if original is None:
        path.unlink(missing_ok=True)
        return
    staged = _stage(path, original)
    try:
        os.replace(staged, path)  # ruff: ignore[PTH105] - required atomic same-dir replace
    finally:
        staged.unlink(missing_ok=True)


def write_refresh_outputs(
    snapshot_bytes: bytes,
    provenance_bytes: bytes,
    *,
    snapshot_path: Path = SNAPSHOT_PATH,
    provenance_path: Path = PROVENANCE_PATH,
) -> None:
    """Atomically replace each file and restore the first if the second fails."""
    original_snapshot = snapshot_path.read_bytes() if snapshot_path.exists() else None
    staged_snapshot: Path | None = None
    staged_provenance: Path | None = None
    snapshot_replaced = False
    primary_error: BaseException | None = None
    try:
        staged_snapshot = _stage(snapshot_path, snapshot_bytes)
        staged_provenance = _stage(provenance_path, provenance_bytes)
        os.replace(  # ruff: ignore[PTH105] - required atomic same-dir replace
            staged_snapshot,
            snapshot_path,
        )
        snapshot_replaced = True
        os.replace(  # ruff: ignore[PTH105] - required atomic same-dir replace
            staged_provenance,
            provenance_path,
        )
    except BaseException as error:
        primary_error = error
        if snapshot_replaced:
            try:
                _restore(snapshot_path, original_snapshot)
            except OSError as restore_exception:
                error.add_note(
                    f"Failed to restore {snapshot_path}: {restore_exception!r}"
                )
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        for staged in (staged_snapshot, staged_provenance):
            if staged is None:
                continue
            try:
                staged.unlink(missing_ok=True)
            except OSError as cleanup_exception:
                cleanup_errors.append(cleanup_exception)
        if cleanup_errors:
            if primary_error is not None:
                for cleanup_failure in cleanup_errors:
                    primary_error.add_note(
                        f"Failed to clean up staged catalog file: {cleanup_failure!r}"
                    )
            else:
                first_cleanup_error, *remaining_cleanup_errors = cleanup_errors
                for cleanup_failure in remaining_cleanup_errors:
                    first_cleanup_error.add_note(
                        f"Additional staged-file cleanup failure: {cleanup_failure!r}"
                    )
                raise first_cleanup_error


def main() -> None:
    request = Request(
        SOURCE,
        headers={"User-Agent": "finki-hub-model-catalog-refresh"},
    )
    with urlopen(request, timeout=30) as response:  # ruff: ignore[S310] - fixed HTTPS source
        payload = response.read()
    snapshot_bytes, provenance_bytes = build_refresh_outputs(
        payload,
        SNAPSHOT_PATH.read_bytes(),
        refreshed_at=(
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
    )
    write_refresh_outputs(snapshot_bytes, provenance_bytes)


if __name__ == "__main__":
    main()
