import json
from hashlib import sha256
from pathlib import Path

import pytest

from app.llms.model_catalog_policy import MODEL_CATALOG
from app.llms.model_catalog_remote import CatalogMetadataError
from app.llms.model_catalog_types import SnapshotCatalog
from scripts import refresh_model_catalog as refresh

SNAPSHOT_PATH = Path(__file__).parents[1] / "app" / "llms" / "models_snapshot.json"


def _remote_payload(
    *,
    overrides: dict[str, dict[str, object]] | None = None,
) -> bytes:
    overrides = overrides or {}
    providers: dict[str, dict[str, object]] = {}
    for policy in MODEL_CATALOG:
        provider = providers.setdefault(
            policy.provider,
            {"id": policy.provider, "name": policy.provider, "models": {}},
        )
        model_id = policy.model.value.removeprefix("openrouter:")
        model = {"id": model_id, "name": policy.display_name}
        model.update(overrides.get(policy.model.value, {}))
        models = provider["models"]
        assert isinstance(models, dict)
        models[model_id] = model
    return json.dumps(providers).encode()


def _snapshot_payload() -> bytes:
    return SNAPSHOT_PATH.read_bytes()


def test_refresh_requires_complete_valid_remote_coverage_without_writes() -> None:
    payload = json.loads(_remote_payload())
    provider = next(iter(payload))
    del payload[provider]["models"][next(iter(payload[provider]["models"]))]
    original = _snapshot_payload()

    with pytest.raises(CatalogMetadataError, match="coverage is incomplete"):
        refresh.build_refresh_outputs(
            json.dumps(payload).encode(),
            original,
            refreshed_at="2026-09-14T00:00:00Z",
        )

    assert SNAPSHOT_PATH.read_bytes() == original


def test_refresh_rejects_malformed_payload_without_writes() -> None:
    original = _snapshot_payload()

    with pytest.raises(CatalogMetadataError, match="not valid JSON"):
        refresh.build_refresh_outputs(
            b"not-json",
            original,
            refreshed_at="2026-09-14T00:00:00Z",
        )

    assert SNAPSHOT_PATH.read_bytes() == original


def test_refresh_allows_new_curated_id_without_preseeded_snapshot() -> None:
    current = SnapshotCatalog.model_validate_json(_snapshot_payload())
    without_new = SnapshotCatalog(
        models=tuple(entry for entry in current.models if entry.id != "gpt-6-astra"),
    )

    snapshot_bytes, _ = refresh.build_refresh_outputs(
        _remote_payload(),
        without_new.model_dump_json().encode(),
        refreshed_at="2026-09-14T00:00:00Z",
    )

    refreshed = SnapshotCatalog.model_validate_json(snapshot_bytes)
    assert refreshed.models[0].id == "gpt-6-astra"


def test_refresh_merges_partial_nested_remote_metadata() -> None:
    snapshot_bytes, _ = refresh.build_refresh_outputs(
        _remote_payload(
            overrides={"gpt-6-astra": {"reasoning": False}},
        ),
        _snapshot_payload(),
        refreshed_at="2026-09-14T00:00:00Z",
    )

    refreshed = SnapshotCatalog.model_validate_json(snapshot_bytes)
    capabilities = refreshed.models[0].capabilities
    assert capabilities is not None
    assert capabilities.reasoning is False
    assert capabilities.tool_call is True
    assert capabilities.structured_output is True
    assert capabilities.temperature is False


def test_refresh_outputs_preserve_order_and_provenance_hashes() -> None:
    payload = _remote_payload()
    snapshot_bytes, provenance_bytes = refresh.build_refresh_outputs(
        payload,
        _snapshot_payload(),
        refreshed_at="2026-09-14T00:00:00Z",
    )

    snapshot = SnapshotCatalog.model_validate_json(snapshot_bytes)
    provenance = json.loads(provenance_bytes)
    assert [entry.id for entry in snapshot.models] == [
        policy.model.value for policy in MODEL_CATALOG
    ]
    assert provenance["sha256"] == sha256(payload).hexdigest()
    assert provenance["snapshot_sha256"] == sha256(snapshot_bytes).hexdigest()
    assert provenance["remote_coverage"] == {"complete": True, "missing": []}


def test_refresh_write_rolls_back_first_file_if_second_replace_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "models_snapshot.json"
    provenance_path = tmp_path / "models_snapshot_provenance.json"
    original_snapshot = b"original snapshot"
    original_provenance = b"original provenance"
    snapshot_path.write_bytes(original_snapshot)
    provenance_path.write_bytes(original_provenance)
    real_replace = refresh.os.replace
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated provenance replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(refresh.os, "replace", fail_second)

    with pytest.raises(OSError, match="provenance replace failure"):
        refresh.write_refresh_outputs(
            b"new snapshot",
            b"new provenance",
            snapshot_path=snapshot_path,
            provenance_path=provenance_path,
        )

    assert snapshot_path.read_bytes() == original_snapshot
    assert provenance_path.read_bytes() == original_provenance
