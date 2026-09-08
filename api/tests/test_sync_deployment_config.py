from pathlib import Path
from types import SimpleNamespace

import pytest

import app.sync_rag_corpus as sync_cli

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_production_sync_identity_is_required_from_compose_configuration() -> None:
    compose = (REPO_ROOT / "compose.prod.yaml").read_text(encoding="utf-8")
    release_pin_sample = (REPO_ROOT / "production-release-pins.sample").read_text(
        encoding="utf-8"
    )
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    compose_prod = (REPO_ROOT / "compose.prod.yaml").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    script = (REPO_ROOT / "api" / "app" / "sync_rag_corpus.py").read_text(
        encoding="utf-8",
    )

    assert (
        "RAG_SYNC_DEPLOYMENT_IDENTITY: ${RAG_SYNC_DEPLOYMENT_IDENTITY:?"
        "RAG_SYNC_DEPLOYMENT_IDENTITY is required}"
    ) in compose
    assert (
        "RAG_SYNC_EXPECTED_BUNDLE_SHA256: ${RAG_SYNC_EXPECTED_BUNDLE_SHA256:?"
        "RAG_SYNC_EXPECTED_BUNDLE_SHA256 is required}"
    ) in compose
    assert (
        "RAG_SYNC_EXPECTED_SOURCE_COMMIT: ${RAG_SYNC_EXPECTED_SOURCE_COMMIT:?"
        "RAG_SYNC_EXPECTED_SOURCE_COMMIT is required}"
    ) in compose
    assert "RAG_SYNC_RELEASES_DIR:?RAG_SYNC_RELEASES_DIR is required" in compose_prod
    assert "target: /releases" in compose_prod
    assert "read_only: true" in compose_prod
    assert "COPY . ." in dockerfile
    assert (REPO_ROOT / "api" / "app" / "sync_rag_corpus.py").is_file()
    assert (REPO_ROOT / "production-release-pins.sample").is_file()
    for setting in (
        "RAG_SYNC_EXPECTED_BUNDLE_SHA256",
        "RAG_SYNC_EXPECTED_SOURCE_COMMIT",
        "RAG_SYNC_DEPLOYMENT_IDENTITY",
        "RAG_SYNC_RELEASES_DIR",
    ):
        assert f"{setting}=" in release_pin_sample
    assert "--expected-deployment-identity" in script
    assert "--expected-bundle-sha256" in script
    assert "--expected-manifest-sha256" not in script
    assert "RAG_SYNC_EXPECTED_BUNDLE_SHA256" in readme
    assert "RAG_SYNC_EXPECTED_SOURCE_COMMIT" in readme
    assert "cp .env.sample .env" in readme
    assert "cat production-release-pins.sample >> .env" in readme
    assert "replace all placeholders" in readme
    assert "--force-recreate api" in readme
    assert 'parser.add_argument("--deployment-identity")' not in script
    assert "from app.sync_rag_corpus import main" in (
        REPO_ROOT / "scripts" / "sync_rag_corpus.py"
    ).read_text(encoding="utf-8")


def test_replace_all_requires_an_explicit_cli_mode() -> None:
    arguments = sync_cli._parser().parse_args(
        [
            "--bundle",
            "/releases/corpus.json",
            "--apply",
            "--replace-all",
        ],
    )
    assert arguments.replace_all is True


@pytest.mark.anyio
async def test_cli_rejects_raw_pin_before_database_initialization(
    tmp_path, monkeypatch
):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("not-json", encoding="utf-8")
    arguments = sync_cli._parser().parse_args(
        [
            "--bundle",
            str(bundle_path),
            "--apply",
            "--replace-all",
            "--expected-source-commit",
            "a" * 40,
            "--expected-bundle-sha256",
            "0" * 64,
            "--expected-deployment-identity",
            "test-identity",
        ],
    )
    monkeypatch.setenv("RAG_SYNC_EXPECTED_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("RAG_SYNC_EXPECTED_BUNDLE_SHA256", "0" * 64)
    monkeypatch.setenv("RAG_SYNC_DEPLOYMENT_IDENTITY", "test-identity")
    monkeypatch.setattr(
        sync_cli,
        "Settings",
        lambda: pytest.fail("database settings must not load after a bad bundle pin"),
    )
    with pytest.raises(sync_cli.BundleValidationError, match="bundle SHA-256"):
        await sync_cli._run(arguments)


@pytest.mark.anyio
async def test_apply_cli_raises_single_pool_for_lock_connection(monkeypatch):
    captured: dict[str, object] = {}

    class CapturingDatabase:
        def __init__(self, _dsn, *, min_size, max_size):
            captured["min_size"] = min_size
            captured["max_size"] = max_size

        async def init(self):
            return None

        async def disconnect(self):
            return None

    async def fake_synchronize(*_args, **_kwargs):
        return SimpleNamespace(dry_run=False, deleted=())

    arguments = sync_cli._parser().parse_args(
        [
            "--bundle",
            "/releases/corpus.json",
            "--apply",
            "--replace-all",
            "--expected-source-commit",
            "a" * 40,
            "--expected-bundle-sha256",
            "b" * 64,
            "--expected-deployment-identity",
            "test-identity",
        ],
    )
    monkeypatch.setenv("RAG_SYNC_EXPECTED_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("RAG_SYNC_EXPECTED_BUNDLE_SHA256", "b" * 64)
    monkeypatch.setenv("RAG_SYNC_DEPLOYMENT_IDENTITY", "test-identity")
    monkeypatch.setattr(
        sync_cli,
        "Settings",
        lambda: SimpleNamespace(
            DATABASE_URL="postgresql://test",
            DATABASE_POOL_MIN_SIZE=1,
            DATABASE_POOL_MAX_SIZE=1,
        ),
    )
    monkeypatch.setattr(sync_cli, "Database", CapturingDatabase)
    monkeypatch.setattr(
        sync_cli,
        "load_bundle_path",
        lambda *_args, **_kwargs: SimpleNamespace(
            source_commit="a" * 40,
            raw_bundle_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(sync_cli, "synchronize", fake_synchronize)
    monkeypatch.setattr(sync_cli, "init_http_client", lambda: None)

    async def close_http_client():
        return None

    monkeypatch.setattr(sync_cli, "close_http_client", close_http_client)

    await sync_cli._run(arguments)

    assert captured == {"min_size": 1, "max_size": 2}
