import json
import os
import subprocess
from pathlib import Path

type JsonValue = (
    dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_COMPOSE_ENVIRONMENT = {
    "API_KEY": "compose-test-api-key",
    "AUTH_SECRET": "compose-test-auth-secret",
    "AUTH_URL": "http://localhost:3000",
    "CREDENTIAL_ENCRYPTION_KEY": "compose-test-credential-key",
    "GPU_API_URL": "http://gpu-api:8888",
    "LOG_LEVEL": "info",
    "PGADMIN_EMAIL": "compose-test@example.invalid",
    "PGADMIN_PASSWORD": "compose-test-password",
    "POSTGRES_DB": "compose_test",
    "POSTGRES_PASSWORD": "compose-test-password",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "compose_test",
    "TZ": "UTC",
    "WORKERS": "1",
}


def _compose_object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AssertionError(f"Expected {label} to be a JSON object")
    return value


def _render_compose(
    compose_file: str,
    extra_file: Path | None = None,
) -> dict[str, JsonValue]:
    command = ["docker", "compose", "-f", compose_file]
    if extra_file is not None:
        command.extend(("-f", str(extra_file)))
    command.extend(("config", "--format", "json"))

    environment = {"PATH": os.environ["PATH"], **_COMPOSE_ENVIRONMENT}
    result = subprocess.run(  # noqa: S603
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    rendered: dict[str, JsonValue] = json.loads(result.stdout)
    return _compose_object(rendered, "Compose configuration")


def _assert_embedding_worker_contract(
    compose_file: str,
    *,
    extra_file: Path | None = None,
) -> None:
    rendered = _render_compose(compose_file, extra_file)
    services = _compose_object(rendered["services"], "services")
    api = _compose_object(services["api"], "api service")
    worker = _compose_object(services["embedding-worker"], "embedding worker service")

    assert worker["command"] == ["python", "-m", "app.embedding_worker"]
    depends_on = _compose_object(worker["depends_on"], "worker dependencies")
    database_dependency = _compose_object(depends_on["db"], "database dependency")
    assert database_dependency["condition"] == "service_healthy"
    assert worker["restart"] == "unless-stopped"
    assert worker["healthcheck"] == {"disable": True}
    assert "ports" not in worker
    assert "volumes" not in worker
    assert "working_dir" not in worker
    assert worker["networks"] == api["networks"]
    assert worker.get("pull_policy") == api.get("pull_policy")

    api_environment = _compose_object(api["environment"], "API environment")
    worker_environment = _compose_object(worker["environment"], "worker environment")
    expected_environment_keys = {
        "DATABASE_POOL_MAX_SIZE",
        "DATABASE_POOL_MIN_SIZE",
        "DATABASE_URL",
        "GPU_API_URL",
        "LOG_LEVEL",
        "TZ",
    }
    assert set(worker_environment) == expected_environment_keys
    assert {key: worker_environment[key] for key in expected_environment_keys} == {
        key: api_environment[key] for key in expected_environment_keys
    }

    if compose_file == "compose.yaml":
        api_build = _compose_object(api["build"], "API build")
        worker_build = _compose_object(worker["build"], "worker build")
        assert worker_build == api_build
        assert isinstance(worker_build["context"], str)
        assert worker_build["context"].endswith("/api")
        assert worker_build["dockerfile"] == "Dockerfile"
        assert worker["image"] == api["image"] == "finki-hub/chat-bot-api:latest"
        assert worker["pull_policy"] == api["pull_policy"] == "never"
    else:
        assert "build" not in worker
        assert (
            worker["image"] == api["image"] == ("ghcr.io/finki-hub/chat-bot-api:latest")
        )


def test_compose_defines_embedding_worker_runtime_contract():
    override = os.environ.get("COMPOSE_TEST_OVERRIDE")
    extra_file = Path(override) if override else None

    for compose_file in ("compose.yaml", "compose.prod.yaml"):
        _assert_embedding_worker_contract(compose_file, extra_file=extra_file)
