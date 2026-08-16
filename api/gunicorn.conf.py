import os

host = os.getenv("HOST", "0.0.0.0")  # ruff: ignore[S104] -- container ingress requires all interfaces
port = os.getenv("PORT", "8880")
bind = f"{host}:{port}"

workers = int(os.getenv("WORKERS", "4"))

worker_class = "uvicorn.workers.UvicornWorker"

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")


def post_fork(_server: object, _worker: object) -> None:
    # Build the PostHog client AFTER fork — its flush thread does not survive os.fork().
    from app.utils.posthog_client import init_posthog  # ruff: ignore[I001, PLC0415] -- client must initialize after fork
    from app.utils.settings import Settings  # ruff: ignore[PLC0415] -- settings are read in the worker process

    init_posthog(Settings())


def worker_exit(_server: object, _worker: object) -> None:
    from app.utils.posthog_client import shutdown_posthog  # ruff: ignore[I001, PLC0415] -- hook runs in the worker process

    shutdown_posthog()
