import os

host = os.getenv("HOST", "0.0.0.0")  # noqa: S104
port = os.getenv("PORT", "8880")
bind = f"{host}:{port}"

workers = int(os.getenv("WORKERS", "4"))

worker_class = "uvicorn.workers.UvicornWorker"

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")


def post_fork(server: object, worker: object) -> None:
    # Build the PostHog client AFTER fork — its flush thread does not survive os.fork().
    from app.utils.posthog_client import init_posthog  # noqa: PLC0415
    from app.utils.settings import Settings  # noqa: PLC0415

    init_posthog(Settings())


def worker_exit(server: object, worker: object) -> None:
    from app.utils.posthog_client import shutdown_posthog  # noqa: PLC0415

    shutdown_posthog()
