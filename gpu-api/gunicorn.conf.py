import os

host = os.getenv("HOST", "0.0.0.0")  # ruff: ignore[S104] -- container ingress requires all interfaces
port = os.getenv("PORT", "8888")
bind = f"{host}:{port}"

workers = 1

timeout = 300

# uvicorn.workers is deprecated; uvicorn-worker is the maintained package.
worker_class = "uvicorn_worker.UvicornWorker"

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
