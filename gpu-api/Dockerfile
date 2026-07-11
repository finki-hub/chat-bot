FROM ghcr.io/astral-sh/uv:python3.14-bookworm AS builder

ENV UV_VENV_PATH=/app/.venv \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d \( -name "tests" -o -name "test" \) -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /app/.venv/lib/python3.14/site-packages/torch/test 2>/dev/null || true

FROM python:3.14-slim-bookworm AS final

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=builder /app/gunicorn.conf.py /app/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=3s CMD wget --quiet --tries=1 --spider http://127.0.0.1:8888/health/ || exit 1

EXPOSE 8888

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]
