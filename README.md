# FINKI Hub / Chat Bot

RAG chat bot and web front-end for the [`FINKI Hub`](https://discord.gg/finki-studenti-810997107376914444) Discord server, powered by [FastAPI](https://github.com/fastapi/fastapi), [LangChain](https://github.com/langchain-ai/langchain), and [Next.js](https://github.com/vercel/next.js). It uses [PostgreSQL](https://github.com/postgres/postgres) with [pgvector](https://github.com/pgvector/pgvector) for storage and vector search, supports hosted and Ollama LLM providers, and uses the GPU service for embeddings and reranking.

It answers questions using a retrieval pipeline over an FAQ dataset (the `question` table) and over chunked source-of-truth documents (the `document` / `chunk` tables — laws, rulebooks, procedures), retrieved together in a single reranked pass. It also manages links, chat feedback, diplomas, professor publications/groups, and thesis committee recommendations.

## Services

This project comes as a monorepo of microservices:

- API ([`/api`](/api)) for managing questions, documents, links, diplomas, recommendations, feedback, and chat (default port: 8880)
- GPU API ([`/gpu-api`](/gpu-api)) for locally executing GPU-accelerated embeddings and reranking (default port: 8888)
- Web ([`/web`](/web)) for the chat front-end — Next.js with a thin BFF (default port: 3000)
- Database (PostgreSQL + pgvector) for keeping questions, links, documents/chunks, diplomas, professor data, feedback, and embeddings

The Docker images are available as [`ghcr.io/finki-hub/chat-bot-api`](https://github.com/finki-hub/chat-bot/pkgs/container/chat-bot-api), [`ghcr.io/finki-hub/chat-bot-gpu-api`](https://github.com/finki-hub/chat-bot/pkgs/container/chat-bot-gpu-api) and [`ghcr.io/finki-hub/chat-bot-web`](https://github.com/finki-hub/chat-bot/pkgs/container/chat-bot-web).

## Quick Setup (Production)

It's highly recommended to do this in Docker.

To run the chat bot:

1. Download [`compose.prod.yaml`](./compose.prod.yaml)
2. Download [`.env.sample`](.env.sample), rename it to `.env`, and set the required values. At minimum, set a non-default `API_KEY` before exposing the service. Sponsored Luna remains disabled unless its separate rollout settings are deliberately configured; never use `API_KEY` as its provider credential. If you configure MCP servers, set non-default per-server `api_key` values in `MCP_SERVERS`.
3. Run `docker compose -f compose.prod.yaml up -d`

The API runs on port `8880`, the GPU API on `8888`, and the web front-end on `3000`. This also brings up a `pgAdmin` instance on port `5555` by default.

## Quick Setup (Development)

Requires Python 3.14 (`>=3.14,<3.15`) and [`uv`](https://github.com/astral-sh/uv) for local API/GPU API tooling. The web app requires Node `^24 || ^26`.

1. Clone the repository: `git clone https://github.com/finki-hub/chat-bot.git`
2. Install dependencies: in each directory (`api` and `gpu-api`), run `uv sync`
3. Prepare env. variables by copying `.env.sample` to `.env`. The sample database values work for local Docker, but set `API_KEY` to use authenticated write/feedback endpoints. OpenAI, Google, Anthropic, and Ollama models use per-user credentials saved from the web settings dialog; sponsored Luna is a separate, disabled-by-default server feature.
4. Run it: `docker compose up -d`. Unlike production, the dev compose builds the `api`, `gpu-api` and `web` images locally from source (it does not pull from ghcr), so the first run builds the containers. The per-directory `uv sync` from step 2 is for local/IDE tooling only — the containers build their own environment.

This also brings up the API Swagger UI (OpenAPI docs) at `localhost:8880/docs`, the GPU API docs at `localhost:8888/docs`, and pgAdmin at `localhost:5550`.

The web front-end ([`/web`](/web)) runs as the `web` service (Next.js + BFF) on port `3000`; `docker compose up -d` builds and starts it with the rest of the stack. In the container it reaches the API at `http://api:8880` and reuses `API_KEY` as its `CHAT_API_KEY`.

To run the web app standalone for local development:

```bash
cd web
npm install
npm run dev   # serves http://localhost:3000
```

Standalone, it needs `web/.env.local` with `API_BASE_URL` (the chat API base, e.g. `http://localhost:8880`), `CHAT_API_KEY` (the master `x-api-key`, used server-side by the BFF), `RESUMABLE_STREAM_REDIS_URL`, `AUTH_URL`, `AUTH_SECRET`, and at least one Auth.js OAuth provider: Google (`AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`) or Microsoft Entra ID (`AUTH_MICROSOFT_ENTRA_ID_ID`, `AUTH_MICROSOFT_ENTRA_ID_SECRET`, `AUTH_MICROSOFT_ENTRA_ID_ISSUER`).

## Configuration

The root [`.env.sample`](.env.sample) contains the main variables used by the Docker stacks:

- `API_KEY` - required for authenticated API writes, embedding fill jobs, diploma sync, and feedback submission; change the sample value before deployment. This is the chat API/BFF authentication secret, not a sponsored provider key.
- `AUTH_URL`, `AUTH_SECRET`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `AUTH_MICROSOFT_ENTRA_ID_ID`, `AUTH_MICROSOFT_ENTRA_ID_SECRET`, `AUTH_MICROSOFT_ENTRA_ID_ISSUER` - used by the web BFF for Auth.js login; configure Google, Microsoft Entra ID, or both. For Microsoft, use `https://login.microsoftonline.com/common/v2.0` to allow personal, work, and school accounts, or `https://login.microsoftonline.com/<tenant-id>/v2.0` to restrict logins to one tenant.
- `MCP_SERVERS` - optional JSON array of named MCP tool servers. Each entry supports `name`, `url`, `transport` (`streamable_http` or `sse`), optional per-server `api_key`, and optional `allowed_tools` / `blocked_tools` lists for tool exposure control. Existing `MCP_HTTP_URLS`, `MCP_SSE_URLS`, and `MCP_API_KEY` values are still forwarded by the compose files for compatibility, but new deployments should use `MCP_SERVERS`
- `CREDENTIAL_ENCRYPTION_KEY` - required secret used to encrypt per-user BYOK provider keys at rest; rotate only with a re-encryption plan for stored credentials
- `BYOK_ALLOWED_BASE_URLS` - optional comma-separated allowlist of HTTPS provider endpoints users may select with their own credentials
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` - used by the database service and by the API `DATABASE_URL`
- `DATABASE_POOL_MIN_SIZE`, `DATABASE_POOL_MAX_SIZE` - asyncpg pool sizing per API worker
- `GPU_API_URL` - API-to-GPU-API base URL; Docker defaults to `http://gpu-api:8888`
- OpenAI, Google, Anthropic, and Ollama models are BYOK-only and do not use deployment credentials, except for the explicitly isolated sponsored Luna path below. Ollama defaults to `https://ollama.com`; custom Ollama endpoints must be HTTPS and allowed by `BYOK_ALLOWED_BASE_URLS`.
- `RESUMABLE_STREAM_REDIS_URL` - server-only Redis/Valkey URL used by the web BFF for resumable chat streams
- `RERANKER_MIN_SCORE`, `SOURCE_RERANKER_MIN_SCORE`, `CHAT_HISTORY_MAX_TURNS` - retrieval and chat tuning

The API owns the executable chat catalog. `/chat/models` returns the fixed, ordered
allowlist enriched with display metadata from models.dev. Metadata is cached in process
for six hours; refresh failures serve the last successful catalog, and cold-start
failures use the bundled snapshot. Remote metadata cannot add executable model IDs,
change providers or alter execution policy.

The hosted catalog includes GPT-5.6 Sol, Terra, and Luna, GPT-5.5, and the current
Gemini 3.5 Flash, Gemini 3.1 Pro Preview, and Gemini 3.1 Flash Lite models. The
Ollama catalog uses multilingual Qwen3 30B and 14B quantizations selected to fit a
24 GB GPU; the advertised context limits still require enough remaining memory for
the KV cache.

`BAAI/bge-m3`, `text-embedding-3-large`, and `gemini-embedding-001` are active for authenticated chat requests. Corpus fill jobs remain limited to local BGE-M3 because they do not have a user credential boundary. Legacy embedding columns and historical model values remain readable for existing data.

The compose files also support optional variables that are not listed in `.env.sample` because they have built-in defaults: `POSTHOG_KEY`, `POSTHOG_HOST`, `RERANKER_MODEL`, and `WEB_API_BASE_URL`.

The standalone web app reads `API_BASE_URL`, `CHAT_API_KEY`, `RESUMABLE_STREAM_REDIS_URL`, `AUTH_URL`, `AUTH_SECRET`, and any configured OAuth provider credentials from `web/.env.local`. Optional web-facing variables include `SITE_URL`, `NEXT_PUBLIC_POSTHOG_KEY`, and `NEXT_PUBLIC_POSTHOG_HOST`.

### Sponsored Luna rollout

Sponsored Luna is an API-only, server-side fallback for users who do not have a
personal OpenAI credential. A user's BYOK credential always wins and does not
consume sponsored quota. The feature is disabled by default; when disabled, no
sponsored provider request or quota admission is attempted and BYOK behavior is
unchanged.

| Setting | Default | Meaning and security requirement |
| --- | --- | --- |
| `SPONSORED_LUNA_ENABLED` | `false` | Kill switch. Keep `false` until the release gate below passes. |
| `SPONSORED_OPENAI_API_KEY` | empty | Dedicated provider secret for sponsored inference. It must not be copied from or substituted with `API_KEY`, a user BYOK key, or a Discord secret. |
| `SPONSORED_OPENAI_BASE_URL` | empty | Optional dedicated HTTPS OpenAI-compatible endpoint for sponsored inference only. It is canonicalized and validated separately from `BYOK_ALLOWED_BASE_URLS`; it must not be used as a user endpoint allowlist. |
| `SPONSORED_LUNA_UPSTREAM_MODEL` | `gpt-5.6-luna` | Upstream model name sent to the provider; it may differ from the public catalog ID. |
| `SPONSORED_DAILY_USER_LIMIT` | `5` | Maximum five sponsored requests per user per UTC calendar day by default. BYOK requests are not counted. |
| `SPONSORED_DAILY_GLOBAL_LIMIT` | empty | Required positive global sponsored-request cap when enabled. Leaving it empty is fail-closed and is valid only while the feature is disabled. |
| `SPONSORED_MAX_OUTPUT_TOKENS` | `1024` | Sponsored output-token request; runtime hard-clamps it to 1024 even if a larger value is supplied. |
| `SPONSORED_REQUEST_LEASE_SECONDS` | `600` | Active-request lease duration. A second concurrent sponsored request for the same user is rejected until the lease expires or is released. |

The user and global counters use UTC dates and reset at the next UTC midnight.
The sponsored OpenAI client is streaming-enabled and requests usage metadata;
provider keys, URLs, prompts, and raw provider errors must never be logged or
forwarded to clients. Only the API service receives the `SPONSORED_*` variables;
the web container and Discord bot do not need them.

#### Live enablement preflight (release gate)

Run this on the deployment host with real values loaded in `.env`. It validates
the separate key, HTTPS Base URL, upstream model, positive global cap, compose
rendering, and one uncached streaming request. It is intentionally fail-closed:

```bash
set -eu
set -a
. ./.env
set +a
test "${SPONSORED_LUNA_ENABLED:-false}" = "true"
test -n "${SPONSORED_OPENAI_API_KEY:-}"
test -n "${SPONSORED_OPENAI_BASE_URL:-}"
test -n "${SPONSORED_LUNA_UPSTREAM_MODEL:-}"
test "${SPONSORED_DAILY_GLOBAL_LIMIT:-0}" -gt 0
case "${SPONSORED_OPENAI_BASE_URL}" in https://*) ;; *) exit 1 ;; esac
docker compose --env-file .env -f compose.prod.yaml config --quiet
curl --fail-with-body --no-buffer --silent --show-error \
  -H "x-api-key: ${API_KEY}" \
  -H 'content-type: application/json' \
  --data '{"user_id":"00000000-0000-4000-8000-000000000016","interface":"web","inference_model":"gpt-5.6-luna","query_transform_mode":"raw","messages":[{"role":"user","content":"deployment preflight"}],"max_tokens":1024}' \
  "${CHATBOT_URL:-http://127.0.0.1:8880}/chat/" | tee /tmp/sponsored-luna-preflight.sse
/bin/sh scripts/verify-sponsored-luna-preflight.sh /tmp/sponsored-luna-preflight.sse
```

Do not run that command with placeholder credentials and do not report live
enablement as successful without the real provider response. If real provider
credentials are unavailable, keep `SPONSORED_LUNA_ENABLED=false`; the command
above remains a release gate, not a claimed live test.

## Local Checks

API:

```bash
cd api
uv sync
uv run pytest
uv run ruff check .
uv run mypy .
```

GPU API:

```bash
cd gpu-api
uv sync
uv run pytest
uv run ruff check .
uv run mypy .
```

Web:

```bash
cd web
npm install
npm run check
npm run lint
npm run test
npm run e2e:install
npm run e2e
```

The API container entrypoint is [`api/start.sh`](api/start.sh), which runs migrations and then starts `gunicorn -c gunicorn.conf.py app.main:app`. The GPU API container starts the same FastAPI/Gunicorn entrypoint directly from [`gpu-api/gunicorn.conf.py`](gpu-api/gunicorn.conf.py). The web container runs the Next.js standalone server built by [`web/Dockerfile`](web/Dockerfile).

Database migrations live in [`api/resources/migrations`](api/resources/migrations) as ordered `.sql` files. Add new schema changes as the next numbered migration instead of editing an already-applied migration.

## Endpoints

This is an incomplete list. You may view all available endpoints on the OpenAPI documentation (`/docs`).

API service (`/api` directory, default port `8880`):

- `/questions/list`, `/questions/names`, `/questions/name/{name}`, `/questions/closest`, `/questions/nth/{n}`, `/questions/unfilled` - read and search stored FAQ questions
- `/questions/` (POST), `/questions/{name}` (PUT/DELETE), `/questions/fill` - manage questions and fill question embeddings (write/fill endpoints require `x-api-key`)
- `/documents/list`, `/documents/name/{name}` - list or fetch source-of-truth documents with chunk counts
- `/documents/` (POST), `/documents/{name}` (DELETE), `/documents/fill` - ingest/replace/delete Markdown documents and fill chunk embeddings (requires `x-api-key`)
- `/links/list`, `/links/names`, `/links/name/{name}`, `/links/nth/{n}` - read stored links
- `/links/` (POST), `/links/{name}` (PUT/DELETE) - manage links (requires `x-api-key`)
- `/diplomas/sync`, `/diplomas/fill-embeddings` - sync defended thesis records from the upstream Diplomas API and fill their embeddings (requires `x-api-key`)
- `/recommendations/` - recommend thesis committee alternatives from historical defenses and professor-paper expertise
- `/groups/` - list precomputed professor groups by source, year, or professor
- `/chat/` - stream chat responses; `/chat/models` lists available chat models; `/chat/feedback` records web/Discord feedback (feedback requires `x-api-key`)
- `/health/` - liveness check; `/health/health` - detailed database health check

GPU API service (`/gpu-api` directory, default port `8888`):

- `/embeddings/embed` - generate embedding vectors for input text(s)
- `/rerank/` - re-rank documents by relevance to a query
- `/health/` - liveness check; `/health/health` - detailed GPU API health check

Web BFF (`/web`, default port `3000`):

- `/api/chat` - browser-facing chat stream proxy
- `/api/models` - browser-facing model list proxy
- `/api/health` - browser-facing API health probe
- `/api/feedback` - browser-facing feedback proxy

## License

This project is licensed under the terms of the MIT license.
