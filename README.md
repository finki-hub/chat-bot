# FINKI Hub / Chat Bot

RAG chat bot and web front-end for the [`FINKI Hub`](https://discord.gg/finki-studenti-810997107376914444) Discord server, powered by [FastAPI](https://github.com/fastapi/fastapi), [LangChain](https://github.com/langchain-ai/langchain), and Next.js. It uses [PostgreSQL](https://github.com/postgres/postgres) with [pgvector](https://github.com/pgvector/pgvector) for storage and vector search, and supports multiple LLM providers plus self-hosted GPU-backed models.

It answers questions using a retrieval pipeline over an FAQ dataset (the `question` table) and over chunked source-of-truth documents (the `document` / `chunk` tables — laws, rulebooks, procedures), retrieved together in a single reranked pass. It also manages links, chat feedback, diplomas, professor publications/groups, and thesis committee recommendations.

## Services

This project comes as a monorepo of microservices:

- API ([`/api`](/api)) for managing questions, documents, links, diplomas, recommendations, feedback, and chat (default port: 8880)
- GPU API ([`/gpu-api`](/gpu-api)) for locally executing GPU-accelerated tasks like embeddings generation, reranking, and self-hosted model streaming (default port: 8888)
- Web ([`/web`](/web)) for the chat front-end — Next.js with a thin BFF (default port: 3000)
- Database (PostgreSQL + pgvector) for keeping questions, links, documents/chunks, diplomas, professor data, feedback, and embeddings

The Docker images are available as [`ghcr.io/finki-hub/chat-bot-api`](https://github.com/finki-hub/chat-bot/pkgs/container/chat-bot-api), [`ghcr.io/finki-hub/chat-bot-gpu-api`](https://github.com/finki-hub/chat-bot/pkgs/container/chat-bot-gpu-api) and [`ghcr.io/finki-hub/chat-bot-web`](https://github.com/finki-hub/chat-bot/pkgs/container/chat-bot-web).

## Quick Setup (Production)

It's highly recommended to do this in Docker.

To run the chat bot:

1. Download [`compose.prod.yaml`](./compose.prod.yaml)
2. Download [`.env.sample`](.env.sample), rename it to `.env`, and set the required values. At minimum, set non-default `API_KEY` and `MCP_API_KEY` values before exposing the service.
3. Run `docker compose -f compose.prod.yaml up -d`

The API runs on port `8880`, the GPU API on `8888`, and the web front-end on `3000`. This also brings up a `pgAdmin` instance on port `5555` by default.

## Quick Setup (Development)

Requires Python 3.14 (`>=3.14,<3.15`) and [`uv`](https://github.com/astral-sh/uv) for local API/GPU API tooling. The web app requires Node `^24 || ^26`.

1. Clone the repository: `git clone https://github.com/finki-hub/chat-bot.git`
2. Install dependencies: in each directory (`api` and `gpu-api`), run `uv sync`
3. Prepare env. variables by copying `.env.sample` to `.env`. The sample database values work for local Docker, but set `API_KEY` to use authenticated write/feedback endpoints and set provider keys for whichever LLMs you want to use.
4. Run it: `docker compose up -d`. Unlike production, the dev compose builds the `api`, `gpu-api` and `web` images locally from source (it does not pull from ghcr), so the first run builds the containers. The per-directory `uv sync` from step 2 is for local/IDE tooling only — the containers build their own environment.

This also brings up the API Swagger UI (OpenAPI docs) at `localhost:8880/docs`, the GPU API docs at `localhost:8888/docs`, and pgAdmin at `localhost:5550`.

The web front-end ([`/web`](/web)) runs as the `web` service (Next.js + BFF) on port `3000`; `docker compose up -d` builds and starts it with the rest of the stack. In the container it reaches the API at `http://api:8880` and reuses `API_KEY` as its `CHAT_API_KEY`.

To run the web app standalone for local development:

```bash
cd web
npm install
npm run dev   # serves http://localhost:3000
```

Standalone, it needs `web/.env.local` with `API_BASE_URL` (the chat API base, e.g. `http://localhost:8880`) and `CHAT_API_KEY` (the master `x-api-key`, used server-side by the BFF for feedback submission).

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
- `/stream/` - stream a chat response from a self-hosted model
- `/health/` - liveness check; `/health/health` - detailed GPU API health check

Web BFF (`/web`, default port `3000`):

- `/api/chat` - browser-facing chat stream proxy
- `/api/models` - browser-facing model list proxy
- `/api/health` - browser-facing API health probe
- `/api/feedback` - browser-facing feedback proxy

## License

This project is licensed under the terms of the MIT license.
