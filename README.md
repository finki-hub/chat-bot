# FINKI Hub / Chat Bot

RAG chat bot for the [`FINKI Hub`](https://discord.gg/finki-studenti-810997107376914444) Discord server, powered by [LangChain](https://github.com/langchain-ai/langchain) and [FastAPI](https://github.com/fastapi/fastapi). Uses [PostgreSQL](https://github.com/postgres/postgres) and [pgvector](https://github.com/pgvector/pgvector) for keeping documents. Has support for many LLMs.

It currently answers questions using a retrieval pipeline over an FAQ dataset (the `question` table), and separately manages a collection of links. Retrieval over additional document types is planned.

## Services

This project comes as a monorepo of microservices:

- API ([`/api`](/api)) for managing documents, links and chatting (default port: 8880)
- GPU API ([`/gpu-api`](/gpu-api)) for locally executing GPU accelerated tasks like embeddings generation and reranking (default port: 8888)
- Front-end (planned — not yet part of this repository)
- Database (PostgreSQL + pgvector) for keeping documents and embeddings

The API Docker image is available as [`ghcr.io/finki-hub/chat-bot-api`](https://github.com/finki-hub/chat-bot/pkgs/container/chat-bot-api), while the GPU API Docker image is available as [`ghcr.io/finki-hub/chat-bot-gpu-api`](https://github.com/finki-hub/chat-bot/pkgs/container/chat-bot-gpu-api).

## Quick Setup (Production)

It's highly recommended to do this in Docker.

To run the chat bot:

1. Download [`compose.prod.yaml`](./compose.prod.yaml)
2. Download [`.env.sample`](.env.sample), rename it to `.env` and change it to your liking
3. Run `docker compose -f compose.prod.yaml up -d`

The API will be running on port `8880`. This also brings up a `pgAdmin` instance. You may use it to view or create documents. It's accesible on port `5555` by default.

## Quick Setup (Development)

Requires Python 3.14 (`>=3.14,<3.15`) and [`uv`](https://github.com/astral-sh/uv).

1. Clone the repository: `git clone https://github.com/finki-hub/chat-bot.git`
2. Install dependencies: in each directory (`api` and `gpu-api`), run `uv sync`
3. Prepare env. variables by copying `.env.sample` to `.env` - minimum setup requires the database configuration, it can be left as is
4. Run it: `docker compose up -d`. Unlike production, the dev compose builds the `api` and `gpu-api` images locally from source (it does not pull from ghcr), so the first run builds the containers. The per-directory `uv sync` from step 2 is for local/IDE tooling only — the containers build their own environment.

This also brings up the FastAPI Swagger UI (OpenAPI docs) at `localhost:8880/docs`.

## Endpoints

This is an incomplete list. You may view all available endpoints on the OpenAPI documentation (`/docs`).

API (`/api`):

- `/questions/list` - get all questions
- `/questions/name/<name>` - get a question by its name
- `/questions/fill` - generate (fill) embeddings for stored questions for a given model (streams progress via SSE)
- `/links/list` - get all links
- `/chat` - chat with the bot (streaming response); `/chat/models` lists available chat models
- `/health` - detailed health check

GPU API (`/gpu-api`):

- `/embeddings/embed` - generate embedding vectors for given input text(s)
- `/rerank` - re-rank documents by relevance to a query

## License

This project is licensed under the terms of the MIT license.
