# FINKI Hub Web Chat

A polished, claude.ai / ChatGPT-style web chat for the FINKI Hub chat API. It streams
answers with the full agent UX (live tokens + a "searching…" tool indicator), keeps
multiple conversations locally, lets the user pick the model, renders Markdown, and
supports like/dislike feedback. Chrome is Macedonian; answers are Macedonian Cyrillic.

## Architecture

```
Browser (Next.js client, React 19)
  useChat + AI Elements UI, Dexie (IndexedDB) history, Zustand UI state
        │  same-origin fetch — /api/*  (never sees API_BASE_URL or the api key)
Next.js Route Handlers (BFF, server-only)
  POST /api/chat      → protocol-v2 SSE → AI SDK UI-message-stream
  POST /api/feedback  → inject x-api-key, proxy to /chat/feedback
  GET  /api/models    → proxy + cache
        │  server fetch (API_BASE_URL, x-api-key)
Python chat API (protocol-v2)  POST /chat/  ·  GET /chat/models  ·  POST /chat/feedback
```

The **BFF** is the only component that knows `API_BASE_URL` and the master `x-api-key`.
The browser talks exclusively to same-origin `/api/*`, so the key never reaches the
client and there are no `NEXT_PUBLIC_*` vars.

## Stack

- **Next.js 15** (App Router) · **React 19** · **TypeScript 6**
- **AI SDK v5** — `ai@^5` + `@ai-sdk/react@^2` (the line that pairs with `ai@5`; there is
  no `@ai-sdk/react@5`). The BFF builds the stream with `createUIMessageStream`; the client
  consumes it with `useChat`.
- **AI Elements** (shadcn registry, vendored into `components/ai-elements/`) + **Streamdown** v2
- **Tailwind CSS v4** (+ `tw-animate-css`), shadcn primitives (`components/ui/`, slate base)
- **Dexie** (IndexedDB) for local history · **TanStack Query** (models) · **Zustand** (UI state)
- **Tooling (finki-hub org standard):** npm, `eslint-config-imperium` (perfectionist +
  react + jsx-a11y + vitest + prettier), `.editorconfig`, ultra-strict `tsconfig`.
- **Tests:** Vitest + React Testing Library + `fake-indexeddb` (unit/component) · Playwright (e2e).

## Prerequisites

- Node.js `^24 || ^26` and npm (ships with Node — no pnpm/yarn).
- A reachable **protocol-v2** FINKI Hub chat API (the parser also tolerates the legacy
  bare-`data:` stream during a rollout window).

## Setup

```bash
cd web
npm install
npm run e2e:install   # one-time: download the Playwright Chromium browser
```

Create `web/.env.local` (server-only — never exposed to the browser):

```bash
# Base URL of the Python chat API (protocol-v2). No /api prefix; /chat/ has a trailing slash.
API_BASE_URL=http://localhost:8880
# Master x-api-key for POST /chat/feedback. Injected by the BFF; never sent to the browser.
CHAT_API_KEY=replace-with-the-master-api-key
```

`lib/env.ts` is `import 'server-only'` and reads these at request time; the build and dev
server need the file present (placeholder values are fine for building).

## Commands

| Command | What it does |
|---|---|
| `npm run dev` | Run the app at http://localhost:3000 (needs the env vars). |
| `npm run build` | Production build (`next build`). |
| `npm start` | Serve the production build. |
| `npm run typecheck` | `tsc --noEmit` against the strict config. |
| `npm test` | Vitest unit/component suite (jsdom + fake-indexeddb). |
| `npm run lint` | ESLint (`eslint-config-imperium`). |
| `npm run format` | ESLint `--fix` (sorting + style). |
| `npm run e2e` | Playwright e2e (mocked BFF; no live API needed). |
| `npm run e2e:install` | One-time Playwright browser download. |

## Project layout

```
web/
  app/
    layout.tsx · providers.tsx · page.tsx     # client chat shell (sidebar + thread + composer)
    api/chat|feedback|models/route.ts         # the BFF
  components/
    ai-elements/   # vendored AI Elements (conversation, message, prompt-input)
    ui/            # vendored shadcn primitives
    chat/          # thread, message, search-status, composer, answer-actions
    shell/         # sidebar, conversation-list
  lib/
    api-types.ts · env.ts · sse.ts · chat-translate.ts · transport.ts
    db.ts · user.ts · messages.ts · use-models.ts · ui-store.ts · i18n.ts · utils.ts
  e2e/             # Playwright spec + SSE helpers
  test/            # Vitest suites
```

## How the streaming pipeline works

`useChat` POSTs the conversation to `/api/chat`. The BFF maps it to the Python `ChatSchema`
(oldest-first, last-is-user, ≤50 msgs / ≤8000 chars/turn; camelCase → snake_case), calls
`POST {API_BASE_URL}/chat/`, reads `X-Response-Id`, parses the protocol-v2 named SSE events
(`token`/`status`/`reset`/`error`/`done`), and re-emits them as an AI SDK UI-message-stream:
`text-delta` for tokens, transient `data-status`/`data-error` parts, and `reset` ends the
current text part and starts a new one (preamble drop — the client renders only the last
text part). Conversation history lives client-side in Dexie in `UIMessage` shape.

## Deferred (not in v1)

Auth/login, rate-limiting/quotas, server-side conversation storage / cross-device sync,
FAQ-links browse, the recommender UI, attachments/voice. The BFF seam, the local store, and
the modular shell are designed so these can be added without a rewrite. A Docker image +
`compose.yaml` wiring is a follow-up.
