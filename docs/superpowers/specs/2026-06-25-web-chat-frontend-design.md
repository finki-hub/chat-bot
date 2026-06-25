# Web Chat Frontend — Design Spec

- **Date:** 2026-06-25
- **Status:** Approved (design); pending spec review → implementation plan
- **Location:** new `web/` app at the repo root
- **Author:** drafted with Claude Code (brainstorming)

## 1. Goal

A polished, claude.ai / ChatGPT-style chat web app that talks to the existing FINKI Hub chat API. It streams answers with the full agent UX (live tokens + a "searching…" tool indicator), keeps multiple conversations locally, lets the user pick the model, renders Markdown answers, and supports like/dislike feedback.

### In scope (v1)

- Streaming chat against `POST /chat/`, consuming the **protocol-v2** SSE events (token / status / reset / error / done).
- Multiple conversations with local (on-device) history; new / rename / delete; sidebar navigation.
- Model picker populated from `GET /chat/models`.
- Markdown rendering of answers (bare URLs autolinked; no tables — matches the bot's output contract).
- Per-answer actions: copy, regenerate, and **like/dislike** feedback.
- A thin **BFF** (Next.js Route Handlers) that holds the API key, normalizes the stream, and is the future seam for auth/rate-limiting.

### Out of scope (deferred, but designed not to block)

- Authentication / login / accounts.
- Rate-limiting / quota / abuse protection (access is "open, trusted/internal" for the draft).
- Server-side conversation storage / cross-device sync.
- FAQ/links browse UI and the diploma-recommender UI.
- Attachments / file upload, voice, multi-modal.

## 2. Locked decisions

| Decision | Choice | Why |
|---|---|---|
| v1 surface | **Chat only** | Fastest path to a working, polished product. |
| Client topology | **Next.js app + thin BFF** | BFF holds `x-api-key` (feedback), same-origin (no CORS, hides API URL), future auth/rate-limit seam, and translates the SSE. |
| Access / abuse | **Open, no limits (trusted/internal)** | Acceptable for the draft phase; BFF leaves a stub seam to add limits later. |
| Conversations | **Local only (IndexedDB / Dexie)** | Matches the stateless backend; zero backend work; no accounts. |
| Streaming + UI | **Vercel AI SDK v5 + AI Elements** | `useChat` owns streaming/stop/regenerate/status; AI Elements = shadcn-composable chat components (you own the source). |
| Chrome language | **Macedonian** (default) | Matches the audience and the bot's Cyrillic answers; structured so EN can be added later. |

## 3. Backend API contract (verified)

> Verified against live code. **Important branch note:** protocol-v2 lives on `origin/main` (commit `dd7066f`, PR #475). The current working branch `feat/diploma-recommender` is one commit behind and still has the older **plain-text** stream. The frontend targets **protocol-v2**, which is what will be deployed (rollout pending). The `web/` work should be developed against a protocol-v2 API (merge `main` into the working branch, or point at a deployed protocol-v2 instance). A defensive note for the client is in §10.

### 3.1 Base & routing

- Base URL: `http://<host>:8880`. **No `/api` prefix**, no reverse proxy in compose — routers mount at their bare prefixes (`/chat/`, `/health/`, …). `api/app/main.py`.
- The frontend never calls the Python API directly from the browser; it goes through the **BFF** (§5), so the browser only ever sees same-origin `/api/*` Next.js routes.

### 3.2 `POST /chat/` — the streaming endpoint

- **Method/Path:** `POST /chat/` (trailing slash is canonical; `POST /chat` can 307-redirect and drop the body).
- **Auth:** none. CORS defaults to `*` (credentials forced off). A browser *can* call it directly, but we proxy via the BFF anyway.
- **Request body** (`ChatSchema`, `api/app/schemas/chat.py`):

```jsonc
{
  "messages": [                          // required, 1..50 items, OLDEST-FIRST
    { "role": "user" | "assistant",      // last element MUST be role:"user" → else HTTP 422
      "content": "string" }              // max 8000 chars per turn
  ],
  "system_prompt":         "string|null",// default null → server's FINKI agent prompt
  "embeddings_model":      "Model enum", // default BAAI/bge-m3
  "inference_model":       "Model enum", // default claude-sonnet-4-6 — picks the streaming LLM/provider
  "query_transform_model": "Model enum", // default gpt-5.4-mini
  "temperature": 0.3,                    // 0.0..1.0
  "top_p":       1.0,                    // 0.0..1.0
  "max_tokens":  4096                    // >= 1
}
```

- **Response:** HTTP 200, `Content-Type: text/event-stream`, chunked. **protocol-v2** frames are standard SSE: `event: <name>\ndata: <JSON>\n\n` (`origin/main:api/app/llms/agents.py`).

#### Event table (the core of the client)

| `event:` | `data:` JSON | Fires when | Client effect |
|---|---|---|---|
| `token` | `{"text": "..."}` | each answer fragment | append to the answer text part (streamed) |
| `status` | `{"state":"tool_call","label":"🔍 Пребарувам…","tool":"<name>"}` | agent `on_tool_start` | show a "searching…/tool" chip |
| `reset` | `{}` | once, before the answer, when a tool ran | **drop any preamble** streamed so far; start the answer fresh |
| `error` | `{"code":"no_answer"\|"interrupted"\|"agent_error","message":"..."}` | failure | `interrupted` = keep partial + soft notice; others = error + Retry |
| `done` | `{}` | always last (even after error) | finalize the message |

- **Response id:** returned **only** as the `X-Response-Id` response header (UUIDv4), readable from the first byte. It is the key for feedback. `api/app/api/chat.py`.
- **Cancellation:** no server cancel endpoint — abort the `fetch` (`AbortController`); the server sees the disconnect. `useChat`'s `stop()` handles this.
- **Pre-stream failures are JSON, not SSE:** 422 (validation), 503 (`{"detail":"Failed to retrieve or re-rank context for the query."}`), 500 (`{"detail":"An unexpected internal server error occurred."}`). The BFF must branch on `response.ok` / content-type **before** reading the stream.

### 3.3 `GET /chat/models`

- No auth. Returns a flat, sorted `string[]` of model ids (no display metadata). The client derives provider/family labels from the id.

### 3.4 `POST /chat/feedback`

- **Auth:** requires `x-api-key` (the master key — same one that authorizes all admin writes). **The browser must never hold this**; the BFF injects it server-side.
- **Request body** (`FeedbackSchema`, `api/app/schemas/feedback.py`):

```jsonc
{
  "response_id":   "UUID",               // required — from X-Response-Id
  "client":        "web",                // required literal for this app
  "user_id":       "string (min 1)",     // required — anonymous, stable per-browser id (see §7)
  "feedback_type": "like" | "dislike",   // required
  // optional context, all client-attested:
  "question_text": "string?", "answer_text": "string?",
  "inference_model": "string?", "embeddings_model": "string?", "query_transform_model": "string?",
  "client_ref": "string?", "channel_id": "string?", "guild_id": "string?"
}
```

- Upserts on `(response_id, client, user_id)` → re-submitting flips like↔dislike. Returns `FeedbackAckSchema {id, response_id, feedback_type}`.

### 3.5 Conversation model (decisive)

The backend is **100% stateless** w.r.t. chat history: no sessions/threads/conversations table, no conversation id, no GET to list/resume past chats. The client owns history and **resends the full `messages[]`** (oldest-first, last turn `user`) every request. Caps the client must respect: **≤ 50 messages, ≤ 8000 chars/turn**. (Server-side, only the last `CHAT_HISTORY_MAX_TURNS=10` turns are forwarded to the LLM, but the client may send up to the caps.)

### 3.6 Answer content contract

Answers are **Macedonian Cyrillic**, use **bare URLs only** (no `[text](url)` Markdown links), and **no Markdown tables**. The renderer should autolink bare URLs and does not need table support.

## 4. Architecture

```
┌─────────────────────────────────────────────┐
│ Browser — Next.js client (React 19)          │
│  AI Elements UI + useChat (AI SDK v5)         │
│  Dexie (IndexedDB) for conversation history   │
└───────────────┬─────────────────────────────┘
                │ same-origin fetch (/api/*)
┌───────────────▼─────────────────────────────┐
│ Next.js Route Handlers (BFF, server-only)    │
│  POST /api/chat     → translate SSE           │
│  POST /api/feedback → inject x-api-key, proxy │
│  GET  /api/models   → proxy + cache           │
└───────────────┬─────────────────────────────┘
                │ server fetch (API_BASE_URL, x-api-key)
┌───────────────▼─────────────────────────────┐
│ Python chat API (protocol-v2)  :8880          │
└─────────────────────────────────────────────┘
```

The BFF is the only component that knows `API_BASE_URL` and the `x-api-key`. The browser talks exclusively to same-origin `/api/*`.

## 5. BFF — Next.js Route Handlers

### 5.1 `POST /api/chat` — protocol-v2 → AI SDK UI message stream translator

This is the one non-trivial piece. It:

1. Receives the `useChat` request: `UIMessage[]` plus a `body` carrying `{ model, embeddingsModel, queryTransformModel, temperature, topP, maxTokens }` (sent via `sendMessage(..., { body })` / transport `prepareSendMessagesRequest`).
2. Converts `UIMessage[]` → `ChatSchema.messages`: map each message to `{ role, content }` where `content` is the concatenation of its text parts; enforce oldest-first, last-is-user, and the 50/8000 caps.
3. Calls `POST {API_BASE_URL}/chat/` with the assembled `ChatSchema`.
4. Branches on the response: if not `text/event-stream` (422/503/500 JSON), emit an `error` into the UI stream and finish. Otherwise read `X-Response-Id` from the headers immediately.
5. Builds the UI message stream with `createUIMessageStream` / `createUIMessageStreamResponse` and drives the writer from the parsed protocol-v2 events:

| protocol-v2 | BFF action on the UI message stream |
|---|---|
| (stream start) | attach message metadata `{ responseId, inferenceModel }` |
| `token {text}` | `text-delta` into the current answer text part (lazily `text-start` on first token) |
| `status {label,tool}` | `writer.write({ type:'data-status', data:{label,tool}, transient:true })` |
| `reset {}` | `text-end` the current part, then `text-start` a **new** answer text part (new id) — see §5.2 |
| `error {code,message}` | `writer.write({ type:'data-error', data:{code,message}, transient:true })`; if not `interrupted`, also stop |
| `done {}` | finish the stream |

6. Defines a typed `MyUIMessage` so parts/metadata are type-safe: `metadata: { responseId?: string; inferenceModel?: string }`, data parts `data-status` and `data-error`.

### 5.2 `reset` handling (preamble drop)

`reset` means "discard the pre-tool preamble." The BFF handles it by ending the current answer text part and starting a **new** text part with a fresh id. The client's message renderer displays **only the last text part** of an assistant message, so the preamble part is dropped while live streaming is preserved on every path (including the common no-tool path, which simply has one text part). The `data-status` chip is transient (handled in `onData`, not persisted), so it disappears once the answer streams.

### 5.3 `POST /api/feedback`

Thin proxy. Receives `{ responseId, feedbackType, questionText?, answerText?, inferenceModel? }` from the client; assembles the full `FeedbackSchema` with `client:"web"` and the anonymous `user_id` (passed from the client, see §7); injects `x-api-key`; forwards to `POST {API_BASE_URL}/chat/feedback`; returns the ack. The key is read from a server-only env var and never reaches the browser.

### 5.4 `GET /api/models`

Proxies `GET {API_BASE_URL}/chat/models`, cached briefly (it changes rarely). Returns the `string[]`; the client groups by inferred provider for the picker.

## 6. Client app

- **State / streaming:** `useChat<MyUIMessage>` from `@ai-sdk/react`, configured with a `DefaultChatTransport({ api: '/api/chat', prepareSendMessagesRequest })`. It owns message state, `status`, `stop()`, and `regenerate()`.
- **Transient data parts:** an `onData` handler maps `data-status` → a per-message "searching" indicator and `data-error` → inline error UI (Retry for non-`interrupted`).
- **Components (AI Elements, shadcn registry — vendored into `components/ai-elements/`):**
  - `Conversation` / `ConversationContent` / `ConversationEmptyState` / `ConversationScrollButton`
  - `Message` / `MessageContent` / `MessageResponse` (streaming Markdown via Streamdown)
  - `PromptInput` / `PromptInputTextarea` / `PromptInputSubmit` (streaming/stop status) / `PromptInputSelect` (model picker)
  - A small custom **`SearchStatus`** chip fed by the `data-status` part.
- **Message rendering rule:** for each assistant message, render the **last** text part (preamble drop, §5.2); render the search chip when a `data-status` is active and no answer text has arrived yet; render per-answer actions (copy / regenerate / like-dislike) once `done`.
- **App shell (our own):** collapsible left **sidebar** with the conversation list + new-chat / rename / delete, and the main thread pane. This is the ChatGPT/claude.ai layout; AI Elements provides the thread internals, we provide the shell + persistence wiring.
- **Like/dislike:** calls `POST /api/feedback` with `message.metadata.responseId`; optimistic toggle; hidden when `responseId` is absent for a turn.

## 7. Conversation persistence (Dexie / IndexedDB)

- **Stores:**
  - `conversations`: `{ id, title, model, createdAt, updatedAt }`
  - `messages`: `{ id, conversationId, role, parts, metadata, createdAt }` — stored in AI SDK `UIMessage` shape so `metadata.responseId` survives reloads.
- **Hydration:** on opening a conversation, load its `UIMessage[]` and seed `useChat({ messages })`.
- **Persist:** on `useChat` `onFinish` (and on user turns), upsert messages for the active conversation; bump `updatedAt`; derive `title` from the first user message.
- **Anonymous user id:** mint a stable UUID once, stored in `localStorage` (e.g. `finkiHub.anonUserId`), sent to `/api/feedback` as `user_id` (the backend requires a non-empty `user_id`, namespaced per client).
- **Caps:** when assembling a request, the client trims to ≤ 50 messages / ≤ 8000 chars per turn (the BFF re-validates).

## 8. Markdown rendering

Handled by AI Elements `MessageResponse`, which wraps **Streamdown** (purpose-built for streaming model output: tolerates incomplete Markdown mid-stream, hardened against unsafe content, GFM autolinking). The bot emits bare URLs (autolinked) and never tables, so default config suffices; code blocks render with syntax highlighting. Cyrillic must be covered by the chosen UI font (e.g. Inter).

## 9. Error handling & edge cases

| Case | Behavior |
|---|---|
| Pre-stream JSON error (422/503/500) | BFF emits a `data-error`; client shows an inline error with Retry (no streaming bubble). |
| `error: interrupted` (partial shown) | keep the partial answer; append a soft "одговорот е прекинат" notice; **no** clean Retry. |
| `error: no_answer` / `agent_error` (nothing shown) | full error state + Retry. |
| User presses Stop | `useChat.stop()` aborts the fetch; finalize the partial cleanly. |
| Network drop mid-stream | treated like `interrupted`; keep partial. |
| `reset` after preamble | new text part; render-last drops preamble (§5.2). |
| Missing `X-Response-Id` | hide like/dislike for that turn. |
| Offline | composer disabled + banner; reads come from Dexie. |

## 10. Defensive protocol note (rollout window)

Because protocol-v2 is merged but rollout is pending, the BFF parser should be tolerant: if a frame arrives as bare `data: <text>` with **no** `event:` line (the legacy plain-text path), treat it as a `token` (after un-escaping literal `\n` → newline). This makes the client work against both the old and new API during the deploy window without a rewrite.

## 11. Libraries (pinned)

| Concern | Choice | Notes |
|---|---|---|
| Framework | **Next.js (App Router, React 19) + TypeScript** | Route Handlers host the BFF. |
| Streaming / chat state | **AI SDK v5** (`ai`, `@ai-sdk/react`) | `createUIMessageStream` in the BFF; `useChat` in the client. |
| Chat UI | **AI Elements** (+ shadcn/ui, Tailwind v4, Streamdown, lucide-react) | Vendored components you own and edit. |
| Local persistence | **Dexie** (IndexedDB) | Conversations + messages. |
| Models fetch | **TanStack Query** | Cache `/api/models`; future reads. |
| UI state | **Zustand** | Active model, sidebar open/closed. |
| Testing | **Vitest + React Testing Library**, **Playwright** | Unit (SSE→stream mapping), component, streaming e2e. |

Supporting libs (Dexie/TanStack Query/Zustand) are swappable; AI SDK + AI Elements are load-bearing. Dexie is the one supporting lib we keep regardless (local history).

## 12. Project layout

```
web/
  app/
    layout.tsx
    page.tsx                     # chat screen (sidebar + thread)
    api/
      chat/route.ts              # protocol-v2 → UI message stream translator
      feedback/route.ts          # injects x-api-key, proxies
      models/route.ts            # proxy + cache
  components/
    ai-elements/                 # vendored AI Elements (conversation, message, prompt-input, …)
    shell/                       # sidebar, conversation-list, app shell
    chat/                        # search-status chip, answer actions (copy/feedback/regenerate)
  lib/
    db.ts                        # Dexie schema + helpers
    api-types.ts                 # ChatSchema/FeedbackSchema/MyUIMessage types
    sse.ts                       # protocol-v2 parser (+ legacy-text fallback)
    transport.ts                 # DefaultChatTransport config / prepareSendMessagesRequest
    env.ts                       # server-only env access
  package.json                   # standalone npm app
  ...config (tsconfig, tailwind, next, vitest, playwright)
```

`web/` is a standalone npm app for now. Docker image + `compose.yaml` wiring is deferred to a later iteration (the implementation plan notes it as a follow-up, not v1).

## 13. Environment & config

Server-only (BFF) env vars — never exposed to the browser:

- `API_BASE_URL` — e.g. `http://api:8880` (in-compose) or the deployed protocol-v2 URL.
- `CHAT_API_KEY` — the `x-api-key` value for `/api/feedback`.

No public (`NEXT_PUBLIC_*`) vars are required for v1; the browser only calls same-origin `/api/*`.

## 14. Testing strategy

- **Unit:** the protocol-v2 parser and the BFF event→UI-message-stream mapping, driven by recorded fixtures covering: plain token stream; tool path (`status` → `reset` → answer); each `error` code; `done`; and the legacy bare-`data:` fallback.
- **Component:** composer (Enter/Shift-Enter, stop), message rendering (render-last text part, search chip, bare-URL autolink), feedback toggle.
- **E2E (Playwright):** a mocked `/api/chat` streaming a tool sequence; assert the chip appears, preamble is dropped, the answer renders, and like/dislike posts.

## 15. Non-goals (explicit)

Auth/login, rate-limiting/quota, server-side conversations / sync, FAQ-links browse, recommender UI, attachments/voice. Each is designed-around (the BFF seam, the local store, the modular shell) so adding them later doesn't require a rewrite.

## 16. Risks & coordination

- **Branch skew:** the working branch lacks protocol-v2; develop against `main`/a protocol-v2 deployment. Mitigated by the §10 fallback.
- **Open paid path:** `POST /chat/` is unauthenticated and unmetered. Accepted for the trusted/internal draft; the BFF is the place to add per-IP/session limits before any public launch.
- **Feedback key:** the master `x-api-key` lives only in the BFF env. If the app is ever made fully static, feedback must move to a public endpoint or a separate token.
- **AI SDK v5 version coupling:** AI Elements + `useChat` track AI SDK v5; pin versions and vendor the components.

## 17. Open / defaulted sub-decisions (override anytime)

- Macedonian-only chrome (vs bilingual MK/EN) — defaulted to MK.
- Like/dislike feedback included in v1 — can be cut to ship chat sooner.
- `web/` as a standalone app now; Docker/compose wiring later.
