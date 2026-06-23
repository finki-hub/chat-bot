# Response Feedback (Like / Dislike) — Implementation Plan

## Context

**Need.** When the chat-bot answers a user (today only through the Discord bot, `E:\finki-hub-discord-bot`), there is no way for the user to signal whether the answer was good or bad. We want 👍/👎 buttons on each answer so satisfaction can be captured. A separate React web frontend is planned later and must support the *same* feedback against the *same* backend. A dead analytics service (`E:\finki-hub-analytics`, MongoDB event sink) exists and may later aggregate this data, but is out of scope for v1.

**Goal.** Capture binary like/dislike per answer, durably, in a way that is **consumer-agnostic** (Discord now, React web later) and decoupled from the analytics service. Only the user who asked may rate; feedback is editable (flip like↔dislike); storage lives where the team already operates a database. v1 covers **both** Discord answer surfaces — slash-command replies (`/ask`, `/prompt`) and thread/reply continuations.

**The core problem — response identity.** Nothing identifies an answer end-to-end today. `POST /chat` (`api/app/api/chat.py:82-116`) streams plain SSE tokens with **no id** and the retrieval pipeline discards chunk/question ids after building context (`api/app/llms/context.py`). The Discord bot keys everything by Discord `messageId` — useless to a browser. So the backend must mint a **canonical `response_id`** that every consumer shares.

**Approach (settled over prior discussion, decisions locked with the requester).**

- **Identity:** the chat-bot mints a `response_id` (UUID) per `/chat` call and returns it as an **`X-Response-Id` HTTP response header** — *not* an SSE event. The Discord bot consumes `/chat` via native `fetch` + `eventsource-parser` (verified: `requests.ts:43-50`, `:69-86`), so `result.headers.get('x-response-id')` is readable the instant the response resolves, before the body is read. An SSE-event carrier was rejected: the bot's parser concatenates **every** `data:` line into the visible answer regardless of `event:` name (`requests.ts:69-74`), so a meta event would render as garbage text.
- **Ownership without server state:** the **asker's id is carried in the button `customId`** (`chatFeedback:<like|dislike>:<response_id>:<asker_id>`, ≈78 chars, under Discord's 100-char limit). The click handler compares `interaction.user.id` to the parsed `asker_id`. This is restart-safe and works identically on the interaction surface and the thread/message surface (where `interaction.message.interactionMetadata` is `null` and the pagination-style guard would otherwise fail open). A user cannot forge a `customId` — Discord only echoes back ids the bot set — so this is safe.
- **Storage:** a new consumer-agnostic `feedback` table in the chat-bot's existing PostgreSQL, written via a new `POST /chat/feedback` endpoint — the **shared contract** for all consumers. The bot has no database (in-memory `Map` + `config/bot.json` only), so the chat-bot owns durability.
- **Source-level attribution deferred:** which chunks/model produced an answer is *not* in scope; that needs the retrieval-pipeline surgery we are explicitly avoiding (`context.py` discards ids). Feedback rates the **visible answer**. The `response_id` we introduce now is the hook a future attribution feature would build on, but no `context.py` change happens here.
- **Analytics deferred:** the analytics service is not revived for v1 (dead ~8 months, zero tests, committed secrets, no bot client). It becomes the v2/v3 aggregation home as its own hardening project.

**This plan's scope:** chat-bot (`response_id` header + `feedback` table + `POST /chat/feedback`) → Discord bot, **both** answer surfaces (read header, attach buttons, handle clicks, POST feedback) → forward-compatible contract for the React web app (Phase 2, planned not built) → analytics + attribution (Phase 3, deferred). Decisions locked: storage = chat-bot Postgres; rate the visible answer only; binary like/dislike; only the original asker may rate; introduce `response_id` now; cover both Discord surfaces; feedback **on by default** per guild (disableable via `/config`); Q&A captured **client-attested** (best-effort) and stored without a retention limit; v1 review via **direct DB** (pgAdmin/psql) — no read endpoint, no analytics revival yet.

---

## Architecture & data flow

```
/ask|/prompt (interaction)  ┐
thread reply / msg-reply     ├─►  POST /chat ───►  chat-bot: mint response_id (uuid4) in chat()
(handleChatMessage)         ┘                       set X-Response-Id on the returned StreamingResponse
                                                    (single chokepoint — covers all 6 providers)
   ◄──── X-Response-Id: <uuid> + SSE body ─────────
   stream complete, answer in hand:
     - read response_id from result headers (sendPrompt returns it; null on 503/error)
     - attach ActionRow [👍|👎] to the FINAL answer message via Message.edit({components})
       customId = chatFeedback:<type>:<response_id>:<asker_id>
       asker_id = interaction.user.id  (slash)   |  message.author.id  (thread/reply)
     - cache {response_id → question, answer, models} in an in-memory enrichment Map (best-effort)

 user clicks 👍/👎  (a FRESH interaction with its own token)
   handleButton: customId.split(':') → ['chatFeedback','like'|'dislike', response_id, asker_id]
   guard 1: interaction.user.id === asker_id            else reject (not your answer)
   guard 2: guild feedback enabled (guild context only) else reject (re-checked at click time)
   POST /chat/feedback {response_id, client:'discord', user_id, feedback_type, ...enrichment}
        │  (x-api-key: API_KEY)
        ▼
 chat-bot Postgres  ◄── ON CONFLICT(response_id,client,user_id) DO UPDATE  (flip / latest-wins)
   ephemeral ack ("Thanks!" / "Updated")
```

Acceptance and ownership depend only on data carried in the `customId` (`response_id` + `asker_id`), so feedback survives a bot restart. The in-memory enrichment Map is a *cache* that adds question/answer/model text when still present; its absence never blocks feedback.

---

## Chat-bot changes (`api/`)

### 1. `feedback` table — append to `api/resources/schema.sql`

Idempotent DDL in the file's existing style (`CREATE TABLE IF NOT EXISTS`, `gen_random_uuid()` — provided by core PostgreSQL 13+, already used by the `question` table at `schema.sql:6`; the `uuid-ossp` extension at `schema.sql:1` stays for the existing `uuid_generate_*` usage but is **not** what supplies `gen_random_uuid()`). `TEXT` + `CHECK` instead of `ENUM` because the whole file re-executes as one transaction on every migration run (`connection.py:122-148`) and `CREATE TYPE` is not idempotent.

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid (),
    response_id     UUID NOT NULL,                                   -- canonical answer identity (X-Response-Id)
    client          TEXT NOT NULL CHECK (client IN ('discord', 'web')),
    user_id         TEXT NOT NULL,                                   -- namespaced per client (Discord uid / web account)
    feedback_type   TEXT NOT NULL CHECK (feedback_type IN ('like', 'dislike')),
    client_ref      TEXT,                                            -- Discord messageId / web handle (traceability)
    channel_id      TEXT,
    guild_id        TEXT,                                            -- Discord-only, nullable
    question_text   TEXT,                                            -- client-attested (see provenance note), nullable
    answer_text     TEXT,                                            -- client-attested, nullable
    inference_model TEXT,
    embeddings_model TEXT,
    query_transform_model TEXT,                                      -- nullable; Discord never supplies it (see bot §)
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS feedback_response_client_user_idx
    ON feedback (response_id, client, user_id);
```

`created_at`/`updated_at` are `TIMESTAMP` (naive) and asyncpg returns naive datetimes for them — matching the existing `QuestionSchema` behavior. (The `ZoneInfo(settings.TZ)` pattern at `chat.py:21` applies only to datetimes the new code *constructs*, of which there are none here — do **not** attach a tz to DB-sourced timestamps.)

**Deployment step — mandatory and explicit.** The lifespan does **not** run migrations (its docstring at `main.py:49` is wrong; `app/main.py:46-62` only calls `db.init()`). Schema changes apply **only** via `python -m app.migrations` (`app/migrations.py:7-19` → `db.run_migrations()`). Forgetting this means `X-Response-Id` is minted but every `POST /chat/feedback` 500s on a missing table. Add the migration run to the deploy runbook alongside the code rollout.

### 2. `X-Response-Id` header — single chokepoint in `chat()`

**Do NOT** add the header inside `stream_sync_gen_as_sse` (`agents.py:14-37`) or any per-provider stream function. The default inference model (`CLAUDE_SONNET_4_6`) routes through `stream_anthropic_agent_response`, which returns its own `StreamingResponse` directly (`anthropic.py:172`; same for `openai.py:171`, `google.py:237`, `ollama.py:203`, `gpu_api.py:137`) and **bypasses `stream_sync_gen_as_sse`** — that function is only the tool-less fallback path. Patching it works in dev/Ollama and silently never emits the header in production with the default Claude agent.

Verified call chain: `chat()` (`api/app/api/chat.py:82-116`) `return await handle_chat(...)` → `handle_chat` (`api/app/llms/chat.py:19-50`) → `stream_response_with_agent` (`api/app/llms/streams.py`) → each provider fn returns a `StreamingResponse` **object** (never a bare generator), the same instance all the way up. So mint the id in the handler and mutate that object's headers before returning:

```python
from uuid import uuid4

async def chat(payload: ChatSchema, db: Database = db_dep) -> StreamingResponse:
    ...
    response_id = uuid4()
    response = await handle_chat(payload, context)
    response.headers["X-Response-Id"] = str(response_id)   # MutableHeaders, set before the ASGI server streams the body
    return response
```

Starlette's `StreamingResponse` stores headers at construction and serializes them only when the ASGI server calls the response, so setting `.headers[...]` after construction but before `return` lands before the first body byte. On the retrieval-error path (`RetrievalError` → 503 `JSONResponse`, `main.py:116`) `handle_chat` is never reached, so no header is sent — the bot must tolerate its absence (it does; see Discord §2).

### 3. `POST /chat/feedback` — endpoint, schemas, data layer

New router `api/app/api/feedback.py` with `prefix="/chat"` and tag `Feedback`, registered in `app/main.py` after `chat_router` (`main.py:103`) with a matching `openapi_tags` entry (`main.py:74-80`). The route is `@router.post("/feedback", ...)` → path `POST /chat/feedback`. **Guard with `dependencies=[api_key_dep]`** (`verify_api_key`, `auth.py:6-24` → 401 on missing/invalid) like every other write route (`questions.py:124`); `/chat` itself stays unguarded as today. Match the codebase's per-route conventions: explicit `status_code`, `operation_id`, and a `responses` block.

```python
# api/app/schemas/feedback.py
class FeedbackSchema(BaseModel):                                # request body (client-attested fields)
    response_id: UUID
    client: Literal["discord", "web"]
    user_id: str = Field(min_length=1)
    feedback_type: Literal["like", "dislike"]
    client_ref: str | None = None
    channel_id: str | None = None
    guild_id: str | None = None
    question_text: str | None = None
    answer_text: str | None = None
    inference_model: str | None = None
    embeddings_model: str | None = None
    query_transform_model: str | None = None

class FeedbackAckSchema(BaseModel):                             # response model — minimal ack, no 15-column mapping
    id: UUID
    response_id: UUID
    feedback_type: Literal["like", "dislike"]
```

```python
# api/app/api/feedback.py
@router.post(
    "/feedback",
    dependencies=[api_key_dep],
    status_code=status.HTTP_200_OK,                            # 200: upsert may insert OR update
    operation_id="submitFeedback",
    responses={401: {"description": "Missing or invalid API key"}},
)
async def submit_feedback(payload: FeedbackSchema, db: Database = db_dep) -> FeedbackAckSchema:
    row = await upsert_feedback(db, payload)                   # raises 500 via the generic handler on unexpected None
    logger.info("feedback %s recorded for response %s", payload.feedback_type, payload.response_id)
    return FeedbackAckSchema(id=row["id"], response_id=row["response_id"], feedback_type=row["feedback_type"])
```

Data layer `api/app/data/feedback.py` mirrors `questions.py` exactly: file-level `# mypy: disable-error-code="arg-type"` (`questions.py:1`), `db.fetchrow`, parameterized `$1..$n` (all values parameterized, **no** column-name interpolation → **no** `# noqa: S608`), and an explicit `Record → schema` map at the endpoint (the codebase never returns a raw asyncpg `Record` to FastAPI — `questions.py:89-117` maps columns by hand; we do the minimal three-field map above).

```python
async def upsert_feedback(db: Database, fb: FeedbackSchema):   # returns the asyncpg Record (id, response_id, feedback_type)
    query = """
    INSERT INTO feedback (response_id, client, user_id, feedback_type, client_ref,
                          channel_id, guild_id, question_text, answer_text,
                          inference_model, embeddings_model, query_transform_model)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ON CONFLICT (response_id, client, user_id)
    DO UPDATE SET feedback_type = EXCLUDED.feedback_type,
                  updated_at = NOW()
    RETURNING id, response_id, feedback_type
    """
    return await db.fetchrow(query, fb.response_id, fb.client, fb.user_id, fb.feedback_type, ...)
```

Toggle semantics: opposite button flips `feedback_type` (latest wins, `updated_at` advances); same button re-click is an idempotent no-op (re-writes the same value). No toggle-off/delete in v1.

**Bot-observable contract:** `200` = recorded/updated; `401` = bad/missing key; `5xx` = transient → the bot may retry/log. The bot must not assume a `201`.

**Provenance (decided for v1):** `question_text`/`answer_text`/model fields are **client-attested** — the server stores what the consumer reports and does not verify it against what was actually asked/answered under that `response_id`. Acceptable for v1 because the Discord bot is server-side trusted. The endpoint still validates `response_id` is a UUID and the enums via Pydantic. The durable, server-authoritative alternative (write a `response` row at chat time) is the natural Phase 3 bridge to source attribution; not built now.

---

## Discord bot changes (`E:\finki-hub-discord-bot`, `src/`)

The bot answers on **two surfaces**, and v1 covers both:
- **Interaction surface** — `/ask`, `/prompt` slash/context commands → `handlePromptWithStreaming` (`streaming.ts:27-44`) → `safeStreamReplyToInteraction` (`messages.ts:182-262`), which sends via deferred `interaction.reply`/`editReply` + `followUp`.
- **Thread / reply surface** — follow-ups in private chat threads and message-replies → `handleChatMessage` (`reply.ts:53-119`) → `safeStreamReplyToMessage` (`messages.ts:264-314`), which sends via `message.reply`. **No interaction, no token, `interactionMetadata` is null.**

Both call `sendPrompt` (`requests.ts:43-86`) — so the changes below apply to **both** hooks and **both** send helpers.

### 1. Surface `X-Response-Id` out of `sendPrompt` (touches both callers)

`sendPrompt` currently discards the `Response` after streaming. Read `result.headers.get('x-response-id')` right after `await fetch(...)` (`requests.ts:50`, before the `getReader()` loop) and **return it** (`Promise<string | null>`; `null` on the 503/error path). `sendPrompt` has **two** callers — `streaming.ts:32` and `reply.ts:102` — both must be updated to capture the returned id (a signature change breaks `reply.ts` otherwise).

### 2. Return the final `Message` handle and attach buttons to it

The previous "use `interaction.editReply({components})`" idea is wrong: `editReply` edits the **first** reply, and both helpers return `string[]` (ids), not `Message` objects — so on a multi-message answer the buttons would land on the wrong message. Fix at the source:

- Change **both** `safeStreamReplyToInteraction` and `safeStreamReplyToMessage` to return the **final `Message` handle** (alongside or instead of `messageIds`). The interaction helper already holds the followUp `Message[]` internally (`messages.ts:198, 205-208`) and the index-0 reply via `interaction.fetchReply()`; the message helper already has the `message.reply` result. Return the last one.
- Attach via `await finalMessage.edit({ components: [actionRow] })`. `Message.edit()` works uniformly for an interaction reply, a `followUp`, and a `message.reply` result, single- or multi-message.
- Only attach when `response_id` is present (the 503 path sends no header) **and** `answer.length > 0` (already checked at `streaming.ts:39`). The no-context fallback ("Не можев да пронајдам…") is a *valid* low-confidence answer and **should** stay rateable; only empty/failed streams are excluded (there is no reliable error sentinel, so suppression beyond `answer.length > 0` is best-effort).
- Cache `{ response_id → { question: options.messages.at(-1)?.content, answer, embeddings_model: options.embeddings_model, inference_model: options.inference_model } }` in a module-level `Map` with the same size-bounded eviction as `conversation.ts:11-34` (1000 entries). Best-effort enrichment; absence never blocks feedback. Note: the bot's transformed options have **no** `query_transform_model` (only `embeddings_model`/`inference_model`, snake_case per `Chat.ts:24-25`), so that column is always null from Discord.

### 3. Build the ActionRow

Pagination wraps buttons in a Components-V2 `ContainerBuilder` (`pagination.ts:32`, with `MessageFlags.IsComponentsV2`) — do **not** copy that. The chat reply is plain `content`, so use a plain row:

```ts
const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
  new ButtonBuilder().setCustomId(`chatFeedback:like:${responseId}:${askerId}`)
    .setLabel(labels.like).setStyle(ButtonStyle.Secondary),
  new ButtonBuilder().setCustomId(`chatFeedback:dislike:${responseId}:${askerId}`)
    .setLabel(labels.dislike).setStyle(ButtonStyle.Secondary),
);
```

`askerId` = `interaction.user.id` (interaction surface) or `message.author.id` (thread/reply surface). customId ≈78 chars < 100.

### 4. New `chatFeedback` button command

New `src/modules/chat/commands/button/chatFeedback.ts` (`export const name = 'chatFeedback'`, `export const execute = async (interaction, args) => {...}`, like `listQuestions.ts:10-15`). Auto-discovery scans **compiled `./dist`** `modules/*/commands/button/` (`modules.ts:34, 64-69`), so a build is required for the new file to load. `handleButton` (`handlers.ts:175-238`) splits `customId` on `:` → `args = ['like'|'dislike', responseId, askerId]`, auto-defers ephemerally (keep it deferred — do **not** add `chatFeedback` to `nonDeferredCommands`), and `chatFeedback` must not be permission-gated (`hasCommandPermission`, `handlers.ts:208-232`). `execute`:

1. **Asker guard** — reject (ephemeral, reuse `commandErrors.buttonNoPermission`) if `interaction.user.id !== askerId` (parsed from `args`). Restart-safe; works on both surfaces.
2. **Opt-out guard** — in a guild, `getConfigProperty('feedback', interaction.guild.id)`; if disabled, reject with a localized notice (re-checked here, not only at attach). In a DM (`interaction.guild === null`), no per-guild config applies → feedback is allowed by default.
3. **Assemble + POST** `/chat/feedback` via a new `sendFeedback()` in `requests.ts`, reusing the **existing** `getChatbotUrl()` (`environment.ts:24-33`, `CHATBOT_URL`) and the **existing** `getApiKey()` (`environment.ts:35-41`, env `API_KEY`) sent as `x-api-key` — exactly as `fillEmbeddings` already does (`requests.ts:160-172`). No new env var. Payload: `response_id` + `feedback_type` from `args`, `client:'discord'`, `user_id: interaction.user.id`, `client_ref: interaction.message.id`, `guild_id`/`channel_id` from the interaction, and `question_text`/`answer_text`/models from the enrichment Map **if present**.
4. **Ack** — ephemeral `editReply`: localized "recorded" / "updated". On POST failure: Winston-log + ephemeral retry message.

### 5. i18n keys (plain TS object exports under `src/translations/`)

No i18n library. Add to existing objects: `labels.ts` → `like`, `dislike` (button labels); `commands.ts` → `commandResponses`: `feedbackRecorded`, `feedbackChanged`, and `commandErrors`: `feedbackDisabled` (reuse existing `commandErrors.buttonNoPermission`, "Командата не е ваша.", for the not-the-asker case).

### 6. Per-guild config + DM behavior

Follow the `ticketing` pattern: add `feedback: z.object({ enabled: z.boolean().optional() }).optional()` to `RequiredBotConfigSchema` (`BotConfig.ts:24-29`); add `feedback: { enabled: true }` to `DEFAULT_CONFIGURATION` (`defaults.ts:19-23` — every key must be present due to `satisfies Record<BotConfigKeys, unknown>`); optional `getFeedbackProperty` accessor mirroring `getTicketingProperty` (`index.ts:139-147`). Editable via `/config` with no extra wiring (`getConfigKeys()` auto-discovers, `index.ts:108-109`). **Default is on (opt-out):** `enabled: true` — feedback appears in every guild unless an admin disables it (treat a missing/undefined `feedback.enabled` as enabled so existing guild configs get the feature without an edit). **DMs:** no guild → no per-guild gate; feedback buttons are attached and honored by default, asker = the DM user.

---

## Phase 2 (planned, not built): React web app

The web app reuses the **same** `POST /chat/feedback` and the **same** `X-Response-Id` header — it reads the header off its `/chat` `fetch` response, renders its own React like/dislike buttons, and POSTs with `client:'web'`. The table and endpoint already accommodate it (Discord-only fields nullable). Open items for that phase:

- **Identity/auth:** no Discord asker. "Only the asker can rate" becomes "only that authenticated session/account." It **must not** trust a client-supplied `user_id` (a browser could spoof it) — needs real session auth.
- **Calling pattern:** the feedback endpoint is `x-api-key`-guarded, which a browser can't hold safely. The web app should call through its own backend/BFF that injects the key and the trusted `user_id`, not POST directly from the browser. (Recommended.)
- **CORS:** add the web origin to `ALLOWED_ORIGINS` and set `EXPOSE_HEADERS` to an explicit list including `X-Response-Id` (`settings.py:41-42`, `main.py:86-97`). The default `['*']` does **not** reliably expose custom headers to browser JS and is invalid with credentialed CORS. **No CORS/settings change is needed for v1 Discord** — the bot uses native `fetch` (not a browser), so `Access-Control-Expose-Headers` is irrelevant to it and `X-Response-Id` is readable today.

## Phase 3 (deferred): analytics + source attribution

- **Analytics aggregation:** revive `E:\finki-hub-analytics` as its own hardening project (rotate committed secrets, add tests + monitoring), then forward feedback events to its `event_type="feedback"` ingest (works as-is) for clustering/sentiment/trend queries. Not a rider on this feature.
- **Source attribution:** to answer "which chunks/model produced the disliked answers," write a server-authoritative `response` row at chat time (question, answer, models, retrieved chunk/question ids + scores) keyed by `response_id`, by retaining the ids `context.py` currently discards. The feedback row then joins to it.

---

## Phased sequence (build order)

1. **Chat-bot schema** — append the `feedback` table + unique index to `schema.sql`; run `python -m app.migrations`; confirm idempotent re-run.
2. **Chat-bot `response_id`** — mint in `chat()`, set `X-Response-Id` on the returned `StreamingResponse` (single chokepoint). Verify the header on the **default Claude agent path** and via a non-Claude model override (below).
3. **Chat-bot endpoint** — `FeedbackSchema` + `FeedbackAckSchema`, `data/feedback.py` upsert, `api/feedback.py` router (api-key-guarded, `operation_id`, `status_code`, `responses`, logging), wire into `main.py`.
4. **Bot — `sendPrompt`** — return the header value; update **both** callers (`streaming.ts`, `reply.ts`).
5. **Bot — send helpers** — return the final `Message` handle from **both** `safeStreamReplyToInteraction` and `safeStreamReplyToMessage`.
6. **Bot — attach + enrich** — at **both** hooks (`handlePromptWithStreaming`, `handleChatMessage`): attach the ActionRow to the final message via `Message.edit`, populate the enrichment Map, tolerate missing header/empty answers.
7. **Bot — handler** — `chatFeedback` button command (asker guard via `customId`, opt-out guard, `sendFeedback()` reusing `getApiKey()`/`getChatbotUrl()`, ephemeral ack); i18n keys; per-guild config.
8. **Verify end-to-end** on both surfaces (below).

Chat-bot (1-3) and bot (4-7) are separate repos/deploys; the chat-bot half ships and is independently verifiable first.

---

## Open decisions (resolved)

1. **Per-guild default — RESOLVED: on by default (opt-out).** `feedback.enabled = true`; feedback appears everywhere, individually disableable via `/config`. Maximizes collected signal.
2. **Toggle-off:** v1 = no delete (same-button re-click is a no-op; opposite flips). True clear-feedback needs read-then-delete — deferred.
3. **Button visual state:** v1 = ephemeral ack only, buttons unchanged. Re-coloring/disabling the clicked button needs `interaction.update()`, which requires adding `chatFeedback` to `nonDeferredCommands` (`handlers.ts:38-47`) to avoid the auto-defer conflict — nice-to-have, deferred.
4. **Enrichment durability:** v1 = best-effort in-memory Map (lost on bot restart → feedback still records `response_id` + rating + `answer_text`-if-cached, just without question/models). The durable fix is the Phase 3 server-side `response` record.

---

## Verification (manual end-to-end — no test suite; lint/mypy gate)

Chat-bot uses `uv` + Docker Compose; CI runs ruff format check + ruff check + mypy only (no pytest job).

1. **Migrate:** `docker compose exec api python -m app.migrations`; `docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d feedback"`.
2. **Header — key-free path:** the header is set in `chat()` regardless of provider, so it can be verified without Claude creds by overriding the model: `curl -i -N -X POST http://localhost:8880/chat/ -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"..."}],"inference_model":"<ollama-or-gpu-model>"}'` → confirm `X-Response-Id:` in the response headers. Then repeat with the **default** model (Claude agent path) to confirm the chokepoint covers it.
3. **Feedback write + toggle:** `curl -X POST http://localhost:8880/chat/feedback -H "x-api-key: $API_KEY" -H 'content-type: application/json' -d '{"response_id":"<uuid>","client":"discord","user_id":"123","feedback_type":"like"}'` → `200` + ack body; `SELECT * FROM feedback;`. Re-POST `dislike` → same row flips, `updated_at` advances (upsert, not a 2nd row). POST without `x-api-key` → `401`. Mismatched trailing slash (`/chat/feedback/`) → note FastAPI's `redirect_slashes` 307; the bot must call the exact path.
4. **Bot — both surfaces:** in a test guild, (a) run `/ask`, confirm 👍/👎 on the final message (use a long answer to exercise the multi-message followUp path); (b) in a chat thread / via a message-reply, confirm buttons appear on that answer too. On each: click as asker (records), opposite (flips), as a different user (rejected "not yours"); disable `feedback` in `bot.json` and confirm a stale click is rejected; restart the bot and confirm a click still records (minus enrichment). Confirm a DM answer also gets working buttons.
5. **Static gates:** `cd api && uv run ruff format --check . && uv run ruff check . && uv run mypy app` — new Python needs full type annotations; DB-sourced timestamps stay naive (do not wrap in `ZoneInfo`); budget for the COM812 format↔check converge loop. Bot: `tsc`/build so the new button command is discovered from `dist`.

---

## Out of scope (documented follow-ups)

- Free-text reason on dislike (Discord modal) — Phase 2+ UX.
- Reviving the analytics service (Phase 3, own project).
- Retrieval source attribution / server-side `response` record (Phase 3).
- Aggregation/reporting endpoints or dashboards over the `feedback` table — **v1 review is via direct DB access** (pgAdmin/psql, already in compose); a query endpoint and the analytics service are the Phase 2/3 home.
- Restyling/disabling buttons after a click (`interaction.update()` path).
