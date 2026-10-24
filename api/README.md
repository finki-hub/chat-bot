# API retrieval diagnostics

## Discord conversation storage

Discord requests with a `user_id` and an `X-Conversation-Id` are stored in
`chat_conversation`/`chat_message` after the stream ends: the question, the answer,
sources and diagnostics. Requests rejected before admission (credentials, quota)
are not stored; accepted requests that fail or disconnect are stored with their
outcome. Web conversations are still stored by the web app. Nothing expires.

`DELETE /chat/state/users/{user_id}/history` deletes the user's conversations, the
feedback they gave, and feedback others gave on their Discord answers, in one
transaction. It also bumps `chat_user.history_generation` and records the deleted
conversation IDs in `chat_conversation_tombstone` (IDs only), so an answer that was
still streaming cannot write the history back.

Discord feedback is accepted only for a stored Discord answer, and the question and
answer text are taken from the database rather than from the request. Because
storage finishes in the background, a 404 may carry `X-Feedback-Generation`; the
client may retry briefly with that header unchanged. A 409 means the history was
deleted meanwhile and must not be retried.

On shutdown, pending writes get up to five seconds before they are cancelled.

## OpenCode Go BYOK

Save a separate `opencode-go` credential through the existing chat-state credential
API. The curated model IDs are `opencode-go:gpt-6-luna`, `opencode-go:grok-4.7`,
`opencode-go:claude-haiku-5-5`, `opencode-go:minimax-m3`, `opencode-go:qwen3.8-max`,
`opencode-go:glm-5.3`, `opencode-go:kimi-k3`, `opencode-go:deepseek-v4.1-flash`, and
`opencode-go:deepseek-v4-pro`.

Send the conversation's UUID in `x-conversation-id` on chat and title requests.
Main, title, and agent-setup fallback calls use `User-Agent: finki-hub/1.0` and that same `x-opencode-session` value.
Chat rejects missing/invalid IDs; optional title generation falls back
locally if session context or a Go user key is unavailable. Go credentials never
fall back to deployment credentials or keys saved for other providers.

The default base is `https://opencode.ai/inference/go/v1`; explicit default Go
URLs normalize to the default. Custom proxies still require the exact configured
`BYOK_ALLOWED_BASE_URLS` allowlist. Responses and Chat Completions use Bearer auth;
Messages uses `x-api-key` and an SDK base without `/v1` to avoid duplicating it.
Reasoning is protocol-specific. Grok 4.7, GLM-5.3 and Kimi K3 require reasoning,
so the main chat flow keeps it enabled and does not expose an effective off
setting; auxiliary query/title calls intentionally retain low effort. DeepSeek's
off setting explicitly sends `reasoning_effort: none` (omission would enable
default high thinking). An explicit off request for Grok 4.7, GLM-5.3 or Kimi K3
maps to low effort, not disabled thinking. Messages models explicitly
disable thinking and GPT-6 Luna uses `reasoning.effort: none` when off.
MiniMax M3 sends `thinking.type: adaptive` when reasoning is requested and
`disabled` when off, forwarding its supported `top_p` and `temperature` on both
paths. Adaptive thinking accepts small positive output budgets without silently
switching off; thinking counts toward `max_tokens`, so an exhausted budget may
return no answer text. Qwen3.8 Max retains enabled thinking with a bounded budget
of `min(4096, max_tokens - 1)` and requires `max_tokens > 1024` when reasoning is
requested. Both chat-schema validation and the direct client factory reject
incompatible Qwen reasoning budgets rather than downgrading the request.
Haiku adaptive thinking is unchanged; off-mode query/title calls still accept
small budgets and do not receive unsupported generic Messages sampling knobs.
Apply additive migration `0012_allow_opencode_go_chat_credentials.sql`.
Sponsored access, default models, retrieval, and embeddings are unchanged.

Endpoint and usage evidence: <https://opencode.ai/v2/docs/console/go>.
Standard/off-peak catalog prices are estimates, not subscription billing or
long-context/peak rates. Live account validation is separate from mocked tests.

Bounded live SDK validation on 2026-10-13 returned nonempty answers for all nine
models. Go accepted Chat Completions' `max_completion_tokens` alias and explicit
DeepSeek off mode; GLM/Kimi still emitted reasoning at low effort. One on-mode
request per protocol succeeded, though the trivial prompt did not produce visible
reasoning summaries for Responses/Messages. A DeepSeek streaming cancellation
closed cleanly, and a two-request tool round returned preserved reasoning with a
nonempty final answer. This smoke test does not establish every model/tool flow
or verify hidden reasoning from absence of a visible trace.

Follow-up MiniMax live validation on 2026-10-14 used a small combinatorics prompt:
adaptive/on (`max_tokens=2048`, `top_p=0.95`) returned visible thinking and an
answer; disabled/off (`max_tokens=128`, `top_p=0.1`) returned an answer without
a thinking block. Both SDK-emitted requests asserted the configured sampling
values. This verifies request forwarding and observed thinking behavior, not
the statistical effect of changing `top_p`.

Thinking contracts: [DeepSeek](https://api-docs.deepseek.com/api/create-chat-completion),
[Grok](https://docs.x.ai/developers/model-capabilities/text/reasoning),
[GLM](https://docs.z.ai/guides/capabilities/thinking-mode), and
[Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart).
MiniMax's model-specific Messages contract:
<https://platform.minimax.io/docs/api-reference/text-anthropic-api>.

## Agent search

Chat does not retrieve context up front. The agent calls the `search_knowledge_base`
tool with its own short query, as often as it needs. Per answer, LangChain middleware
allows at most 4 searches and 15 tool calls (over-limit calls are blocked and the model
answers with what it has) and 10 model calls (the run ends). A failed search is
returned to the model as a failed tool result instead of ending the answer. If agent
setup fails, the provider answers without tools from one search on the question. Each search runs raw
retrieval over FAQs and document chunks (plus a Cyrillic variant for romanized
queries), reranks, and appends the most relevant catalog links. The request fields
`query_transform_mode` and `query_transform_model` are accepted for client
compatibility and ignored by chat; chat titles still use the transform model.

Every agent tool call is streamed as a `tool_call` event (`id`, `name`, `label`,
`input`) and a `tool_result` event (`id`, `name`, `ok`, `summary`); text written
around tool calls is kept. `sources` is re-sent after each search, deduplicated,
interleaved across searches and capped at 12.

## Retrieval fields

The API emits the following bounded, content-free retrieval fields on
`$ai_generation` and the `chat.timing` log. Counts, ids and reranker fields describe the
most recently finished search of the answer, while `retrieval.*` span durations add up
across searches (parallel searches overlap, so the sum exceeds wall-clock time).
`retrieval_hit` is true when any search returned context; each search also reports
`hit` and `source_count` on its `tool_called` event.

## Reranker fields

`reranker_fallback_reason` distinguishes:

| Value | Meaning |
| --- | --- |
| `not_run` | Reranking did not produce a selection outcome, including early no-candidate/error paths. |
| `none` | Valid reranked candidates cleared the existing score floor. |
| `score_floor` | Valid results were all below the score floor; the first valid ranked candidate was retained. This is not a service error. |
| `empty_or_invalid_response` | The parsed response contained no usable index/score pairs. The existing dense-only, FAQ-first, top-k fallback was attempted. |
| `error` | Reranking raised an exception; the existing dense fallback was attempted. |

`reranker_fallback` preserves the existing boolean: true for score-floor retention
and exception fallback. For newly recovered empty/invalid responses it is true
when dense recovery actually selected context; otherwise false. Dense fallback
does not expose unscored citation sources or select lexical-only candidates.

`reranker_invalid_result_count` counts discarded result-list entries, saturated at
**1000**. It is zero before reranking, for an empty list, or for a missing/non-list
result container (there are no list entries to count). Valid entries require an
in-range integer index (excluding booleans) and a finite numeric score (excluding
booleans). Invalid entries do not change the order of valid entries.

These diagnostics contain no query/document text, URLs, credentials, or exception
messages. Thresholds, retrieval budgets, model selection, and normal ranking order
are unchanged.
