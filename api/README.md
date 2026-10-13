# API retrieval diagnostics

## OpenCode Go BYOK

Save a separate `opencode-go` credential through the existing chat-state credential
API. The curated model IDs are `opencode-go:gpt-6-luna`, `opencode-go:grok-4.7`,
`opencode-go:claude-haiku-5-5`, `opencode-go:minimax-m3`, `opencode-go:qwen3.8-max`,
`opencode-go:glm-5.3`, `opencode-go:kimi-k3`, `opencode-go:deepseek-v4.1-flash`, and
`opencode-go:deepseek-v4-pro`.

Send the conversation's UUID in `x-conversation-id` on chat and title requests.
Main, retrieval rewrite/HyDE/contextualization, title, and agent-setup fallback
calls use `User-Agent: finki-hub/1.0` and that same `x-opencode-session` value.
Chat rejects missing/invalid IDs; optional title/query generation falls back
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

Thinking contracts: [DeepSeek](https://api-docs.deepseek.com/api/create-chat-completion),
[Grok](https://docs.x.ai/developers/model-capabilities/text/reasoning),
[GLM](https://docs.z.ai/guides/capabilities/thinking-mode), and
[Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart).

## Retrieval fields

The API emits the following bounded, content-free fields on `retrieval_used`,
`retrieval_miss`, `$ai_generation`, and the `chat.timing` log. Requested and effective
query-transform modes retain their existing meanings: effective mode describes
the variants actually retained for retrieval.

These fields enrich events that the API already emits; they do not add terminal
telemetry for failures before stream instrumentation starts. In particular, a
pre-stream retrieval exception can return an error without a `retrieval_*` event,
`$ai_generation` event, or `chat.timing` log. `not_run` is a diagnostic default,
not a guarantee that every failed or cancelled request emits an event.

## `query_transform_fallback_reason`

| Value | Meaning |
| --- | --- |
| `not_run` | No completed variant-generation outcome was recorded (initial/error/cancellation default). |
| `not_requested` | The requested variant mode was `raw`. This does not describe history contextualization. |
| `credential_missing` | A transform was requested, but no matching user-provider credential was available; retrieval used raw mode. |
| `no_usable_variants` | Transform calls completed or failed, but neither requested variant survived. Exceptions, blank output, and output unchanged from the search query all count as unusable. |
| `partial_variants` | Only one of the requested rewrite and HyDE variants survived. |
| `none` | All requested transformed variants survived. |

Sponsored inference credentials are not injected into query transformation. A
credential for a different provider does not make the selected transform available.

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
