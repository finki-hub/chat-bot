# API retrieval diagnostics

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
