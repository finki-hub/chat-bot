# Focused FAQ recovery evaluation

The dated [verified before/after report](robust_faq_report.md) records the completed
RAW and eight-case credential-missing comparisons, including unchanged negative
leakage and retrieval-only limitations.

`robust_faq_golden.jsonl` is a new public, retrieval-only regression fixture. It does
not change the existing golden set or any frozen thesis results. It contains 38
authored queries in 25 task/entity families: eight required Student Service
recoveries, twelve broader FAQ preservation/diagnostic cases, and eighteen negative
local-recovery controls.

Titles and sparse content checks were verified against public FAQ rows in the
authorized 9 September corpus snapshot. They are not assertions that the rules
remain current. Resolve every expected title and content token against the actual
before/after corpus before executing a comparison. A missing or changed anchor
must be reported as a preflight mismatch, not silently skipped or relabelled.
There are no private thesis questions, email threads, production IDs, or captured
answers in this fixture.

## Expectations

- `fix_required`: recover the expected FAQ into the candidate pool and final
  context, retain the specified public content, and expose its visible source for
  all eight required cases. Visibility is mandatory even when the baseline missed
  the FAQ; measure confidence separately without bypassing the unchanged0.3 source
  floor or0.1 context floor.
- `preserve_or_diagnose`: report expected-source recall before and after; newly
  broken baseline successes are regressions. Pre-existing misses remain visible.
- `no_local_entity_recovery`: the new local entity recovery must not activate.
  Report any newly introduced Student Service final-context leakage. This does
  **not** require all existing FAQ/document retrieval to be empty: another valid
  source may legitimately match the question.

Two Computer Center controls request location/hours but carry
`answer_fact_available: false` and `fact_availability_scope: expected_faq_only`:
the inspected FAQ has technical support/contact information, not those facts.
Their candidate/context/source hits remain measured, but `content_pass` is null,
not vacuous success from an empty token list. This says nothing about other corpus
sources or generated-answer behavior. No LLM answer accuracy is measured.

`diagnostic_holdout` denotes planned spelling/intent regression coverage, not a
secret or statistically independent test set. Once implementers see or tune on
these cases, they are ordinary regression cases. Report by family as well as by
query; six variants of a location question are not six independent tasks.

## Before/after boundary

`robust_faq_catalog_controls.jsonl` adds four synthetic pure-helper safety cases:
normalized-title collision, alias/exact-entity ambiguity, missing target title and
an unknown institution modifier. These do not describe actual catalog records and
are **not** additional live replay cases or performance observations. The shared
resolver must reject ambiguous/missing anchors rather than selecting the first row.

Use the actual production retrieval/context path under the approved baseline and
candidate revisions with the same frozen public corpus, BGE versions and budgets.
Do not replace production selection with an oracle rerank containing an injected
target FAQ, duplicate production normalization inside a scorer, or count a useful
link as proof that FAQ content was retrieved.

Save candidate membership, final-context/content checks, source visibility, actual
rerank queries/scores, recovery activation, fallbacks, budgets,
execution status and code/corpus/model/fixture hashes. Keep positive and negative
denominators separate. Any execution error, missing case or corpus/config mismatch
invalidates a paired quality claim. No paid query transforms or answer generation
are needed; retrieval recall is not answer accuracy or citation entailment.

## Parent-authorized replay

`robust_faq_replay.py` imports the actual production
`get_retrieved_context_with_sources` and observes its candidate builder, embedding
calls, emitted rerank query and selected IDs. It does not inject a target FAQ,
canonical question or candidate order. Thus it measures the current generic
canonical template, not the earlier eight-FAQ oracle experiment. All calls are
sequential.

Run from `api/` using its installed Python, in a fresh disposable process. The
parent must supply the correct baseline/candidate application import path,
inherited `DATABASE_URL` and verified self-hosted `GPU_API_URL`. Do not pass secrets
as command arguments or execute this against a live application worker. The GPU
manifest is an existing public model/runtime identity artifact verified by the
parent; its hash is recorded without opening a new health/model probe.

```text
python -B -m tests.eval.robust_faq_replay --corpus-manifest CORPUS.json --gpu-manifest GPU.json --revision-label BASELINE_SHA --output baseline-raw.json
python -B -m tests.eval.robust_faq_replay --corpus-manifest CORPUS.json --gpu-manifest GPU.json --revision-label CANDIDATE_LABEL --output candidate-raw.json
python -B -m tests.eval.robust_faq_compare --baseline baseline-raw.json --candidate candidate-raw.json
```

No hosted provider or answer generation is allowed.

Safety/limits:

- Refuses existing output artifacts. Reads corpus/anchors in server-enforced
  read-only repeatable-read transactions, 10-second statement deadlines, without
  startup hooks, migrations or writes. Serializes concurrent production reads on
  one connection. Transactions are short and do not span GPU calls.
- FAQ IDs/names/body hashes must match the supplied authoritative manifest; every
  required content token is checked against its actual current FAQ body before
  GPU requests. Before/final full-row hash sets include FAQ/document/chunk vectors
  and revisions but export no bodies or vectors. Drift invalidates the run. These
  hashes are runner-specific, not the prior manifest's unspecified MD5 recipe.
- HTTP permits only POSTs to the configured GPU embedding/reranking endpoints;
  synchronous/hosted HTTP is blocked. No API/PostHog initialization or tools.
- At most three observed embedding calls and one logical rerank per case;
  candidate pool capped at125, normal initial_k30/top_k10, unchanged0.5/0.1/0.3
  gates. Counts actual HTTP attempts including production retries. Transport
  failures, invalid reranker output and local plan errors invalidate quality
  claims rather than disappearing from denominators.
- GPU request deadlines30 seconds, case deadline90, whole run600; records partial
  results and sanitized failure classes. Safety violations stop further cases.
  Selected/attempted/completed/not-attempted counts remain separate.
- Benchmark validity is stricter than production's graceful fallback behavior:
  **no transport or database read errors are accepted**. The database adapter
  centrally counts exception classes for fetches, transaction entry/exit and
  corpus reads, including cancellations, without SQL, arguments or exception
  messages. Original exceptions/cancellation propagate unchanged. A failed read
  hidden by optional canonical search or neighboring-chunk fallback still
  invalidates the case and whole replay; lexical-search error diagnostics also
  invalidate cases when using a mock/non-instrumented adapter.
- Top-level and per-case `db_read_failures` ledgers are required and must report
  zero faults. They are not paired-manifest equality fields: each artifact is
  independently checked. Counts denote failed/cancelled read operations, not
  independent cases. Earlier preliminary artifacts without this accounting are
  insufficient for a quality claim and require a new run/output path.
- Source fingerprints come from the actually imported application, not the
  runner's directory. Baseline and candidate must use the same public fixture,
  corpus, GPU identity artifact, runtime dependencies, budgets, floors and mode.

The pure comparison exits2 on invalid/incomparable artifacts,1 on retrieval
regressions/required-recovery failures/unsafe negative activation,0 on the bounded
gate passing. Visible-source loss on a baseline success is a regression; new source
cards are not fabricated to meet a quota. Reports retain paired case IDs, source
manifest hashes and family grouping. Caller labels identify revisions; actual
application source hashes establish which code executed, including uncommitted
overlays. This is not an end-to-end chat or answer-quality benchmark.

`run_eval.py` retains its legacy `anchor` schema and must not be pointed at this
file. Its new raw-mode local recovery parity calls the same async planner and
candidate builder, adds the actual canonical vector/rerank input, and isolates
optional canonical-search failure from mandatory searches. Existing ideal/prod ANN
diagnostics and requested configuration keys remain; direct/title recovery has
separate candidate-pool and local-stage fields. No historical results are rewritten.
