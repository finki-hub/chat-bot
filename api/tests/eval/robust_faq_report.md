# Focused FAQ retrieval: verified before/after

**Both comparisons passed their validity and release gates.** Execution: 2026-10-13, 23:17:05–23:18:36 UTC.
The 38 cases are authored retrieval regressions,
not representative/blinded evaluation or generated-answer accuracy/entailment.
This neither reconstructs historical vectors nor revises or validates thesis results.

## Method

Baseline: immutable API revision `a755194`, image `sha256:75f7fd76897dcc98e30bfe363108e0c9021cf767b7584c0082fa38c57f8a6375`.
Candidate: frozen, uncommitted overlay with the same base image/dependencies.
Live API revision `4fd138` was neither the baseline nor modified. Both used GPU image
`sha256:9b0a6239b93b11b5287fed2d4c6832afea4e23e41477235b51da862d56f182f2`
at `4fd138`, healthy throughout, with unchanged BGE-M3/BGE-reranker-v2-m3 contracts.

Isolated production retrieval captured actual candidates, emitted queries, scores,
context and source payloads—no oracle-injected FAQ/vector/query or UI rendering test.
All four runs shared 73 FAQs, 60 documents and 2,391 chunks, with matching full-row/
vector fingerprints and no drift. All 130 candidate Python hashes matched frozen
local code; prompt/model resources were also checked.

## RAW results: 38 authored cases, 25 families

| Observation | Baseline | Candidate |
|---|---:|---:|
| Required FAQ candidate hits | 6/8 | 8/8 |
| Required selected-context hits | 6/8 | 8/8 |
| Required expected-content checks | 6/8 | 8/8 |
| Required visible FAQ sources | 3/8 | 8/8 |
| Required zero-candidate cases | 2/8 | 0/8 |
| Preservation context hits | 12/12 | 12/12 |
| Preservation available-content checks | 10/10 | 10/10 |
| Preservation visible sources | 7/12 | 7/12 |
| Negative local recovery activations | 0/18 | 0/18 |
| Negative Student Service context hits | 3/18 | 3/18 |

New context hits were `rfq-001` (`Kade e sluzba`) and `rfq-002` (`Каде е слузба`).
The actual additional dense/rerank query was **Каде се наоѓа Студентска служба?**
(generic title form). Its target score was 0.9817007184; the full selected FAQ
retained **ТМФ, до кабинетот 117** and was visible. All eight required cases passed
candidate/context/content/visibility checks with unchanged 0.5/0.1/0.3 gates.

The two Computer Center location/hours controls retrieved the expected FAQ but
kept `content_pass:null`: that FAQ lacks those requested facts. This is not absence
throughout the corpus or evidence of correct generated answers. No preservation
regressions, required failures or new negative failures occurred.

**Residual leakage remains:** `rfq-031` and `rfq-032` concern other faculties;
`rfq-035` quotes a service term while explicitly not requesting Student Service.
Baseline leakage **3/18 remained 3/18**, with no negative local recovery or new
measured leakage. This is not global negative precision; these dense-retrieval
issues remain follow-up work.

## Missing credentials and resource use

A separate pair used exactly the same eight required definitions, requested
`rewrite_hyde`, supplied no credentials, and actually used RAW with
`credential_missing`. Candidate/context/content again improved 6/8→8/8 and
visibility 3/8→8/8. Both gates passed. This subset has **no negative or preservation
denominator**; its results must not be pooled as independent additional tasks.

Across all four runs: **92 completed executions, 113 embedding and 68 rerank HTTP
calls**, all self-hosted. There were zero case errors, database read failures or
GPU HTTP failures. Maximum pool size was 81/125 and maximum embeddings per case 2.
Read-only SQL and strict read-failure/GPU-only guards were verified. No answer
generation, paid-provider calls, deployment, chat/quota/corpus writes or thesis edits.

Recovery is not free: RAW embeddings increased 38→51 and reranks 26→28; the eight-case
credential subset increased 8→16 embeddings and 6→8 reranks. Execution was sequential,
baseline first and not randomized; observed latency differences are **not a causal
speedup claim**. No embedding caching was added.

## Reproduction and evidence

Use isolated baseline/candidate imports and shared dependencies, with authorized
DB/GPU settings inherited in-memory, never credentials in arguments or a live worker.
Public manifests and raw observations stay local, not uploaded with this report.

```text
python -B -m tests.eval.robust_faq_replay --corpus-manifest PUBLIC_CORPUS.json --gpu-manifest PUBLIC_GPU.json --revision-label a755194 --mode raw --output baseline-raw.json
python -B -m tests.eval.robust_faq_replay --corpus-manifest PUBLIC_CORPUS.json --gpu-manifest PUBLIC_GPU.json --revision-label FROZEN_CANDIDATE --mode raw --output candidate-raw.json
python -B -m tests.eval.robust_faq_compare --baseline baseline-raw.json --candidate candidate-raw.json
```

For the credential-missing pair, supply the exact `required-eight.jsonl` subset
with `--fixture`, use `--mode credential_missing`, and compare with the same subset
fixture. See [runner guidance](robust_faq_README.md) for limits and validity gates.

Evidence: `verification-report.md`, comparisons, observations and postflight/execution/import/model manifests.
Old-runner results are preliminary; spellings are not independent and diagnostic holdouts were visible.

Historical local checks: **1,005 passed, 60 skipped, 3 known Windows Docker/sh
failures**; Ruff and mypy clean. CI was pending when this report was first written.

| Frozen input | SHA-256 |
|---|---|
| 38-case fixture | `1ffcb8b32ca97f327bb1d483ddb088f20ed3da2afce0cf7334961aec8152e1b5` |
| Runner | `af72b2eb083f690a2f45a72490c9c7c4888e0272d147a44227083532a90c8866` |
| Comparator | `e9eda9e4a591e93a16b98bb202a5122a893e826ac33ef7c3af12193ae766b0ba` |
| Candidate application source-manifest digest | `db22f78050fa960c21c7aa12dbd205711031a5897122dc054d16cf3b6d73e9bc` |

### Post-review provenance addendum

The 92 measured executions, results and frozen hashes above describe the implementation
recorded at `7e6dadb1934178cfc04fcc5bcd6e7818ff71ad02`, not the post-review revision.
Subsequent retrieval-task cleanup and legacy matched-RAW budget validation fixes were
validated offline with [cleanup regression tests](../test_faq_query_recovery_context.py)
and [runner/comparison integration tests](test_compare_eval_faq_recovery.py).
No new live DB/GPU study was performed for these fixes. The historical application
source-manifest digest is not a fingerprint of the latest `context.py`; the measured
results and hashes are retained unchanged. CI for the post-review commit remains pending.
