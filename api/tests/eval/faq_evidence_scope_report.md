# Office evidence scope: verified before/after

**Validity, scope, separate P1 preservation and combined release gates passed.**
Execution: **2026-10-14 01:31:44–01:34:14 UTC**. Authored public retrieval regressions,
not blinded/representative evaluation, generated-answer accuracy, citation entailment
or thesis validation. Parent independently reran the combined comparison; separate
read-only evidence review verified the raw rows and frozen identities without blockers.

## Frozen application/runtime inputs

Baseline: **PR1114 post-review `819cbd720ae174a7e75506c3e606ef7844427ef8`**, staged from Git (130 Python files).
Candidate: frozen uncommitted P2 overlay on that revision (131 files). Source-map digests:
- Baseline: `e4f1bcb101792fb169d502e8640d56cc3dfd5e841bb0b20c3691aa4fe99e67d6`.
- Candidate: `2073364984ecf785730481cc3838ed0368c2afb8927c67f4e64affc43ef12d10`.

Isolated staged imports passed source/prompt/model-resource hash checks, using
identical immutable `a755194` **dependencies**, image
`sha256:75f7fd76897dcc98e30bfe363108e0c9021cf767b7584c0082fa38c57f8a6375`.
Neither arm imported live API/web, including the separately observed merged `e347`.
This does not establish P2 deployment. **These identities describe benchmark inputs,
not a future rebased candidate**; unrelated BYOK/main changes can alter its digest.

Actual production retrieval ran RAW, `credentials=None`, sequential/baseline-first,
with unchanged dense/context/source floors **0.5/0.1/0.3**. Passive observation
captured actual pre/post-filter pools, rerank input/index mapping, selected context
and returned source payloads—no candidate/query/score injection or UI rendering test.

## Scope pair: 38 cases / 32 families

| Required guards (14 cases): cases with forbidden evidence | Baseline | Candidate |
|---|---:|---:|
| Post-filter FAQ pool | 13 | 0 |
| Selected evidence | 12 | 0 |
| Curated exact facts in **whole returned context** | 12 | 0 |
| Visible source payloads/cards | 8 | 0 |

All 14 guards activated actual constraints and passed. The three prior incident
counterparts—Law faculty (`sfq-001`), Medical faculty with General Contact
substitution (`sfq-002`), and explicit Student Service exclusion (`sfq-003`)—lost
all forbidden pool/context/fact/source hits. Ten guards returned empty context;
this is scoped abstention, **not evidence that another institution has no office**.
Allowed alternatives—including Archive's Dean-building reference—were preserved;
this is not a global institution/document filter.

- **Local controls:** target candidate/context, fixture content checks and visible
  sources **12/12 in both arms**; seven candidate cases had compatible FINKI constraints.
- **Ordinary controls:** **10/10 unrestricted**, with identical full-context hashes,
  selected IDs and visible sources; unchanged behavior is not answer/source gold.
- **Missing facts:** both Computer Center contexts retained; location/hours content
  checks stayed **null**, visible target hits **0/2**. No fact/answer accuracy credit.

## Separate unchanged P1 pair: 38 cases / 25 families

All eight required cases retained candidate/context/fixture-fact/visible-source
hits **8/8 in both arms**. Preservation retained **12/12 contexts, 10/10 scored
content checks and 7/12 visible targets**; two missing-fact checks remained null.
Across 18 negatives: no new recovery/leakage, candidate recovery **0/18**, and known
Student Service context leakage (`rfq-031/032/035`) **3/18 → 0/18**. No regressions
or required/negative failures. This new post-review-819 comparison neither changes
the dated P1 `a755194` study/report nor proves global negative precision.

## Corpus, resources and safety

Fresh preflight anchor: **2026-10-14 01:29:31 UTC**, FAQ fingerprint
`f977b21326f740eeaa18c0c63e231236ec42b102eb2db20f2c56762257a2be92`.
An unrelated Years of Studies edit occurred before the study; the older `c789…`
anchor was explicitly replaced, not silently reused. All four runs matched their
before/after fingerprints: **73 FAQs / 60 documents / 2,391 chunks**, BGE-M3
`bge-m3-v1`, 1,024 dimensions. Five office bodies, all 11 curated quotations and
positive tokens passed preflight. No corpus writes, synchronization or reindexing.
Shared GPU stayed healthy, unchanged before/after: revision `4fd138`, image
`sha256:9b0a6239b93b11b5287fed2d4c6832afea4e23e41477235b51da862d56f182f2`.

**152 completed executions, 197 embedding + 132 rerank = 329 self-hosted HTTP
requests; zero case, GPU transport or SQL read faults**, including every per-case
read ledger. Peak pool **81/125**. No provider rewrite/generation or paid inference.
Strict catalog/reranker-failure and low-confidence contracts were tested offline,
not by injecting live faults. Sequential timing supports **no causal speedup claim**.
Independent useful-link enrichment was not observed; curated quote checks are not
entailment. Parent-reported prepublication checks: **436 passed**, API Ruff/format
and mypy clean. Multiple spellings/diagnostic holdouts are not independent/blinded.

## Evidence and reproduction

Local evidence root: `C:/Users/Stefan/AppData/Local/Temp/opencode/faq-evidence-scope-verified-20261014T013144Z/`.
Read: four observations, combined comparison, summary/execution/import/resource
proofs, fresh corpus and GPU manifests.
Reproduction commands for four isolated arms and the combined gate are in the
[README](faq_evidence_scope_README.md#runnable-clis-independent-verifier-only);
use the **frozen wired runner**, `frozen-corpus.json` and `gpu-identity.json`, with
authorized settings inherited in-memory, never credentials in arguments. Fresh
execution needs current preflight; do not overwrite this evidence or thesis data.

| Frozen input | SHA-256 |
|---|---|
| Scope fixture | `a8e73330d4ef0bf90ac5cb76ff74cc5aa04f29c95c54715dfc1e124bd40005a6` |
| Public fact catalog | `8cfc15d01b1efab984da248a5ebbf681d504e2394cdbe6a425cdf47dcb8433e3` |
| Wired runner | `427b4ab83aba1740b2a90bb323570b55270952f0d68c5da7c7438cd1e60a8b5d` |
| Scope comparator | `64cfee73c101ae1c014d5771b08040ec1dc1d774c0375edd35ef0f339df1db06` |
| Original P1 fixture | `1ffcb8b32ca97f327bb1d483ddb088f20ed3da2afce0cf7334961aec8152e1b5` |
