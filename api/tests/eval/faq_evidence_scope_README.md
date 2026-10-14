# Office evidence scope — separate public diagnostic

Wired stage: the shared runner accepts the separate scope schema via
`--scope-catalog`, observes the actual production filter and checks post-filter
rerank inputs. The separate scope comparator also requires the unchanged P1
preservation pair. **The actual paired benchmark passed validity, scope,
preservation and combined release gates** on 2026-10-14; see the
[verified measured report](faq_evidence_scope_report.md). These are retrieval-only
authored regressions, not generated-answer accuracy or deployment verification.

The measured study uses the fresh **01:29:31 UTC `f977…` corpus anchor**, not the
earlier 00:32:57/`c789…` preparation snapshot retained below. An unrelated Years of
Studies edit preceded the benchmark; all four measured runs had matching
before/after fingerprints. See the report for frozen execution/source identities.

## Frozen inputs and limits

`faq_evidence_scope_golden.jsonl` contains 38 authored queries, separately from
unchanged `robust_faq_golden.jsonl`. Run the original P1 full 38 cases as a separate
preservation pair. The baseline is **PR1114 post-review** at
`819cbd720ae174a7e75506c3e606ef7844427ef8`, not the earlier `7e6dadb1` or original
`a755194` application. The candidate is the frozen, uncommitted P2 overlay rebased
onto `819cbd7`. Both isolated application imports use the same current wired
runner, immutable `a755194` dependencies, GPU and corpus. The independent verifier
refreshes actual runtime/corpus preflight before execution. No deployment,
reindexing, paid generation or historical thesis changes.

| Expectation | Cases | Required meaning |
|---|---:|---|
| `scope_guard_required` |14 | Enforce explicit office/institution or resolved-entity exclusion; no forbidden FAQ, visible source or curated fact quotation survives |
| `preserve_local_evidence` |12 | Compatible FINKI/unqualified evidence must not newly regress; full-name aliases, office channels and general contact included |
| `preserve_ordinary_path` |10 | No scope restriction; preserve actual ordinary candidate/final/context/source behavior, without invented source/answer gold |
| `fact_missing_control` |2 | Computer Center location/hours remain known absent from that FAQ; no content/answer credit |

Known regressions copy `rfq-031`, `rfq-032`, `rfq-035` exactly into `sfq-001..003`.
Other guards cover additional faculties, an explicitly synthetic/unverified
institution qualifier, dean/archive/computer/general contact attributes, Cyrillic,
Latin and diacritics. An unknown qualifier is not proof of a real foreign
institution: it is insufficient authority to substitute our local office facts.

Explicit exclusions with another request must retain allowed evidence. In
particular, an Archive location can legitimately mention the dean building;
the word `деканат` alone is not forbidden evidence. Normal negation, harmless
quotations, transfer/credit recognition/eligibility and comparison queries remain
ordinary paths, not a general institution or process classifier.

The expectation labels are manually declared inputs, not a duplicate production
parser. `faq_evidence_scope_cases.py` loads/validates them and expands public groups;
it never derives a runtime constraint from query text. No private emails, chats,
production conversation IDs or LLM judgments are included.

## Public evidence and preflight

`faq_evidence_scope_facts.json` defines five actual current FAQ titles, reviewed
public quotations and exact standalone body hashes from the verified P1 corpus.
Current entity bodies support Student Service/Dean/Computer Center. The public
Archive body matches its verified current standalone hash. The General Contact
title/current hash and exact public email quote are now checked against the
authoritative 2026-10-14 00:32:57 UTC public-body artifact (see checkpoint below).
The frozen catalog's earlier pending review labels remain historical provenance;
its gold/hash was not edited. Before GPU execution, require unique current titles, matching reviewed
body hashes, every quoted fact and every positive token in actual public bodies.
Missing/changed evidence fails the gate; original thesis/P1 gold is not rewritten.

These hashes are `md5(content)` consistency fingerprints, not the original
manifest's differently composed row hashes. Quote presence is not entailment,
current institutional approval or answer accuracy. The Student Service location
check uses its full specific sentence, not bare `117`, which could be a legal
article number. Public generic email/phone strings are diagnostic references,
never global production keyword filters.

## Production-path observation

Retain the P1 runner's read-only/central SQL-failure accounting, GPU-only HTTP
allowlist, sequential/case/candidate/time budgets and no-generated-answer scope.
Do not inject a target, canonical query, candidate or score. Capture the actual
filter input/output separately from `_build_candidates`: the reranker indexes
must refer to the **post-filter pool**, not the old pre-merge observation.
For the P1 baseline without the filter hook, post-filter pool equals its actual
ordinary pool; unavailable constraint diagnostics must be explicit.

Save before/after keys, excluded keys/counts, actual constraint/strict-mode
diagnostics, exact emitted rerank query/scores, selected/visible IDs and known
forbidden fact hits computed from the **whole returned context**. The fact-hit
checks are evaluation only. Preserve original query constraints across mocked
contextualization/rewrite/HyDE tests; the legacy harness has no conversation
history and must call the shared production parser/filter on its original input,
not recreate routing logic or parse a rewrite instead.

Comparison gates:
- Every required guard excludes declared forbidden evidence at post-filter,
  selected-context and visible-source stages; curated facts must not reappear in
  unrelated document blocks. Remaining relevant non-forbidden documents may stay.
- An allowed-other entity already present before filtering cannot be removed by
  an unrelated exclusion. Context/source checks respect actual confidence floors:
  no forced low-score answer or visible citation to meet a quota.
- With active constraints, all remaining candidates below 0.1, reranker failure or
  invalid response must not trigger keep-best or unscored dense fallback. Verify
  these branches offline with mocks, not intentional faults in live quality runs.
- Ordinary controls must keep candidate/final/visible identities and context
  content unchanged, with no active restriction; source/fact accuracy is untested.
- P1 preservation uses its unchanged fixture and gate; scope improvements may
  reduce its known 3/18 leakage but must not rewrite the dated P1 report.
- Any execution/SQL/HTTP fault, missing accounting, corpus/code drift or mixed
  runtime settings invalidates a comparison. No silent denominator reduction.

Report all 38 cases and families, splits and before/after fixes/regressions.
Multiple spellings and diagnostic holdouts are not blinded/independent tasks.
Candidate recall, constraint filtering, final context, literal fact absence,
confidence and source visibility are separate measures—not LLM answer accuracy.
Do not create a measured `faq_evidence_scope_report.md` until actual authorized
replays exist. Remote/GPU execution remains the independent verifier's ownership.

## Frozen fixture identities

Frozen fixture SHA-256:
`a8e73330d4ef0bf90ac5cb76ff74cc5aa04f29c95c54715dfc1e124bd40005a6`.
Public catalog SHA-256:
`8cfc15d01b1efab984da248a5ebbf681d504e2394cdbe6a425cdf47dcb8433e3`.
P1 fixture SHA-256 remains:
`1ffcb8b32ca97f327bb1d483ddb088f20ed3da2afce0cf7334961aec8152e1b5`.

Producer signatures are finalized; runner/comparison and legacy parity consume
them directly. `faq_evidence_scope_checks.py` provides passive content-free
pool/result serialization and pure per-case paired assertions. No remote benchmark
was executed by this eval-writer lane; the independent verifier produced the
measured artifacts linked above.
The runner patches `context.apply_faq_constraints_to_candidates` conditionally and
supplies actual pre/post filter objects, whole returned
context and sanitized shared-parser diagnostics (`active`, `strict_abstention`).
Those two keys are normalized from `faq_evidence_constraint_applied` under the
producer's active-constraint strict-abstention contract. A P1 baseline with no hook has identical before/after pools and
empty/unavailable diagnostics; a candidate with no observed hook cannot pass.
The scope comparator validates paired manifests, zero-fault ledgers, fixture
completeness, actual pool/exclusion/diagnostic consistency and positive-content
non-regression; the runner validates rerank input mapping and public token gold.
No content credit is inferred
from title presence, and forbidden quote checks inspect all returned text.

## Runnable CLIs (independent verifier only)

From each verified isolated arm's API root, using the **frozen wired runner** and
authorized read-only DB/GPU environment inherited in-memory. The evidence path
below contains the actual public manifests. Choose a separate existing output
directory (`$out`); do not overwrite the saved evidence or execute both arms
against the current working-tree import. These commands are for authorized replay,
not instructions to deploy or generate answers.

```powershell
$evidence = 'C:/Users/Stefan/AppData/Local/Temp/opencode/faq-evidence-scope-verified-20261014T013144Z'
$out = 'NEW_APPROVED_OUTPUT_DIRECTORY'
$common = @('--corpus-manifest', "$evidence/frozen-corpus.json", '--gpu-manifest', "$evidence/gpu-identity.json", '--mode', 'raw')
# Verified isolated PR1114 post-review baseline import: 819cbd720ae174a7e75506c3e606ef7844427ef8
python -B -m tests.eval.robust_faq_replay @common --fixture tests/eval/faq_evidence_scope_golden.jsonl --scope-catalog tests/eval/faq_evidence_scope_facts.json --revision-label PR1114-819cbd7 --output "$out/baseline-scope-raw.json"
python -B -m tests.eval.robust_faq_replay @common --revision-label PR1114-819cbd7 --output "$out/baseline-preservation-raw.json"
# Switch to the verified isolated frozen P2 candidate import; keep identical dependencies/GPU/corpus/runner.
python -B -m tests.eval.robust_faq_replay @common --fixture tests/eval/faq_evidence_scope_golden.jsonl --scope-catalog tests/eval/faq_evidence_scope_facts.json --revision-label P2-frozen-20733649 --output "$out/candidate-scope-raw.json"
python -B -m tests.eval.robust_faq_replay @common --revision-label P2-frozen-20733649 --output "$out/candidate-preservation-raw.json"
python -B -m tests.eval.faq_evidence_scope_compare --baseline "$out/baseline-scope-raw.json" --candidate "$out/candidate-scope-raw.json" --preservation-baseline "$out/baseline-preservation-raw.json" --preservation-candidate "$out/candidate-preservation-raw.json"
# Read-only recomparison of saved evidence requires no DB/GPU/provider calls:
python -B -m tests.eval.faq_evidence_scope_compare --baseline "$evidence/baseline-scope-raw.json" --candidate "$evidence/candidate-scope-raw.json" --preservation-baseline "$evidence/baseline-preservation-raw.json" --preservation-candidate "$evidence/candidate-preservation-raw.json"
```

Use the same new runner for both isolated application imports, without changing
the application baseline or original fixtures. These are two separate 38-case pairs, not a rerun of
the historical P1 study against its old baseline. Comparisons require matching
runtime settings and the same per-arm source hashes across scope/preservation;
source hashes intentionally differ between P1 and P2. Exit codes: 0 combined gate
passed, 1 valid pair with release failures, 2 invalid evidence. Scope-only
`compare_scope()` reports `scope_gate_pass`, never a standalone release approval.

Legacy `run_eval.py` resolves constraints from original `ex.query` before variant
construction in every mode, calls the same pure production filter before rerank,
and uses the shared finite-score/index validator. Active constraints disable
keep-best below-floor fallback and abstain on rerank failure/invalid response.
Only content-free reason/count/applied/abstained metadata is added to legacy JSON;
private raw queries are not exported. This legacy harness has no conversation
history, does not observe independent link enrichment, and is not the default
release benchmark. Optional pre-existing canonical search behavior is unchanged.

The legacy `faq_evidence_constrained_abstained` diagnostic follows actual denial
stages, not the expected anchor's hit/rank: it is true when filtering empties a
previously nonempty pool or constrained reranking leaves no eligible candidates.
An initially empty retrieval, a missed target with other accepted evidence, or a
zero selection budget alone does not count as constraint-induced abstention.
Catalog failures remain execution errors in this benchmark harness.

Parent-reported independent post-rebase checks: **436 passed**, API Ruff clean,
API formatting clean across **274 files**, and mypy clean across **269 files**.
This final lane step changed only documentation and ignored local manifests, so
it did not repeat those tests. Earlier lane checks (202 eval tests) are historical.
Both CLI help surfaces were previously exercised offline.
Evaluation code hashes are captured in `manifest.evaluation_source_sha256` and
must match across both arms and scope/preservation pairs; changes during a replay
invalidate it. The offline checkpoint below establishes preparation only; the
linked measured report separately records the actual passed retrieval gates.

Post-rebase wired SHA-256, freshly recaptured (unchanged from previous checkpoint):
- `robust_faq_replay.py`: `427b4ab83aba1740b2a90bb323570b55270952f0d68c5da7c7438cd1e60a8b5d`
- `faq_evidence_scope_compare.py`: `64cfee73c101ae1c014d5771b08040ec1dc1d774c0375edd35ef0f339df1db06`
- `run_eval.py`: `1478f641e82ffae49af4141a87a8291788680e57b7497be9cdc181bae470a3a5`

## Current offline validation checkpoint

The final producer fixes resolve all previous blockers. Fresh validation against
all 73 current public titles passes **14/14 required guards** (10 institution
guards, 4 exclusions), **12/12 local controls** (5 unrestricted,
7 compatible FINKI constraints with correct targets), **10/10 ordinary controls**
(all unrestricted), and **2/2 missing-fact controls** (unrestricted, no accuracy
credit). Shared pure-filter checks on the five declared public office controls
also pass all 38 cases: no forbidden FAQ/curated fact survives a required guard,
allowed alternative offices survive, and all controls retain compatible offices.
General Contact is removed as a direct fact carrier for Student/Dean exclusion;
Archive's legitimate Dean-building reference remains permitted. These are pure
controls, not retrieved pools or observed final contexts. Frozen expectations and
quotations are unchanged. No known offline preparation blocker remains; actual
retrieval/source/fact gates subsequently passed in the linked independent study;
their success is not inferred from this earlier pure-parser/filter checkpoint.

Local content-free artifact: `faq_evidence_scope_validation.json` (ignored result,
not committed). It records every case's actual shared-parser result, source/code
hashes, five body checks and 11 exact-quote checks. Existing preflight validator
passes with **zero errors/pending items** against:
- `C:/Users/Stefan/AppData/Local/Temp/opencode/faq-evidence-scope-office-subset-20261014T003257Z.json`
- `C:/Users/Stefan/AppData/Local/Temp/opencode/faq-evidence-scope-corpus-20261014T003257Z.json`

General Contact: `contact@finki.ukim.mk` is present; standalone body MD5 is
`35244932c4341a2ab32cce2de4f4d2c4`, SHA-256
`6d5bcf68a22b3900bd2c47009ea1556927c0c2622318b077a4ac70bc284e9e88`.
All five actual standalone MD5 values match frozen gold and their reported public
SHA-256 values match the supplied bodies. No catalog/fixture provenance or gold
changed, so old/new fixture and catalog hashes are identical to those above.
The General Contact body also repeats the Student Service phone `070 302 440`;
the existing whole-context fact scan remains mandatory even when the Student FAQ
itself was excluded. This validation does not establish that such a substitute
was selected or leaked in an actual run.

Historical P1 retrieval evidence was not rerun. Reviewed FAQ filtering does not exclude all document
chunks or the independent link catalog; whole-context fact checks can reveal
remaining leakage, but no full-prompt/answer-accuracy claim follows.

Offline preflight (no DB/GPU/provider access), from the API root:

```powershell
$corpus = 'C:/Users/Stefan/AppData/Local/Temp/opencode/faq-evidence-scope-corpus-20261014T003257Z.json'
$bodies = 'C:/Users/Stefan/AppData/Local/Temp/opencode/faq-evidence-scope-office-subset-20261014T003257Z.json'
python -B -c "import json,sys; from pathlib import Path; from tests.eval.faq_evidence_scope_cases import load_scope_cases,load_fact_catalog,preflight_scope_cases; print(json.dumps(preflight_scope_cases(load_scope_cases(),load_fact_catalog(),json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig')),json.loads(Path(sys.argv[2]).read_text(encoding='utf-8-sig')))))" $corpus $bodies
```

Expect `valid: true`, `errors: []`, `pending: []`; this is public-source consistency
only, not a benchmark release.

## Frozen candidate identity for verifier handoff

`faq_evidence_scope_candidate_app_manifest.json` is an ignored local manifest of
**131 candidate application Python files**, with app-relative POSIX paths and raw
file SHA-256 values. Its source-map digest uses `robust_faq_replay.digest()`:
`2073364984ecf785730481cc3838ed0368c2afb8927c67f4e64affc43ef12d10`.
Manifest-file SHA-256:
`ff9cef7596c527a861ba762dae7b0e2916f3a82f722b94568f49823a708ddb2b`.
Current producer parser SHA-256:
`3175da6f7c3a7115488d57b271e91c644a0c5e4f2cba82060ea088d94b84cd0f`.
Updated validation artifact SHA-256:
`dc8acfadf0c167458778b70d4276b185cf4b97763ea099f5b2d921abee315f91`.

The verifier matched these inputs for the measured study; hashes alone do not
establish execution or gate success. **This is the frozen benchmark source identity,
not the current source-map digest after any later main/BYOK rebase.** The application baseline
is immutable `819cbd7`, and its own source manifest must be measured separately.
Any relevant code, fixture, catalog, corpus or runtime change requires fresh
evidence rather than reuse of this checkpoint.
