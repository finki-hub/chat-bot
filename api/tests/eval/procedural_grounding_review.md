# Procedural-grounding review — 2026-09-14

**Draft hardening, not a validated fix.** The final prompt still produced three
manual failures in seven frozen cases. The lexical scorer is not semantic truth:
all seven answers passed it in both baseline and final runs. A proposed set of
observed-phrase checks was removed because new unsupported paraphrases also passed.

## Controls and budget

- Base: `df163f5ce24bf4fdbfe31c91ec99430a32730e45` (merged PR #965).
- Returned model throughout: `gpt-5.4-nano-2026-03-17`; requested alias
  `gpt-5.4-nano`. Fixed prompt date: **2026-09-14**.
- Requested temperature 0.3, output cap 1,400 tokens, no explicit reasoning or
  top-p setting, web formatting, empty history and no tools. Direct development
  provider access, Responses API, nonstreaming, 45-second timeout, retries zero,
  provider storage disabled. Sampling is not claimed to be deterministic.
- Reused seven saved answers from the previous evaluation's candidate, generated
  against the exact base above. The runner verified unchanged fixture inputs,
  prompt builders, context formatters and web formatting against that commit.
- Two candidate iterations, **seven new calls each: 14 new calls total**. The second
  was the sole evidence-based prompt iteration; no changes to the final prompt
  after its evaluation. All calls completed on the same returned model.
- Both versions used normal production prompt builders and source formatters.
  Only each frozen case's query/context was supplied as task data, with the same
  date prefix; rubrics, evidence and expectations were withheld. This comparison
  isolates prompt policy on fixed excerpts, not retrieval selection or deployment.

## Manual results

Pass means the core case rubric is satisfied; partial means incomplete or confused
explanation without materially wrong main guidance; fail includes unsupported
concrete status instructions or a contradictory direct answer. The same strict
procedural-grounding standard was applied to both versions.

| Case (`corpus-` prefix omitted) | Reused baseline | First iteration | Final |
|---|---|---|---|
| faq-document-conflict | Partial | Partial | Fail |
| doctoral-current-scope | Pass | Pass | Pass |
| admission-historical-answer | Pass | Partial | Pass |
| fee-academic-year-scope | Partial | Partial | Pass |
| paper-electronic-procedure | Pass | Partial | Pass |
| incomplete-procedure-clarification | Fail | Fail | Fail |
| private-status-honesty | Fail | Fail | Fail |
| **Totals** | **3 pass / 2 partial / 2 fail** | **1 pass / 4 partial / 2 fail** | **4 pass / 0 partial / 3 fail** |

The first iteration omitted the historical application's email subject and receipt
confirmation; the final answer restored both. Final fee applicability preserves
“until replaced”, and the electronic-certificate answer retains zero fee, submission,
Student Affairs processing and automatic notarized-PDF delivery to student email.
It still omits the price-list “without comment” detail. No broad historical or
procedural overrefusal was observed. Generic follow-up offers remain common.

### Short actual failure excerpts

These excerpts are from fictional evaluation requests, contain no private records
or credentials, and are quoted only to explain review decisions. They are **not
correct procedural advice**. Full generated responses are not committed.

- **Baseline private status:** “Ако до крај на денот нема промена на статусот”
  introduced an unsupported waiting threshold. This wording disappeared in both
  new samples, but unsupported status advice remained.
- **Final incomplete request:** “статусот/напредокот на твоето барање во **iKnow**”
  recommends a status feature absent from the excerpt. The answer also fails to
  ask explicitly which certificate is intended, although it admits no turnaround
  time is supplied and refers to the responsible service.
- **Final private status:** “**статусот во iKnow** за истото барање” and
  “*„во обработка“/„завршено“*” introduce status visibility/labels not established
  by the source. The access disclaimer, public process, email delivery and office
  referral are correct, but do not make those additions grounded.
- **Final conflict:** opens “Да. Според изворите:” to whether prices are equal,
  then lists 100 denars for paper and zero for electronic in the price list. The
  contradictory opening fails direct-answer correctness, despite correctly
  describing the FAQ conflict. It also suggests checking the current price in
  iKnow without evidence for that feature.

## Reproduction and interpretation

Follow the README's fixed-date seven-case comparison procedure with a separately
authorized inference budget. Use the exact base prompt and the final prompt in this
change, identical model/settings and production formatting. Save outputs outside
version control. Run the existing scorer, then independently review every proposed
step against the frozen source and each rubric, including all previously good cases.
Baseline reuse is valid only while its model-input dependencies and settings match.

Original local evidence folders under the approved OpenCode temporary directory:

- `pr965-answer-eval/candidate.jsonl`: reused baseline.
- `procedural-grounding-eval/`: first iteration.
- `procedural-grounding-eval-final/`: final captures, prompt snapshot/hash, settings,
  attempt ledger, detailed review and token summary.

These temporary files are not required by tests and may not exist on another
machine. Historical comparison files there include the removed experimental
lexical checks; do not interpret their reported “fixes” as semantic improvements.

The 14 new calls used 36,574 input and 4,676 output tokens (41,250 total, including
23,296 cached input tokens); billed cost was not returned. Single samples on seven
narrow fixtures are not statistical quality estimates or provider-parity evidence.
No production chat quota, private tools, database retrieval or deployment checks
were used. The FAQ mutation is synthetic, and reconstructed fee-table excerpts
retain the corpus warning that original PDF verification is needed.

**Conclusion:** stronger instructions express the intended behavior, but the
observed model still invents procedural details. Local prompt/parser/formatter
tests establish policy text and fixture integrity, not model compliance. Keep this
as a draft with unresolved manual failures, not an accepted grounding fix.
