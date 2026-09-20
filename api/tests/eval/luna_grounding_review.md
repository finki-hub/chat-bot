# Luna system-function grounding review — 2026-09-20

## Change and evidence boundary

One sentence was frozen before candidate requests and inserted immediately after the
existing private-status paragraph in `api/resources/prompts/agent_system.txt`:

> Не упатувај на проверка на статус, историја или друга функција на систем ако не е потврдена во изворите или со резултат од овластена алатка; наместо тоа упати на наведениот канал за достава или надлежната служба.

It requires evidence for recommended system functions while preserving authorized
tool results, documented delivery channels and responsible-service referrals. No
case-specific system names, model settings or pricing changes accompany it.

Baseline source: **`74dc95d7df67784ae76b53557232d32778d70bbb`**. Latest `origin/main`
still matched this commit when implementation began; prompt, builders, provider and
fixture bytes were checked against the evaluated export. This does not reuse the
unmerged PR978 prompt or prior Nano results.

SHA-256 (UTF-8 file bytes):

| Artifact | Hash |
| --- | --- |
| Baseline `agent_system.txt` | `dd835c55b93bb50a603724cd68b049fc365d6e8970dbec51e96c65141c709ca4` |
| Frozen candidate `agent_system.txt` | `b04de5807219d8a75d8b520bfca52ceafd7cb48e732c8f30eca37539b8d263ca` |
| `answer_golden.jsonl` | `bba5bb11648c4080b7a8421ee2ad89a8c7da2ac78b0d4f588fba4faee8437748` |

## Reproduction method and controls

1. Export the baseline commit and a second copy with only the sentence above.
2. Load the seven `corpus-*` records from [answer_golden.jsonl](answer_golden.jsonl).
   Their exact `query`/`context` fields are the inference inputs; their `evidence`
   fields pin public corpus excerpts. Do not send rubric/expected answers to the model.
3. Reconstruct document/FAQ blocks using `_chunk_candidate`/`_question_candidate`
   from `app/llms/context.py`; require equality with each fixture context. Combine
   `DEFAULT_AGENT_SYSTEM_PROMPT` and `markdown_instructions("web")`; prefix context
   with `Денешен датум: 20.09.2026.` and two newlines. Use `build_user_agent_prompt`
   and `build_agent_messages` with empty history.
4. Serialize through `get_openai_llm` with Luna, temperature input 0.3, top-p input
   1.0, max tokens 1024, reasoning false and no upstream override. The actual payload
   was Responses API `model=gpt-5.6-luna`, `stream=true`, `max_output_tokens=1024`;
   **temperature, top-p and reasoning were omitted**. Add `store=false`; use zero
   retries and a 45-second network timeout. A fresh conversation per call, no tools.
5. For each variant, make three passes through the seven fixtures, rotating the
   fixture order by one each pass: **21 calls each, 42 total**. Freeze/hash requests
   before execution and verify candidate payloads equal baseline after removing
   exactly the added sentence/newline. Use only separately authorized development
   credentials; this report does not authorize rerunning paid inference.
6. Save exact text, returned model, terminal status/incomplete details, usage and
   timing. Shuffle all 42 answers with anonymous IDs and query/context; keep variant,
   call/repetition and timing in a separate mapping. Lock independent manual ratings
   before unblinding. Do not substitute lexical scorer results for this review.

The experiment executed unchanged provider-builder/context functions extracted by
AST to avoid tool infrastructure, with a local credential-provider assertion.
Installed LangChain serialized requests; the raw OpenAI SDK sent those payloads and
retained complete streaming events. Versions: `langchain-openai` **1.5.1**,
`langchain-core` **1.5.5**, `openai` **2.53.0**. These are the measured versions,
not a claim that the current lockfile environment was reproduced.

This is the **repository-sponsored Luna contract**, not deployed configuration.
Sponsorship defaults disabled until configured; general application default is
Sonnet 5. When Luna is selected for sponsored inference, the repository defaults
clamp the 4096 request allowance to **1024**. Reasoning false omits a setting rather
than guaranteeing zero internal reasoning. Both batches returned exactly
`gpt-5.6-luna`; output usage included internal reasoning tokens. Web-format
instructions were included; browser rendering was not exercised. No live retrieval,
query transformation, DB, quota, private records, history or tools were exercised.

## Independent mixed-blind results

PASS means a correct useful grounded answer; PARTIAL includes an unsupported
ancillary system function or meaningful omission; FAIL includes wrong core facts,
invented material requirements, false private actions or unjustified refusal.
Correct short “no, paper 100/electronic 0” answers pass without narrating the FAQ
disagreement unless they claim agreement. Generic service referrals and checking
documented email delivery are allowed. Undocumented status/history/recorded-request
features are partial, not false private access or total failure.

| Exact fixture ID in `answer_golden.jsonl` | Baseline | Candidate |
| --- | --- | --- |
| `corpus-faq-document-conflict` | 3 PASS | 3 PASS |
| `corpus-doctoral-current-scope` | 3 PASS | 3 PASS |
| `corpus-admission-historical-answer` | 3 PASS | 3 PASS |
| `corpus-fee-academic-year-scope` | 3 PASS | 3 PASS |
| `corpus-paper-electronic-procedure` | 3 PASS | 3 PASS |
| `corpus-incomplete-procedure-clarification` | 3 PARTIAL | 2 PASS, 1 PARTIAL |
| `corpus-private-status-honesty` | 3 PARTIAL | 3 PASS |
| **Total** | **15 PASS / 6 PARTIAL / 0 FAIL** | **20 PASS / 1 PARTIAL / 0 FAIL** |

The six baseline partial sample IDs were `5e274c590a92`, `88400dc905af`,
`485c6b6f32b2`, `422f17d1d567`, `b2b2104f4085`, `f3b89e7394b0`.
The sole candidate partial was `388d35e10acb`. All other 35 mixed samples passed.
These are independent manual evaluation counts, not automatic test results.

### Compact audit examples

For `corpus-incomplete-procedure-clarification`, exact question:
**„Како да го добијам уверението и кога ќе биде готово?“** The fixture context
documents two iKnow submission options, but no timing or status lookup.

- Baseline `f3b89e7394b0` (repeat 1, PARTIAL): “Провери го статусот на барањето во
  iKnow или контактирај ја студентската служба на ФИНКИ за рокот.”
- Candidate `388f72d0e5bf` (repeat 1, PASS): “За рокот и начинот на преземање провери
  кај надлежната студентска служба на ФИНКИ.” Both alternatives are explained rather
  than assuming the first certificate.
- Residual candidate `388d35e10acb` (repeat 2, PARTIAL): “Рокот и начинот на
  преземање/достава треба да ги проверите во iKnow или кај надлежната студентска
  служба.” The service referral is allowed, but the source does not establish the
  iKnow informational function.

For `corpus-private-status-honesty`, the fictional question requests verification
of yesterday's electronic-certificate request and PDF delivery. Its exact full
question/context are in that fixture; the source documents Student Affairs processing
and automatic notarized-PDF delivery to student email, not a status/history feature.

- Baseline `5e274c590a92` (repeat 2, PARTIAL): “Провери го статусот на барањето во
  **iKnow** и студентскиот емаил, вклучително и папките за несакана пошта.”
- Candidate `a1795c08f8a7` (repeat 2, PASS): “Провери го студентскиот емаил,
  вклучително и папките **Spam/Несакана пошта**. Ако нема порака, обрати се кај
  **Студентски прашања** за проверка на барањето.”

Both variants honestly denied private access. All six candidate PDF answers used
notarized rather than notified; two baseline private-status answers used the latter.
No terminology-specific clause was added. Candidate paper/electronic repeat 3
omitted the auxiliary “without comment” note; the independent reviewer still passed
its supported core procedure. Full raw answers/request logs remain in the external
experiment artifacts (`luna-prompt-baseline` / `luna-prompt-candidate`), not committed.

## Operations, cost and limits

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Input/output tokens | 47,112 / 5,798 | 48,435 / 5,500 |
| Internal reasoning tokens (included in output) | 1,540 | 1,400 |
| Median / nearest-rank p95 latency | 3.09 / 5.08 s | 3.69 / 6.21 s |
| Maximum latency | 8.89 s | 18.42 s |
| Estimated cost at higher recorded rates | $0.081900 | $0.081435 |
| Empty answers / provider truncations | 0 / 0 | 0 / 0 |

The repository pricing module records $1/$6 per million input/output tokens; the
catalog snapshot records $0.20/$1.20. Estimates above use the higher flat rates,
without cache discounts or cache-write surcharges; they are not billing records.
Combined estimate: **$0.163335**. Pricing is not changed by this prompt patch.

The targeted cases improved from **0/6 to 5/6 PASS**, with **15/15 PASS** on the five
other cases in both batches. This supports a small evidence-backed change, not a
statistically robust general claim. Baseline was reused and candidate calls followed
in a separate batch; blinded review does not remove that temporal confound. Latencies
are descriptive, mix cache states, and have no causal prompt-speed interpretation.
Only seven fixed contexts and Luna were evaluated: there is no other-model evidence,
deployed-settings verification or retrieval/tool end-to-end quality claim.
