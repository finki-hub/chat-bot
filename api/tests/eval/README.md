# Retrieval evaluation harness

A golden set + runner for measuring **retrieval quality** of the FINKI RAG pipeline,
so threshold / embedding-model / prompt changes can be compared objectively instead of
by hand. This is what would have caught the "praksa" regression automatically: a
genuinely relevant source that the distance pre-filter silently dropped.

## What it measures

For every `(query -> expected source)` example, the runner drives the **real** retrieval
stack (`transform_query` rewrite + HyDE, multi-query embedding, pgvector ANN over FAQ +
chunks, cross-encoder rerank) and records:

| Metric | Meaning |
|---|---|
| **ANN recall (ideal)** | expected source is in the top-N by raw cosine distance, **distance ceiling disabled** |
| **ANN recall (prod)** | expected source survives the production `MODEL_DISTANCE_THRESHOLDS` cutoff |
| **ceiling drops** | ideal-hit but past the distance cutoff — the source was relevant, but the ceiling discarded it. **The praksa class. A healthy config keeps this at 0.** |
| **k-budget drops** | ideal-hit but crowded out of the per-query-k pool (closer competitors filled the budget) |
| **final recall@k** | expected source is in the reranked, min-score-filtered context the LLM actually sees |
| **MRR** | mean reciprocal rank of the expected source after reranking |

Results are broken down by `source` (faq/chunk) and `difficulty` (easy/hard), and every
final-recall miss is printed with a tag (`CEILING-DROP` / `KBUDGET-DROP` / `ANN-MISS` /
`RERANK-MISS`) so failures are diagnosable, not just counted.

## Dataset — `golden.jsonl`

One JSON object per line. Ground truth is anchored on **stable natural keys** (not row
UUIDs) so the set survives re-ingestion:

```json
{"id": "faq-012", "query": "Колку чини дупликат индекс?", "anchor": {"type": "Q", "name": "Административна такса / Таксена марка"}, "category": "школарина", "difficulty": "easy"}
{"id": "chunk-031", "query": "Кој ја формира комисијата за избор на ректор?", "anchor": {"type": "C", "document_name": "264-statut-ukim-6-6-2019", "chunk_index": 187}, "category": "избори", "difficulty": "easy"}
{"id": "abstain-003", "query": "Кој игра во репрезентацијата?", "anchor": {"type": "none"}, "category": "вон-опсег", "difficulty": "abstain"}
```

* `anchor.type` `Q` → an FAQ row, keyed by its unique `name`.
* `anchor.type` `C` → a document chunk, keyed by `(document_name, chunk_index)` (the unique index).
* `anchor.type` `none` → out-of-scope / prompt-injection; ideally retrieves nothing relevant.

The set was generated from the live corpus (74 FAQ + a stratified sample of substantive
chunks across all 19 documents): an LLM wrote natural Macedonian-Cyrillic student
questions per source, then an independent adversarial reviewer kept only those genuinely
answerable from that exact source. `easy` = direct paraphrase; `hard` = synonyms /
abbreviations / scenario phrasing with minimal lexical overlap (semantic stress test).

## Running it

Run inside the api container (it has `DATABASE_URL`, GPU/OpenAI access, and the app code):

```bash
# copy the harness in (or bind-mount tests/) then:
docker exec -e PYTHONPATH=/app finki-hub-chat-bot-api-1 \
    python /app/tests/eval/run_eval.py \
    --golden /app/tests/eval/golden.jsonl \
    --embedding-model BAAI/bge-m3
```

Compare embedders (this is the praksa A/B):

```bash
# ... --embedding-model text-embedding-3-large
```

Useful flags: `--no-transform` (embed the raw query only, to isolate retrieval from the
rewrite/HyDE step), `--top-k`, `--initial-k`, `--ideal-limit`, `--concurrency`,
`--json out.json` (per-example dump for diffing across runs).

## Extending the set

* Add real user questions from the api logs (`grep "Retrieving context for query"`) — the
  most representative source — and label the expected source by hand.
* Regenerate from the corpus when documents change; anchors are stable keys, so existing
  examples keep working unless the underlying document/section is removed.
* Keep `abstain` examples current with new out-of-scope / injection patterns.
