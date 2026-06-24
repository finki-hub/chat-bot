# Diploma Mentor & Committee Recommender — Implementation Plan

## Context

**Need.** FINKI students proposing a diploma thesis must find a suitable mentor and two committee members. Today that's tribal knowledge. We have a clean historical record — every past/ongoing defense with its title, description, mentor, and two committee members — exposed at `GET https://diplomski-api.finki-hub.com/diplomas` (3,981 records; **3,548 defended** / `status='Одбрана'`; 126 distinct people, 68 mentors).

**Goal.** Given a proposed thesis (Macedonian/Cyrillic), recommend committee members grounded in the most similar historical defenses, in **two request shapes against one shared core**:

- **Mode FULL** — caller provides only the thesis **title**. Recommend a **mentor** *and* an **unordered pair** of two committee members.
- **Mode MEMBERS-ONLY** — caller provides the **title** *and* a known **mentor**. Recommend **only** the two committee members. The mentor is treated as **given**: excluded from the member candidate set, never returned as a recommendation, and (behind a default-off weight) used as a **signal** that up-weights retrieved defenses where that same mentor served — "who serves with this mentor" is exactly the pair the caller wants.

The user constraint, verbatim: *"either the title and mentor, or just the title."* The mode is **inferred from the payload, not from the route** — a single endpoint with an optional `mentor` field. Both modes funnel through the identical `embed → get_closest_diplomas → _post_rerank → score_people → select_committee` path; the only branch lives inside the PURE core. **There is no description on input** — the request carries a single free-text **title** (usually short, occasionally a longer line) plus an optional **mentor**. The query is embedded from the **bare title** (E5 `query:` prefix); stored *corpus* rows keep their richer `title + description` text (see Data model) — a deliberate, standard short-query-vs-rich-document asymmetry.

**Three scoring signals.** Recommendations blend (1) **historical-defense co-occurrence** (who has served together on similar topics) with (2) **topical paper-expertise** — derived from each professor's published papers (title + abstract), scraped into a `professor_document` corpus — and (3) **topic-conditioned co-authorship** ("buddies") — a recency-weighted co-author edge graph among professors, built from the **same retrieved papers** as the expertise signal. The expertise signal is the **cold-start fix**: a junior professor with few or zero past defenses still scores if their papers are topically close to the proposal. The buddy signal **extends the paper subsystem and does not replace expertise**: expertise scores *individuals* by paper-topic closeness; the buddy signal scores *pairs/edges* by topic-conditioned, recency-weighted co-authorship — surfacing pairs the defense graph cannot (co-published but never co-served). All three are additive and **default-off** (`expertise_weight=0.0`, `coauthor_weight=0.0`), so adding the corpus changes nothing numerically until the backtest turns them on — each is its own ablation pivot.

**Approach (settled over prior discussion).** Build it **inside the existing `api` service**, reusing its embed → pgvector KNN → gpu-api rerank pipeline. The recommender *is* that pipeline over new corpora plus a small pure scoring function, so a separate service would only fork plumbing the repo deliberately keeps unshared. **The `document`+`chunk` RAG layer already exists end-to-end** alongside `question` (schema, data layer, SSE fill, KNN+rerank retrieval, schemas, `/documents` API) — both new corpora (`diploma`, `professor_document`) are a **third and fourth copy-adapt of that proven pattern**, not greenfield. Identity is kept **simple**: the professor names from the diploma API are canonical (the source of truth) — no person/alias tables, no fuzzy matching, no normalization at the read path; external sources (papers) get a hand-maintained static `{external_author_id → canonical_name}` map and `api` only ever sees canonical names. The write path (ingestion/scraping) and read path (recommendation) stay separated, exactly as the FAQ/document corpora already separate `/fill-embeddings` from search.

**Professor-paper scraping is IN scope.** It runs as a **`scripts/` entrypoint, not a service** — the rule: split into a service only when ingestion must run scheduled/unattended with persistent retry/rate-limit state. Paper scraping is a periodic manual backfill (the paper analogue of `/diplomas/sync`), idempotent and resumable by `external_id` UPSERT, so the service threshold is not met.

**This plan's scope:** data model (diploma + professor_document + professor_group) → ingestion (admin endpoints + scraper script) → recommender read path (two modes, three signals) → **temporal collaboration groups (an offline community-detection job)** → REST endpoint → **leave-one-out backtest as the quality gate (defenses-only baseline, then defenses+papers ablation, both modes)**. Exposing it to the chat agent as a **native in-process tool** is an explicit follow-up *after* the backtest validates quality (see Out of Scope).

**No hard assumptions:** scoring weights, embedding model, retrieval breadth, status filtering, and the expertise blend are config-driven and resolved empirically in the backtest — see Open Decisions.

---

## Architecture & packaging

**Stays in the EXISTING `finki-hub-chat-bot` monorepo — NO new repo.** A separate repo is justified only by divergent ownership / release / licensing — not the case here.

- **Read path lives INSIDE the `api` service** (`recommend.py` PURE core + retrieval orchestrators + the `/recommendations` route), reusing its embed → pgvector KNN → gpu-api rerank pipeline.
- **Offline parts are `scripts/` jobs** — `scrape_professor_papers.py`, `compute_professor_groups.py`, `backtest.py`. Each builds its own `Database(dsn=Settings().DATABASE_URL)` exactly like `app/migrations.py` and writes tables the `api` reads. (Verified: `scripts/` already holds offline jobs — `convert_faqs.py`/`convert_links.py`/`convert_generic.py`.)
- **Raw/bulk data stays OUT of the code repo** — gitignored or a sibling local-only data repo, mirroring how the documents corpus is handled. Only **derived Postgres tables** + the committed `api/resources/professor_name_map.json` live with the code.
- **Promote scraping/groups to a `worker/` service (SAME monorepo: own `pyproject.toml` + Dockerfile + compose service + scheduler) ONLY when** ingestion must run scheduled/unattended with persistent retry/rate-limit state. Not met today (idempotent `external_id` UPSERT makes a script resumable without a state store).
- **Confirmed stack** (`compose.yaml`): `db` = `pgvector/pgvector:pg17` (the **ONLY** datastore — volume `./pgdata`; no Mongo service exists in compose, any `mongo-data/` dir is stale), `api` (FastAPI + asyncpg + httpx + langchain/langgraph + pydantic; **NO numeric/graph libs**), `gpu-api` (torch + sentence-transformers + transformers; sklearn transitively), `pgadmin`. This dependency-light `api` is why community detection must stay offline.

---

## Datasets & coverage

Four datasets feed the recommender; only the first is required for a shippable v1.

- **A. Diploma defenses** (`diplomski-api.finki-hub.com/diplomas`) — committee corpus; the **only source of canonical Cyrillic professor names**; backtest ground truth.
- **B. Professor papers** (OpenAlex **∪** UKIM-repo, deduped by DOI/title) — expertise + buddy + co-authorship-group signals.
- **C. Canonical professor name map** (`{ORCID/OpenAlex id → canonical Cyrillic name}`, `professor_name_map.json`) — a BUILT artifact, the cross-script JOIN KEY; gates **all** paper-derived signals; built from diploma Cyrillic (A) + UKIM-repo Latin/**FINKI-email** roster + OpenAlex ORCID, anchored by the email transliteration; seeded by the lifted name-fix maps (Prior art). (UKIM CRIS is Latin-only — Open Decision 18.)
- **D. FINKI roster** — the FCSE node set + Latin names/emails comes from **UKIM-repo CRIS Person entities (API)**; the `finki.ukim.mk/mk` HTML page is now **optional** (only for Cyrillic names of professors absent from the diploma data). Seeds C, onboards cold-start professors. → **no *required* HTML scraping.**

**Coverage verdict.** The defense subsystem (A) + all logic-only requirements are **fully satisfiable TODAY** (re-confirm plan-asserted counts on a full `/sync`). Paper-derived signals (expertise, buddy, co-authorship temporal groups) are mechanically buildable but their **quality is gated by two roots**: (i) cross-lingual MK↔EN retrieval efficacy (must be **proven in the backtest**; default-off until then) and (ii) name-map coverage (manual build; bounds buddy/group quality). Truly-new professors with **no defenses AND no mapped papers are dark** until hand-onboarded from roster D. **Both paper signals default OFF → ships safely as a defenses-only recommender.** **Co-service temporal groups need none of B/C** — they build from A alone (clean Cyrillic identity, no name map).

---

## Prior art & reusable resources

A Nov-2025 FINKI diploma thesis solves the **same problem with a different method**: *Avramoski, F. (211063), mentor Prof. Igor Mishkovski* — "Intelligent system for assigning mentors … using GNN." It trains a 3-layer **R-GCN** (PyTorch Geometric; 3 heads: mentor/C2/C3) over a heterogeneous professor–thesis graph. Public code, **no data**: `github.com/filipavramoski/Graph-Neural-Networks-Committee-Suggestion-System`. We reuse its **inputs and benchmark**, not its model.

**Benchmark to beat at GATE A** (their reported test metrics; edge-masking, 70/15/15 split):

| Role | Hits@1 | Hits@3 | MRR | AUC |
|---|---|---|---|---|
| Mentor | 0.66 | **0.97** | 0.81 | 0.84 |
| C2 | 0.48 | 0.93 | 0.70 | 0.74 |
| C3 | 0.59 | 0.96 | 0.77 | 0.81 |

Target ≈ **mentor Hits@3 0.97 / Hits@1 0.66**. *Not directly comparable:* they predict **ordered** C2/C3 with separate heads + edge-masking; we use **unordered**-pair Jaccard + leave-one-out over ~98–126 professors (their Hits@5 = 100% suggests a small/lenient candidate pool). Match-or-beat mentor Hits@k; report member-pair Jaccard separately.

**Design choices this independently confirms.** Same **two input modes** — *Option 1* (title → mentor + 2 members) = our FULL; *Option 2* (title + chosen mentor → restrict to that mentor's collaborators → 2 members) = our MEMBERS-ONLY. **But their "collaboration" edge = committee co-occurrence (served together)** = our `pair_affinity`. They do **not** model publication co-authorship — so our co-author **"buddy" signal is genuinely novel** here (their `research.csv` even collapses each paper to one professor, discarding the co-authorship present in the raw data).

**Discrepancy.** They embed the **English** thesis fields (`Thesis Title EN`/`Thesis_Desc_EN`) with `all-MiniLM-L6-v2` (384-d, English); we embed **Macedonian** with multilingual **BGE_M3** (1024-d). Their EN translations are a possible cross-lingual asset, not required.

**Reusable inputs (lifted from their public `structure/graph_structure.py` — no data needed)** — seed our canonical-name normalization (Open Decision 10) with these, then let the OpenAlex/ORCID name-map supersede them:

```python
# professor-name canonicalization (Cyrillic) — 11 fixes
research_name_fixes = {
    'Александра Каневче Дединец': 'Александра Дединец',
    'Александра Поповкса-Митровиќ': 'Александра Поповска-Митровиќ',
    'Бојан Илиоски': 'Бојан Илијоски',
    'Ефтим Здравески': 'Ефтим Здравевски',
    'Катерина Тројачанец Динева': 'Катарина Тројачанец',
    'Моника Симјаноска Мишева': 'Моника Симјаноска',
    'Наташа Стојановска-Илиевска': 'Наташа Илиевска',
    'Петар Секуловски': 'Петар Секулоски',
}
mentorship_name_fixes = {
    'Александра Поповска Митровиќ': 'Александра Поповска-Митровиќ',
    'Верица Бакева Смиљкова': 'Верица Бакева',
}
# excluded as not-current-FINKI-faculty (15); also dropped 'Андреј Ристески' from research
professors_to_remove = {
    'Ѓорѓи Ќосев - 3P Development', 'Жанета Попеска', 'Коста Митрески', 'Татјана Зорчец',
    'Филип Блажевски', 'Фросина Стојановска', 'Стефан Митески', 'Весна Киранџиска',
    'Милка Љончева', 'Александа Лозаноска', 'Славица Тасевска Николовска', 'Драган Михајлов',
    'Јозеф Шпилнер', 'Евгенија Крајчевска', 'Катерина Русевска',
}
```

Plus their preprocessing: **cap 100 papers/prof, 200 theses/prof** (popularity-bias control — aligns with our most-frequent-mentor baseline), **dedup on `(professor, title)`**.

**Their dataset schemas** (for reference / if we obtain the CSVs): `research.csv` (≈2,974 unique papers) = `Mentor`(=author), `Thesis name`, `Abstract`, `Field of study`, `Publication date`. `commettee_final.csv` (≈2,884 theses) = `Mentor`, `C2`, `C3`, `Thesis Title EN`, `Thesis_Desc_EN`, `Thesis Application Date`, `Thesis Status`, MK title/desc, `Student`.

**Get vs. generate — resolved: generate.** The CSVs are **not public** (gitignored; no releases, no dataset repo, no scraping script anywhere — `research.csv` provenance is undocumented). We generate both ourselves, no dependency on the author:
- **Committee data** ← our `diplomski-api.finki-hub.com/diplomas` `/sync` (Phase 2) — same FINKI source family as their `diplomski.finki.ukim.mk`.
- **Professor papers** ← our OpenAlex pipeline (Phase 5a–5b) — richer than their CSV (true co-authors + topics + year).

Requesting their files (email Filip Avramoski / Prof. Mishkovski; no contact on GitHub) is an optional **accelerator / cross-check** only.

---

## Data model

### `diploma` table

Append to `api/resources/schema.sql` as idempotent `IF NOT EXISTS` blocks (the file is executed wholesale by `app/migrations.py`; the `question`/`link`/`document`/`chunk` tables are untouched). One row = one defense = the natural retrieval unit. Committee members are stored as canonical-name TEXT; their *unordered* nature is enforced in scoring (`frozenset`), not DDL.

```sql
CREATE TABLE IF NOT EXISTS diploma (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id TEXT NOT NULL UNIQUE,     -- content hash (see idempotency note)
    title TEXT NOT NULL,
    description TEXT NOT NULL,             -- NOT NULL: corpus invariant (live API always supplies it)
    mentor TEXT NOT NULL,                 -- canonical name = source of truth
    member1 TEXT,                         -- canonical name
    member2 TEXT,                         -- canonical name
    status TEXT NOT NULL,                 -- 'Одбрана' etc.
    date_of_submission DATE,              -- parsed from 'DD.MM.YYYY'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS diploma_external_id_idx ON diploma (external_id);
CREATE INDEX IF NOT EXISTS diploma_mentor_idx ON diploma (mentor);

-- v1 embedding column, mirroring the question/chunk pattern exactly:
ALTER TABLE diploma ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector(1024);
CREATE INDEX IF NOT EXISTS diploma_embedding_bge_m3_idx
    ON diploma USING hnsw (embedding_bge_m3 vector_cosine_ops);
```

- **Idempotency key (resolved):** the API's `fileId` is null on 306 rows and so unusable as `NOT NULL UNIQUE`; `(student,title,date,description)` still collides on 1 (a genuine duplicate). Use `external_id = sha256(fileId|student|title|dateOfSubmission)`; the one true duplicate harmlessly collapses under UPSERT.
- **Embed `build_proposal_text(title, description)` once per model (corpus rows only)** (see the shared helper in Module decomposition). `description` is non-null in the corpus, so stored rows embed `f"{title}\n{description}"` as the *passage*. The **query side never builds this** — a recommendation query is the bare title (E5 `query:` prefix). `description` is therefore a **corpus-only field**: it lives on the table and in ingestion and **never appears on the request schema**. (Whether to instead embed diplomas title-only for query/corpus symmetry is an Open Decision, resolved by backtest; the default keeps the richer description.) Add the other per-model columns (3072 `halfvec_cosine_ops`, 8192 no-index) later — additive — only if the backtest picks a different/additional model. The exact column/index recipes to copy live at `api/resources/schema.sql` (question/chunk blocks).
- **Both diploma queries run off this one table:** similarity = KNN over `embedding_*`; committee co-occurrence/aggregation = pure-Python over the *retrieved subset* (kept in `recommend.py`, not SQL, so it's unit-testable and weight-swappable).

### `professor_document` table — paper-expertise corpus

Append to `api/resources/schema.sql` after the `diploma` block. **Decision: one row per paper, NOT chunked.** Grounded in the existing patterns:

- A paper's title+abstract is ~150–300 words — well under the e5 512-token window and the `chunking.py` `TARGET_CHARS=1300`. The `document`+`chunk` two-table split exists because Markdown docs are long; papers are not, so chunking would split nothing meaningful.
- One-row-per-paper makes the retrieval unit a single coherent topical statement, so KNN distance is directly "how close is this paper to the proposal" — exactly the per-professor aggregate signal we want.
- This therefore mirrors the **flat `question`** DDL recipe (one embed per row), not the `document`+`chunk` shape — keyed by canonical professor name.

```sql
CREATE TABLE IF NOT EXISTS professor_document (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id TEXT NOT NULL UNIQUE,        -- sha256(source|source_paper_id)  (idempotency, like diploma.external_id)
    professor TEXT NOT NULL,                  -- canonical Cyrillic name = source of truth (joins diploma.mentor/member*)
    source TEXT NOT NULL,                     -- 'openalex' | 'semantic_scholar' | 'ukim_repo'
    source_paper_id TEXT NOT NULL,            -- e.g. OpenAlex work id W...
    title TEXT NOT NULL,
    abstract TEXT,                            -- nullable: many works are title-only
    year INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS professor_document_external_id_idx ON professor_document (external_id);
CREATE INDEX IF NOT EXISTS professor_document_professor_idx ON professor_document (professor);

-- v1 embedding column, mirroring question/chunk/diploma exactly. The column NAME is shared across tables,
-- so MODEL_EMBEDDINGS_COLUMNS / HALFVEC_EMBEDDING_MODELS / MODEL_DISTANCE_THRESHOLDS / MODEL_EMBEDDING_DIMENSIONS
-- (all in api/app/llms/models.py) resolve UNCHANGED — no edit to models.py is needed:
ALTER TABLE professor_document ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector(1024);
CREATE INDEX IF NOT EXISTS professor_document_embedding_bge_m3_idx
    ON professor_document USING hnsw (embedding_bge_m3 vector_cosine_ops);

-- --- co-authorship ("buddy") signal: full author set + topics per paper ---
-- Additive, idempotent — same ADD COLUMN IF NOT EXISTS recipe as the embedding column.
-- Decision: array columns on this one-row-per-paper table, NOT a normalized paper+paper_author
-- join. The buddy query is pure-Python over the RETRIEVED subset (mirroring the diploma
-- co-occurrence rule above), so a relational author query is never needed at the read path;
-- a join would be the same structural surgery this plan rejected for papers (flat over chunked).

-- Full author set per paper. Canonical Cyrillic where the author maps to a FINKI professor
-- (via professor_name_map.json); raw OpenAlex display_name otherwise. Includes the row's own
-- `professor`. Only canonical entries participate in buddy scoring (set-intersection at read).
ALTER TABLE professor_document ADD COLUMN IF NOT EXISTS coauthors TEXT[] NOT NULL DEFAULT '{}';

-- Topic labels from OpenAlex topics[]/primary_topic (subfield display_name; NOT the
-- deprecated concepts[]). Stored for evidence/debuggability only — NOT load-bearing for
-- scoring: the buddy edge is topic-conditioned by the paper's KNN distance to the proposal
-- (the same s = 1 - distance the expertise signal uses), so re-deriving closeness from these
-- discrete strings would be a second, weaker topic metric. Kept open as a topic-faceted
-- backtest slice (Open Decision 14) without re-scraping.
ALTER TABLE professor_document ADD COLUMN IF NOT EXISTS topics TEXT[] NOT NULL DEFAULT '{}';
```

- **`professor` is canonical Cyrillic** — joins implicitly to `diploma.mentor`/`member1`/`member2`. No person/alias table (honors "canonical names are source-of-truth").
- **The row stays one-per-(paper, mapped professor); `coauthors` carries the full canonicalized author set.** Today a multi-FINKI-author paper produces N independent rows keyed `sha256(source|source_paper_id)` per mapped author, discarding the co-author fact. We keep that row shape — `professor` stays the canonical owner so `professor_document_professor_idx` and the per-individual expertise aggregate are untouched, and `external_id` + its UNIQUE/idempotency semantics are **unchanged** (still per-author row, so re-ingest dedupe is identical) — but each row now **also** carries the *full* author set in `coauthors`, canonicalized at **write** time (read path stays free of the name map, honoring "`api` only ever sees canonical names"). Non-FINKI co-authors are stored verbatim (graph completeness, debuggability) but never scored; the buddy edge is reconstructed in Python by intersecting `coauthors` against the canonical-name set within the retrieved subset.
- **Co-authors, abstract, and topics come from BOTH co-primaries, unioned + deduped.** OpenAlex supplies `authorships[]` (full co-author set), an inverted-index abstract, and `topics[]`/`primary_topic`; UKIM-repo supplies multi-`<dc:creator>`, a **plaintext** `<dc:description>` abstract, and multi-`<dc:subject>`. Rows are deduped by **DOI → normalized title** across sources, so `source ∈ {'openalex','ukim_repo','semantic_scholar'}` records *which* source produced the surviving row (source-weighting/dedup-key are Open Decision 17).
- Embed text = `build_proposal_text(title, abstract)` (the same corpus helper; abstract nullable → title-alone fallback). Reconstruct OpenAlex `abstract_inverted_index` → plaintext before storing; UKIM-repo `<dc:description>` is already plaintext.
- Additional per-model columns (3072 halfvec, 8192 no-index) are additive later, only if the backtest picks a different embedding model — identical policy to `diploma`.

### Paper corpus — sources

> **Provenance note (external-API facts below).** The OpenAlex `authorships[]`/`topics[]`/`primary_topic.subfield.display_name`/`publication_year` shapes, the "`concepts[]` deprecated" status, and the UKIM-repo OAI-PMH endpoint + multi-`dc:creator` record shape are **external API facts confirmed by the source-research pass that produced this plan**, not verifiable against this repo. They are recorded here as traceable design inputs; re-confirm against the live APIs before implementing the scraper (section "Professor papers" + `scripts/scrape_professor_papers.py`).

**Strategy: build the corpus as the UNION of OpenAlex + UKIM-repo (+ optional Crossref/DBLP/S2 gap-fill), deduped by DOI → normalized title.** Neither source is a subset of the other (verified this session): OpenAlex (institution `I76245029`) indexes **16,633** total works / **6,231** with the CS concept `C41008148`; the UKIM-repo FCSE community holds **~2,122** journal+conference papers (~1/3 of OpenAlex's UKIM-CS index) but adds local/Macedonian-language conference papers + pre-2018-index items OpenAlex misses. Per-author spot checks confirm divergence both ways (rough, mixed query methods: Kocarev OpenAlex 364 vs repo 170; Madjarov OpenAlex 46 by-ORCID vs repo ~101 surname-inflated; Zdravkova 67 vs ~127). **Truly exhaustive ("every paper ever by every professor") is UNATTAINABLE via APIs** — only Google Scholar approaches it and it is unscrapeable — so the target is a **best-effort, explicitly-bounded union**, NOT exhaustive coverage. The residual gap is the cold-start tail (junior / thinly-indexed professors stay sparse regardless of union), tied to the name-map bound (Open Decision 10).

- **Co-primary #1: OpenAlex** (`api.openalex.org`). The only source giving all of: abstracts at scale (≈79% recent-CS abstract coverage for UKIM `I76245029`; ~2,058 abstracted 2020+ CS works), a free unauthenticated REST API (polite pool via `mailto=`, ~10 req/s), and built-in ORCID author disambiguation with `display_name_alternatives`. **It also yields all three buddy inputs in the same call, no second pipeline:** each Work's `authorships[]` carries the **full** author list — *including non-UKIM co-authors* — each with a stable `author.id` (the disambiguating graph-node key, critical for Macedonian transliteration variants), ORCID, and institutions; `topics[]`/`primary_topic` give the hierarchical field labels (**build topic strings on `topics`/`primary_topic.subfield.display_name`, NOT the deprecated `concepts[]` — `concepts` is the disambiguation filter only, and is no longer maintained upstream**); `publication_year` gives the recency axis the buddy decay needs. Abstracts ship as `abstract_inverted_index` (positions, not plaintext) — reconstruct by sorting tokens by position. Filter at **UKIM** institution level (`I76245029`; no FINKI child entity exists), then narrow by resolved author id / ORCID.
- **Co-primary #2 (paper source + Cyrillic name bridge + co-author backfill): UKIM repository** (`repository.ukim.mk`, DSpace-CRIS) — **FINKI-run** (adminEmail `repository@finki.ukim.mk`), OAI-PMH 2.0 + OpenAIRE-CRIS, data since 2018. Verified this session: a dedicated **FCSE community** (`setSpec com_20.500.12188_5`) with real paper collections — **Journal Articles** (`col_20.500.12188_109`, 826 items), **Conference papers** (`col_20.500.12188_108`, 1,296 items), plus Books + PhD Theses → **~2,122 FCSE journal+conference papers**. Live OAI-PMH 2.0 at `https://repository.ukim.mk/server/oai/request` (the `/server/` path; bare `/oai/request` 404s). Per-record `oai_dc` fields confirmed present: multiple `<dc:creator>` (full co-author list), `<dc:description>` (**PLAINTEXT abstract** — not an inverted index, unlike OpenAlex), multiple `<dc:subject>` (keywords/topics), `<dc:date>` (year), `<dc:type>`, `<dc:identifier>` (handle); full ~34.5k harvest. Used as a **first-class paper source** in the union (not a last resort), **and** to *build the name map*, **and** to backfill local/Macedonian-language CS papers OpenAlex misses. **Caveat (the name-map bridge):** `oai_dc` author names are LATIN `"Last, First"` (e.g. `"Zdravkova, Katerina"`) with **no IDs**, so cross-record disambiguation is on us (exactly what OpenAlex's stable `author.id` solves) — OpenAlex stays the primary *edge* source. The DSpace-CRIS **Person entities** were probed this session (Open Decision 18, **RESOLVED**): they are **Latin-only** (`crisrp.name`/`dc.title`) with **FCSE affiliation + FINKI email** (`person.email`, e.g. `igor.mishkovski@finki.ukim.mk`) and sometimes a Scopus author-id — **no Cyrillic, no ORCID**. So CRIS is *not* the Cyrillic/ORCID bridge; it's a clean **Latin + email FCSE roster** (the email is the name map's deterministic Latin anchor — see Professor name map). Use the canonical `repository.ukim.mk` host; the *repository* host `repository.finki.ukim.mk` has a mismatched TLS cert.
- **Fallback (abstract enrichment): Semantic Scholar Graph API** — plaintext `abstract` for works neither co-primary covers. A per-author backoff step, **not** a bulk crawler (it 429s under unauthenticated load; get an API key before relying on it).
- **finki.ukim.mk staff/bio pages (OPTIONAL — HTML scrape, now mostly unnecessary).** The FCSE roster + Latin names + emails already come from UKIM-repo CRIS Person entities via API (Open Decision 18), and Cyrillic names for the ~126 who've served come from the `diploma` table — so the public staff site is only a fallback for **Cyrillic names of professors who never appear in the diploma data** (cold-start juniors), which transliteration + the email anchor often cover anyway. **Not a per-paper source.** Use `/mk/` pages for Cyrillic. Distinct from the `repository.finki.ukim.mk` TLS note above (the *repository* host, not the public *staff* site). → the pipeline has **no required HTML scraping**.
- **DBLP** — identity/works cross-check only (no abstracts, no usable affiliation).
- **Avoid Google Scholar / ResearchGate.** Scholar has **no official API**; the `scholarly` scraper triggers CAPTCHAs/IP blocks on sustained `search_pubs`/`citedby`, needs paid proxies, and ships no licensed abstracts — non-reproducible. For the buddy graph specifically it is doubly unfit: per-paper full author lists are truncated in listings, co-authors appear only as a hand-curated profile sidebar (not per-paper), and topics are profile-level free-text only. OpenAlex already ingests Crossref/MAG, so an API path dominates a scrape on coverage, legality, and determinism. **We scrape an API, not a webpage.**
- **Avoid LinkedIn (not viable for automation).** A bios/employment source, not a papers source: the optional "Publications" section is sparse free-text with no structured per-paper co-authors/topics; the API is approval-gated, denies research-enrichment use, and caps member-data retention (~48h). Useful at most as a side-channel for a professor's *current title*, never for the edge graph.

### Professor name map (Cyrillic ↔ Latin)

The read path needs canonical Cyrillic names; OpenAlex returns Latin. The bridge is a committed JSON, **`api/resources/professor_name_map.json`**, shape `{ "<orcid_or_openalex_author_id>": "<canonical Cyrillic name>" }`. Canonical names are drawn **only** from the distinct professor set already in the `diploma` table (`mentor` ∪ `member1` ∪ `member2`) — the source of truth. 126 distinct people / 68 mentors → a few-hundred-line file, tractable by hand.

How it's built (one-time, semi-automated, human-verified tail):

1. **Cyrillic** canonical names come from the `diploma` table (A) — already in hand for the ~126 who've served. (UKIM CRIS does **not** supply Cyrillic — Open Decision 18.)
2. From **UKIM-repo CRIS Person entities** pull the FCSE roster → `(Latin name, FINKI email, affiliation, [scopus-author-id])`. The **FINKI email is the deterministic Latin anchor**: `Игор Мишковски` transliterates to `igor.mishkovski`, matching `igor.mishkovski@finki.ukim.mk` — a far stronger Cyrillic↔Latin link than blind fuzzy matching.
3. Resolve each to an **OpenAlex author for the ORCID** (OpenAlex is the only ORCID source here): match by Latin name + `display_name_alternatives` under a **hard UKIM-affiliation constraint** (`authorships.institutions.id = I76245029`) + CS-concept to kill homonyms; cross-check via the professor's repo DOIs where ambiguous. CyrTranslit Macedonian (ж→zh/j, џ→dj/dzh, ќ→k/kj, ѓ→g/gj) seeds candidate Latin spellings.
4. **Anything not confidently resolved is left out of the map and its papers are DROPPED — never guessed.** A wrong canonical name silently corrupts a professor's expertise profile, which is worse than a missing one (the defense signal still covers established professors). The uncertain tail (tens of names) gets a cheap human pass.

At ingest the scraper consults the map and emits `professor = map[author_id]`; authors absent from the map are skipped with a logged count. The committed map pins `(canonical → ORCID → openalex_author_id)`, so re-runs are deterministic and don't re-disambiguate.

**The same map drives the buddy graph — applied to *every* `authorship.author.id` in a Work, not just the queried author.** Co-authors that map to a canonical name become buddy-graph nodes; an authorship whose id isn't in the map stays as raw Latin in `coauthors` and is simply excluded from the canonical-set intersection at scoring time (drop-never-guess, applied to the edge graph). **Honesty caveat — paper coverage is best-effort union, bounded, not exhaustive:** the corpus is a best-effort OpenAlex ∪ UKIM-repo union (deduped), and buddy/expertise coverage is further bounded by the name map, which by construction holds only confidently-resolved professors. A junior / transliteration-tail professor absent from the map contributes no expertise *and* no buddy edges until hand-added — exactly the cold-start cohort the feature targets — so the human-verified-tail size (Open Decision 10) directly bounds buddy/cold-start coverage.

### `professor_group` table — precomputed temporal collaboration groups

Append to `api/resources/schema.sql` after the `professor_document` block, same idempotent `IF NOT EXISTS` recipe. **Written offline by `scripts/compute_professor_groups.py`, read by `api/app/data/professor_groups.py`** — the `api` never computes communities (see "Temporal collaboration groups"). One row = one professor's membership in one community, in one time window. **No embedding column** (the only new table without one).

```sql
CREATE TABLE IF NOT EXISTS professor_group (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    window TEXT NOT NULL,                 -- end-year or 'YYYY-YYYY' bucket
    professor TEXT NOT NULL,              -- canonical Cyrillic (joins diploma.*/professor_document.professor)
    group_id INTEGER NOT NULL,            -- community label within the window
    edge_basis TEXT NOT NULL,            -- 'coauthorship' | 'coservice' | 'blend'
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS professor_group_key_idx
    ON professor_group (window, professor, edge_basis);   -- idempotent UPSERT key
CREATE INDEX IF NOT EXISTS professor_group_window_idx ON professor_group (window);
CREATE INDEX IF NOT EXISTS professor_group_professor_idx ON professor_group (professor);
```

- **`professor` is canonical Cyrillic** — joins implicitly to `diploma.*`/`professor_document.professor`, no person/alias table (same identity rule as everywhere else).
- **Idempotent UPSERT on `(window, professor, edge_basis)`** — re-running the offline job overwrites a window's labels in place; `group_id` is window-local (community identity across windows is tracked by membership overlap, not by a shared id — see the temporal-groups section).
- `edge_basis` distinguishes the two derivable graphs (`coservice` from `diploma`, `coauthorship` from `professor_document`) and an optional `blend` — so all three can coexist for the same `(window, professor)` and be backtested separately.

---

## Module decomposition

### Corpus embed-text helper (replaces the original's hard-coded `f"{title}\n{description}"`)

```python
# api/app/recommenders/text.py  (PURE)
def build_proposal_text(title: str, description: str | None) -> str:
    title = title.strip()
    description = (description or "").strip()
    return f"{title}\n{description}" if description else title  # title alone — no trailing "\n"
```

Called at the **two corpus-ingest sites** — diploma ingest (`build_proposal_text(title, description)`, description always present) and paper ingest (`build_proposal_text(title, abstract)`, abstract nullable → title alone). The **query path does not call it**: a recommendation embeds the bare `title` (E5 `query:` prefix). So index text stays consistent across initial-ingest and re-ingest, while the query-vs-corpus asymmetry (short title vs. rich passage) is deliberate.

### New files

| File | Purpose | Purity |
|---|---|---|
| `api/app/recommenders/text.py` | `build_proposal_text(title, description \| None)` — corpus-ingest embed-text builder (diploma title+description, paper title+abstract); **not** used for queries | PURE |
| `api/app/schemas/diplomas.py` | `DiplomaSchema` (with `distance: float \| None`, like `QuestionSchema`); `RecommendationRequestSchema` (`title` required, `mentor` optional, `mentor_topk`; **no description**); `PersonScoreSchema` (name + blended `score` + `defense_score`/`expertise_score`/optional `buddy_score` components + supporting ids); `RecommendationResponseSchema` (`mode`, `mentor_is_given`, mentor slot + unordered member pair, evidence ids) | I/O |
| `api/app/recommenders/recommend.py` | **PURE** core (no FastAPI/db/httpx): `Mode` enum, `ExpertiseIndex`/`CoauthorIndex`/`RankedPeople`/`Recommendation` dataclasses, `score_people(retrieved, expertise, weights, mode, given_mentor=None, coauthors=None)`, `select_committee(ranked, mode, given_mentor, *, mentor_topk, exclude=())`, plus `_minmax`, `_defense_counts`, `_upweight_defenses_with_mentor`, `_accumulate_coauthor_edges`, and a **NEW** `_half_life_decay(delta_days, half_life_days)` recency kernel (shared by defense/expertise/buddy — no such helper exists today). Liftable + testable. | PURE |
| `api/app/recommenders/config.py` | `ScoringWeights` dataclass + defaults — the single home for tunables (defense + expertise + cold-start + members-only + co-author/buddy fields) | PURE |
| `api/app/data/diplomas.py` | Typed data layer (mirrors `questions.py`/`documents.py`): `upsert_diploma`, `get_closest_diplomas(db, embedded_query, model, limit, threshold, *, exclude_external_id=None)`, `get_diplomas_without_embeddings(db, model)`, `get_defended_external_ids(db)` | db-typed |
| `api/app/data/professor_documents.py` | `upsert_professor_document` (gains `coauthors`/`topics` params), `get_closest_professor_documents` (copy-adapt of `get_closest_chunks` — halfvec branch + threshold + `# noqa: S608`, now selecting `professor, title, distance, coauthors, year`), `fetch_professor_doc_rows_for_fill` (copy-adapt of `fetch_chunk_rows_for_fill`) | db-typed |
| `api/app/llms/diploma_retrieval.py` | `retrieve_similar_diplomas(...) -> list[RetrievedDiploma]` — diploma analogue of `get_retrieved_context`, returns **structured candidates** (row + similarity + rerank score), not a joined string | db+http |
| `api/app/llms/professor_retrieval.py` | `retrieve_expert_professors(...) -> tuple[ExpertiseIndex, CoauthorIndex]` — paper analogue of `diploma_retrieval`; **reuses the already-computed proposal embedding** (does not re-embed); takes `weights, now_year, canonical_set` and builds both indices in one retrieval pass | db(+http) |
| `api/app/schemas/professor_documents.py` | `ProfessorDocumentSchema` (with `distance: float \| None`, like `ChunkSchema`; gains `coauthors: list[str]` / `topics: list[str]`); `FillProfessorEmbeddingsSchema` (mirror `FillChunkEmbeddingsSchema`) | I/O |
| `api/app/api/recommendations.py` | Router `/recommendations`, `POST /` → embed → retrieve (defenses + papers) → score → select; mode-aware response | FastAPI |
| `api/app/api/diplomas.py` | Admin router `/diplomas`: `POST /sync` (fetch live API → upsert) + `POST /fill-embeddings` (SSE backfill), both behind `verify_api_key` | FastAPI |
| `api/app/api/professor_documents.py` | Admin router `/professor-documents`: `POST /fill-embeddings` (SSE backfill) behind `verify_api_key` | FastAPI |
| `api/app/data/professor_groups.py` | Thin reader for the precomputed `professor_group` table (latest/active groups by `window`/`professor`/`edge_basis`); **no graph libraries** — the `api` only reads what the offline job wrote | db-typed |
| `api/resources/professor_name_map.json` | Hand-maintained `{external_author_id → canonical Cyrillic name}` | data |
| `scripts/scrape_professor_papers.py` | Offline scraper UNIONing **OpenAlex + UKIM-repo** (+ S2 abstract fallback), deduped by **DOI → normalized title** → `upsert_professor_document`; persists the **full** authorships + topics per Work (canonicalized via the name map); idempotent, resumable | script |
| `scripts/compute_professor_groups.py` | Offline temporal-community job: windowed edges (co-service from `diploma`, co-authorship from `professor_document`) → pure-Python label-propagation / greedy-modularity per window → cross-window Jaccard tracking → writes `professor_group`. `networkx` allowed here only (never in the `api`) | script |
| `scripts/backtest.py` | Standalone leave-one-out quality gate (pure read); dual-mode + nested defenses-vs-papers-vs-buddies ablation (`--no-papers`, `--no-buddies`) | script |

### Existing files to touch (minimal, additive)

- `api/resources/schema.sql` — append the `diploma` block, then the `professor_document` block, then the `professor_group` block, after the existing `document`/`chunk` block.
- `api/app/main.py` — `include_router(recommendations_router)` + `include_router(diplomas_router)` + `include_router(professor_documents_router)`, plus matching `openapi_tags` entries. Currently `main.py` registers **6** routers (`health_router:101`, `questions_router:102`, `documents_router:103`, `links_router:104`, `chat_router:105`, `feedback_router:106`) with **6** `openapi_tags` (`main.py:75-82`); mirror exactly that registration shape (e.g. the `documents_router` line at `main.py:103`).
- `api/app/utils/settings.py` — add typed class attributes (one each, like the existing `DATABASE_URL`/`GPU_API_URL`/`API_KEY`): `DIPLOMAS_API_URL: str = "https://diplomski-api.finki-hub.com/diplomas"`, `OPENALEX_MAILTO: str = ""`, `OPENALEX_BASE_URL: str = "https://api.openalex.org"`, `SEMANTIC_SCHOLAR_API_KEY: str = ""`, `UKIM_INSTITUTION_ID: str = "I76245029"`. (A corpus-embedding-model knob may instead live in `recommenders/config.py`.)
- `api/app/llms/models.py` — **NO change.** The shared `embedding_bge_m3` column name means `MODEL_EMBEDDINGS_COLUMNS` (`models.py:45`), `HALFVEC_EMBEDDING_MODELS` (`:115`), `MODEL_DISTANCE_THRESHOLDS` (`:131`), and `MODEL_EMBEDDING_DIMENSIONS` (`:140`) resolve for the new tables unchanged. This is an explicit, grounding-confirmed non-change.
- `api/app/llms/context.py` *(optional, narrow)* — lift `_post_rerank` into a new `api/app/llms/rerank_client.py` and import from there, so the diploma path reuses rerank without importing a private name. One-function move; low risk. Note `_post_rerank` (`context.py:48`) is provider-agnostic (POSTs to `{settings.GPU_API_URL}/rerank/`, 1 retry, 30s) and reusable as-is; the drop threshold is the **`settings.RERANKER_MIN_SCORE` attribute** (referenced in `context.py:321`, declared in `Settings` at `settings.py:21`), **not** a module constant in `context.py`.

### Generalize-vs-copy (decisive — there is **no test suite**, so minimize blast radius)

- **The chunk RAG layer is the template, not a thing to build.** `get_closest_chunks` (`data/documents.py:107`), `replace_document_with_chunks` (`data/documents.py:70`, the atomic re-ingest), `stream_fill_chunk_embeddings` (`embeddings.py:348`), `fetch_chunk_rows_for_fill` (`data/documents.py:215`), `ChunkSchema`/`DocumentSchema`/`IngestDocumentSchema` (`schemas/documents.py:51/11/70`, including the `distance: float | None` idiom and the `_strip_required` whitespace-rejecting validator), and `FillChunkEmbeddingsSchema` (`schemas/documents.py:107`) are all the right shapes to copy-adapt.
- **Backfill → COPY-ADAPT, do not generalize.** `stream_fill_embeddings`/`_process_question_batch` (`embeddings.py`) are welded to `question` in ~5 places. The **de-welded twin already exists** — `stream_fill_chunk_embeddings` (`embeddings.py:348`) with `_process_chunk_batch` (`:304`) and `_chunk_document_text` (`:297`, building `Наслов:/Содржина:`). Clone that twin for `diploma` and `professor_document` fills, reusing only the genuinely shared pieces: `generate_embeddings` (`embeddings.py:79`, bounded retry `EMBEDDING_MAX_ATTEMPTS=3`), `_prepare_text_for_embedding` (`text_utils.py:12`, the E5 `query:`/`passage:` prefix — call with `is_document=True` at fill, `is_document=False` at query), `embedding_to_pgvector` (`utils/database.py:1`), `MODEL_EMBEDDING_DIMENSIONS` (`models.py:140`, **not** in `database.py`), `EMBEDDING_BATCH_SIZE` (`embeddings.py:174`), and the dimension-mismatch guard pattern (`embeddings.py:200-208`). Preserve the SSE payload shape (`status/error/index/total/model/id/name/ts`) for the frontend.
- **KNN → COPY-ADAPT.** Copy `get_closest_chunks` (`data/documents.py:107`, the better template — it already JOINs to a parent and aliases columns) into `get_closest_diplomas` and `get_closest_professor_documents` — replicate the halfvec-cast branch + threshold defaulting (`MODEL_DISTANCE_THRESHOLDS.get(model, 0.5)`, not a flat `0.5`) + `# noqa: S608` verbatim; add `exclude_external_id` to the diploma version for leave-one-out and self-match suppression. Don't touch `get_closest_questions`/`get_closest_chunks`. Carry the file-level `# mypy: disable-error-code="arg-type"` (as `questions.py:1` / `documents.py` do).
- **Rerank → REUSE** `_post_rerank` directly (or the lifted `rerank_client.py`); copy a ~6-line `_merge_candidates` equivalent that dedupes on `external_id`. The diploma path reranks; the paper path reranks **only if** `ScoringWeights.rerank_papers` is on (default off — an aggregate over many papers benefits less from per-pair precision and skipping halves GPU load). Note the integration seam for the *chat-RAG* pipeline is `_search_both` (`context.py:88`) + `_build_candidates`; the recommender does **not** touch that seam — it has its own retrieval orchestrators.

---

## Recommender read-path flow

```
POST /recommendations { title, mentor?, mentor_topk=3 }
  ├─ mode = MEMBERS_ONLY if mentor is not None else FULL     (inferred from payload)
  ├─ query_text = title.strip()                              (the bare title — no description on input)
  ├─ embed query_text once (generate_embeddings, E5 query: prefix) → reuse for BOTH signals
  ├─ retrieve_similar_diplomas(db, embedded_query, model, initial_k, top_k):
  │     get_closest_diplomas (status='Одбрана') → _post_rerank({query, documents})
  │     keep score ≥ settings.RERANKER_MIN_SCORE, take top_k; vector-order fallback on failure
  │     → list[RetrievedDiploma] (row fields + similarity + rerank_score)
  ├─ retrieve_expert_professors(db, embedded_query, model, paper_initial_k):
  │     get_closest_professor_documents → (optional rerank) → aggregate per professor
  │     → ExpertiseIndex { by_professor: {name -> score}, supporting: {name -> [titles]} }
  ├─ ranked = score_people(retrieved, expertise, weights, mode, given_mentor=mentor)   ← PURE
  └─ rec = select_committee(ranked, mode, given_mentor=mentor,
                            mentor_topk=mentor_topk or weights.mentor_topk, exclude=())  ← PURE
```

Everything past the embed is a pure function of `(retrieved, expertise, weights, mode, given_mentor)` — no I/O — so the backtest and both endpoints call the identical functions, and the PURE/db/http separation holds.

### Three signals, blended per person

1. **Defense co-occurrence** (original): over the retrieved similar defenses, accumulate per-diploma weight `w_i = f(similarity/rerank, recency(date), defended)` onto that defense's mentor/members; `mentor_score += w_i`, `member_score += w_i` on each member, `pair_score += w_i` on `frozenset({member1,member2})`. Strong for established professors; **zero for someone who never defended near the topic**.
2. **Paper expertise** (cold-start fix): over the retrieved similar *papers*, accumulate per-paper weight onto that paper's `professor`. Nonzero for anyone with topically-close papers — **including junior professors with zero past defenses**.
3. **Co-authorship (buddy)**: over the **same topic-relevant retrieved paper subset**, accumulate an undirected co-author *edge* among canonical professors — for each retrieved paper, for every unordered pair in its canonical author set, add `s × r` where `s = 1 - distance` (the same topic-similarity metric expertise uses) and `r` is a recency decay on the paper's `year`. The edge is therefore **topic-conditioned by construction** (only papers that survived KNN against *this* proposal contribute, scaled by *this* proposal's similarity — the same two professors get a different edge for a different thesis) and **recency-conditioned** (old shared papers decay). This is a **second pair graph, distinct from the defense `pair_score`**: defense `pair_score` = *served together* (committee co-occurrence); the buddy edge = *published together* (papers). They are summed with separate weights in the pair objective; neither is derived from the other; either ablates to zero independently. The buddy edge can be nonzero where `pair_score` is zero — co-published but never co-served — which is the cold-start gap defense co-occurrence structurally can't fill.

### `retrieve_expert_professors` aggregation (returns BOTH indices in one pass)

- Embed `query_text` (the bare title) once and **pass it through** from the defense path (don't re-embed). Note the proposal text is embedded via `_prepare_text_for_embedding(text, model, is_document=False)` — the E5 `query:` prefix is **model-specific** (only `MULTILINGUAL_E5_LARGE`; the default `BGE_M3_LOCAL` gets no prefix), so the buddy path inherits whatever the configured model does, no special-casing.
- `get_closest_professor_documents(db, embedded_query, model, limit=paper_initial_k)` (now also `SELECT`s `coauthors, year` — a column-list-only change, no new SQL shape, `# noqa: S608` unchanged) → from the **one** retrieved paper set build **both** the expertise aggregate and the co-author edges, no extra DB round-trip:
  - **Expertise** (unchanged): for each retrieved paper convert distance→similarity `s = 1 - distance`, then per professor take a **decayed top-N sum**: sum the professor's top `expertise_top_papers` similarities, optionally `recency`-weighted by `year`. Top-N sum (not mean) so a professor with many close papers outranks one with a single lucky hit, but capped so a prolific author doesn't dominate on volume alone.
  - **Buddy edges** (new): `_accumulate_coauthor_edges` over the same papers, weighting each unordered canonical-author pair by `s × r` (topic-similarity × recency decay).

```python
@dataclass(frozen=True)
class ExpertiseIndex:
    by_professor: dict[str, float]    # canonical name -> raw expertise score
    supporting: dict[str, list[str]]  # name -> paper titles (evidence)


@dataclass(frozen=True)
class CoauthorIndex:
    # undirected co-authorship edges among CANONICAL professors, topic+recency weighted.
    # key = frozenset({canonical_a, canonical_b})  (mirrors the defense pair_score identity)
    edges: dict[frozenset[str], float]
    supporting: dict[frozenset[str], list[str]]  # pair -> shared paper titles (evidence)
```

**`_half_life_decay` — a NEW PURE helper (no recency helper exists in the repo today).** The plan's two existing recency fields (`recency_half_life_days`, `expertise_recency_half_life_days`) both default to `0.0` (= OFF), so even the defense/expertise recency kernel is still unbuilt; this introduces the single kernel they will all share. One kernel serves both date and year deltas — the year-delta callers multiply `(now_year - p.year)` by 365 before calling it:

```python
def _half_life_decay(delta_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:          # recency OFF -> no-op weight
        return 1.0
    return 0.5 ** (delta_days / half_life_days)
```

**`_accumulate_coauthor_edges` — PURE, in `recommend.py` next to `_defense_counts`:**

```python
def _accumulate_coauthor_edges(papers, weights, *, now_year, canonical_set) -> CoauthorIndex:
    edges: dict[frozenset[str], float] = {}
    supporting: dict[frozenset[str], list[str]] = {}
    for p in papers:                                  # the retrieved (topic-relevant) subset
        s = 1.0 - p.distance                          # topic-similarity (same metric as expertise)
        r = _half_life_decay((now_year - p.year) * 365.0,
                             weights.coauthor_recency_half_life_days) \
                if (weights.coauthor_recency_half_life_days and p.year) else 1.0
        w = s * r
        canon = [c for c in p.coauthors if c in canonical_set]
        for a, b in itertools.combinations(sorted(set(canon)), 2):
            key = frozenset((a, b))
            edges[key] = edges.get(key, 0.0) + w
            supporting.setdefault(key, []).append(p.title)
    return CoauthorIndex(edges=edges, supporting=supporting)
```

`now_year` is **injected** (PURE — no `datetime.now()` in the core), defaulting to the backtest's reference year so leave-one-out is deterministic. `canonical_set` = the distinct professor set (`diploma.mentor ∪ member1 ∪ member2`), passed in.

**Revised `retrieve_expert_professors` (additive second return value, rename-free):**

```python
async def retrieve_expert_professors(db, embedded_query, model, paper_initial_k, weights, *, now_year, canonical_set
) -> tuple[ExpertiseIndex, CoauthorIndex]:
    papers = await get_closest_professor_documents(db, embedded_query, model, limit=paper_initial_k)  # SELECTs coauthors, year
    expertise = _aggregate_expertise(papers, weights)            # unchanged per-individual top-N decayed sum
    coauthors = _accumulate_coauthor_edges(papers, weights, now_year=now_year, canonical_set=canonical_set)
    return expertise, coauthors
```

### Normalization before blend

Defense, expertise, and buddy raw scores live on different scales (the first two are per-person, the buddy edges are per-pair). Before blending, **min-max normalize each signal across the candidate set to [0,1]** inside `score_people` (pure, deterministic) — `_minmax(defense)`, `_minmax(expertise.by_professor)`, and `_minmax(coauthors.edges)`. This makes `expertise_weight`/`coauthor_weight` interpretable and prevents one signal's raw magnitude from silently dominating.

### `ScoringWeights` (`recommenders/config.py`)

```python
@dataclass(frozen=True)
class ScoringWeights:
    # --- defense signal (original) ---
    similarity_weight: float = 1.0
    rerank_weight: float = 1.0
    recency_half_life_days: float = 0.0       # 0 = off
    pair_affinity_weight: float = 0.5         # two strong individuals vs a pair that has actually served together
    mentor_topk: int = 3
    # --- expertise signal (papers) ---
    expertise_weight: float = 0.0             # 0 reproduces the defenses-only baseline EXACTLY (ablation pivot)
    expertise_top_papers: int = 5             # top-N similarities summed per professor
    expertise_recency_half_life_days: float = 0.0
    rerank_papers: bool = False
    # --- co-author / buddy signal (papers) ---
    coauthor_weight: float = 0.0                 # 0 reproduces the defenses+expertise result EXACTLY (ablation pivot)
    coauthor_recency_half_life_days: float = 0.0 # 0 = recency off; in days for kernel reuse, applied to year deltas (×365)
    coauthor_member_boost: float = 0.0           # MEMBERS-ONLY: boost candidates who are topic-recent buddies of the GIVEN mentor
    # --- cold-start lever ---
    cold_start_defense_floor: int = 2         # professors with < this many retrieved defenses ...
    cold_start_expertise_boost: float = 1.5   # ... get expertise_weight * this
    # --- members-only mode ---
    mentor_match_boost: float = 0.0           # up-weight retrieved defenses where the given mentor served
```

Passed as a dataclass arg — **no globals, no FastAPI, no db** — so the backtest tunes weights without code edits and the core stays liftable. `expertise_weight=0.0`, `mentor_match_boost=0.0`, `coauthor_weight=0.0`, and `coauthor_member_boost=0.0` defaults mean the new corpus, the members-only signal, and the buddy signal are **byte-identical no-ops** until the backtest turns them on.

- **Buddy default-off ⇒ the user-requested "change over time / depend on topic" properties are realized only when the sweep enables them.** With `coauthor_weight=0.0` the buddy edge contributes nothing, and with `coauthor_recency_half_life_days=0.0` it is also not time-weighted (`r=1.0`). The defaults are the **ablation baseline**; the sweep is what activates the requested behavior. The headline buddy run should therefore sweep `coauthor_weight>0` **and** a non-zero recency candidate — start from a **multi-year half-life** (e.g. `coauthor_recency_half_life_days≈1825`, ~5 years) so the time-varying property is actually exercised rather than left implicitly off, then tune.

### `score_people` — blend + members-only hint (PURE)

```python
def score_people(retrieved, expertise, weights, mode, given_mentor=None,
                 coauthors=None) -> RankedPeople:   # coauthors: CoauthorIndex | None (None => signal absent)
    defense = _accumulate_defense_scores(retrieved, weights, mentor_hint=given_mentor)  # {name: raw}
    n_def   = _defense_counts(retrieved)                                                # {name: # retrieved defenses}

    d_norm = _minmax(defense)
    e_norm = _minmax(expertise.by_professor)
    c_norm = _minmax(coauthors.edges) if coauthors is not None else {}                  # normalized buddy EDGES (pairs)

    blended = {}
    for name in set(d_norm) | set(e_norm):
        w_exp = weights.expertise_weight
        if n_def.get(name, 0) < weights.cold_start_defense_floor:
            w_exp *= weights.cold_start_expertise_boost           # junior/cold-start boost
        blended[name] = d_norm.get(name, 0.0) + w_exp * e_norm.get(name, 0.0)

    # MEMBERS-ONLY buddy boost — runs AFTER the blend loop and BEFORE the pop below, so the
    # `n != given_mentor` guard is meaningful (given_mentor still in `blended` here).
    if mode is Mode.MEMBERS_ONLY and given_mentor is not None and coauthors is not None:
        b_raw  = {n: coauthors.edges.get(frozenset((given_mentor, n)), 0.0)
                  for n in blended if n != given_mentor}
        b_norm = _minmax(b_raw)                                   # same normalization discipline
        for n, s in b_norm.items():
            blended[n] += weights.coauthor_member_boost * s       # additive; 0.0 => no-op

    if mode is Mode.MEMBERS_ONLY and given_mentor is not None:
        blended.pop(given_mentor, None)                            # given mentor excluded from candidates
    return RankedPeople(blended=blended, defense=d_norm, expertise=e_norm, coauthor=c_norm)
```

- **The members-only signal** is additive and optional: inside `_accumulate_defense_scores`, when `mentor_hint is not None and diploma.mentor == mentor_hint`, multiply that diploma's `w_i *= (1 + weights.mentor_match_boost)`. So defenses chaired by *that* mentor become stronger evidence of who the right co-members are — even for **junior mentors with few similar-topic defenses**, whose own defenses (any topic) get boosted into the member/pair tally. With `mentor_match_boost=0.0` this is a no-op and FULL mode is unaffected.
- **The buddy signal feeds two consumption points (blended, never replacing expertise):** (1) **MEMBERS-ONLY** — the most direct use: the mentor is given, so the strongest member candidates are *that mentor's* topic-relevant recent co-authors. The boost block above keys on edges incident to the given mentor and adds `coauthor_member_boost × min-max(edge)` per candidate; it surfaces pairs the **defense graph cannot** — a junior mentor who co-published with someone but never co-served gets a nonzero buddy edge where `pair_score` is zero. (2) **FULL** — a second pair graph in `_best_unordered_pair` (see below). `RankedPeople` gains a `coauthor: dict[frozenset[str], float]` field (the `_minmax`-normalized edges), so `select_committee`/`_best_unordered_pair` read it without re-normalizing and the response can expose buddy evidence — mirroring how `defense`/`expertise` are carried.
- **How it surfaces zero-defense professors:** if `name` never appears in any retrieved defense, `d_norm[name]=0` but `e_norm[name]>0` whenever their papers are topically close — so they enter ranking purely on expertise, and the cold-start boost lifts them relative to established professors coasting on co-occurrence. Set `expertise_weight=0` and they vanish (proving the signal is what surfaces them).
- **`pair_affinity` is defense-only; co-authorship is a SECOND pair graph.** Two professors can't "co-publish-on-a-committee", so *committee* pair co-occurrence comes only from defenses, and expertise feeds individual member scores. But professors **do** co-publish — so the buddy edge is a second, distinct pair relationship, **summed (never merged)** into `_best_unordered_pair` alongside the defense-derived `pair_score`, each with its own weight (`pair_affinity_weight` for served-together, `coauthor_weight` for published-together). Neither is derived from the other; each ablates to zero independently. `pair_affinity` itself operates on the blended per-person scores plus the defense `pair_score`, unchanged.

### `select_committee` — FULL vs MEMBERS-ONLY branch (PURE)

```python
def select_committee(ranked, mode, given_mentor, *, mentor_topk, exclude=()):
    if mode is Mode.MEMBERS_ONLY:
        mentor = given_mentor                                       # fixed, GIVEN — never scored/chosen
        candidates = {n: s for n, s in ranked.blended.items() if n != given_mentor}
    else:                                                           # FULL — original path verbatim
        mentor = _argmax over top mentor_topk by mentor_score
        candidates = {n: s for n, s in ranked.blended.items() if n != mentor}
    members = _best_unordered_pair(candidates, ranked, exclude=(*exclude, mentor))  # 2 distinct, frozenset identity
    return Recommendation(mode=mode, mentor=mentor, members=members,
                          mentor_is_given=(mode is Mode.MEMBERS_ONLY), supporting_ids=...)
```

- **FULL** is the original code path verbatim — `argmax` over `mentor_score` (within `mentor_topk` breadth), then exclude the chosen mentor from the pair. No behavior change.
- **MEMBERS-ONLY** never reads `mentor_score`; it sets `mentor = given_mentor` and excludes it from candidates. The member-pair selection (unordered `frozenset`, two distinct, individual-vs-pair blend) is **the same code** — only the exclusion set and the skipped argmax differ.
- **`_best_unordered_pair` now reads `ranked.coauthor` (FULL co-author term).** The pair objective gains a second, independently-weighted pair term over the buddy edges:

  ```
  pair_objective(a, b) = blended[a] + blended[b]
                       + pair_affinity_weight * pair_score[frozenset({a,b})]      # served together (defenses)
                       + coauthor_weight       * ranked.coauthor[frozenset({a,b})]  # published together (papers)
  ```

  where `ranked.coauthor` is the already-`_minmax`-normalized edge map (no re-normalization). With `coauthor_weight=0` the term vanishes and FULL behavior is unchanged. The two pair terms are summed, never merged — distinct relationships, each ablatable.
- The core returns one `Recommendation` for both modes; `mentor_is_given` lets the HTTP layer pick the response variant without the core knowing about HTTP.

### Wiring (`api/app/api/recommendations.py` — FastAPI seam, not pure)

```python
mode = Mode.MEMBERS_ONLY if payload.mentor else Mode.FULL
text = payload.title.strip()  # the bare title; no description on input
# _prepare_text_for_embedding applies the E5 query: prefix ONLY for MULTILINGUAL_E5_LARGE;
# the default BGE_M3_LOCAL gets no prefix — model-specific, not universal.
embedded = await generate_embeddings(_prepare_text_for_embedding(text, model, is_document=False), model)
retrieved  = await retrieve_similar_diplomas(db, embedded, model, initial_k, top_k)
expertise, coauthors = await retrieve_expert_professors(   # now returns BOTH indices
    db, embedded, model, paper_initial_k, weights, now_year=now_year, canonical_set=canonical_set)
ranked = score_people(retrieved, expertise, weights, mode, given_mentor=payload.mentor, coauthors=coauthors)
rec = select_committee(ranked, mode, given_mentor=payload.mentor,
                       mentor_topk=payload.mentor_topk or weights.mentor_topk)
```

`now_year` is derived at the HTTP seam (e.g. current year) and `canonical_set` is the distinct professor set from the `diploma` table; both are passed *into* the PURE core, never read inside it. The buddy signal flows through with no new endpoint.

---

## Request / response schemas

**Decision: one endpoint (`POST /recommendations/`), mode inferred from the optional `mentor` field — not two endpoints.** Both modes share the exact same pipeline and ~95% of the code; the only difference is whether `select_committee` receives a fixed mentor. Two routes would duplicate the embed/retrieve/error-handling glue and the OpenAPI surface for no behavioral gain, and would force the future native chat tool to pick a route instead of forwarding an optional arg. This keeps **one shared core, one entrypoint** — consistent with the rest of the codebase, where mode-like flags are payload fields (cf. `FillChunkEmbeddingsSchema.all_chunks`/`all_models`), not separate routes.

```python
# api/app/schemas/diplomas.py
class RecommendationRequestSchema(BaseModel):
    title: str = Field(min_length=1, examples=["Систем за препорака на ментори базиран на вградувања"],
                       description="Thesis title (Macedonian/Cyrillic) — the ONLY text input. Required. "
                                   "Usually short, occasionally a longer line. No description field exists.")
    mentor: str | None = Field(default=None, examples=["Соња Гиевска"], description=(
        "Optional canonical mentor name. Provided -> MEMBERS-ONLY: recommend only the two members; "
        "the mentor is fixed, excluded from candidates, and used as a signal. Omitted -> FULL: "
        "recommend mentor + two members."))
    mentor_topk: int = Field(default=3, ge=1, le=10,
                       description="Mentor candidate breadth (FULL mode only).")

    @field_validator("title", "mentor")          # mirrors IngestDocumentSchema._strip_required
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        if not stripped:
            raise ValueError("must not be empty or whitespace")   # whitespace-only mentor -> 422, NOT silent FULL
        return stripped

class PersonScoreSchema(BaseModel):
    name: str
    score: float | None                 # blended total (None/echo for a given mentor)
    defense_score: float                # co-occurrence component (explainability/ablation)
    expertise_score: float              # paper component
    buddy_score: float = 0.0            # co-authorship component (explainability/ablation parity; 0.0 when signal off)
    supporting_diploma_ids: list[UUID]  # evidence

class RecommendationResponseSchema(BaseModel):
    mode: Literal["full", "members_only"]
    mentor: PersonScoreSchema           # FULL: recommended (with score+evidence); MEMBERS-ONLY: echoed given mentor
    mentor_is_given: bool               # False in FULL, True in MEMBERS-ONLY
    members: list[PersonScoreSchema]    # the unordered pair (2 entries); ordered by score desc for stable JSON only
    supporting_diploma_ids: list[UUID]  # top-level evidence trail
```

- **FULL** → `mode="full"`, `mentor_is_given=False`, `mentor` = the argmax recommendation with its components and evidence; `members` = the recommended pair.
- **MEMBERS-ONLY** → `mode="members_only"`, `mentor_is_given=True`, `mentor` = the caller's name **echoed back** (its `score` may be `None`/echo since it wasn't chosen); `members` = the recommended pair, excluding that mentor.
- `members` is serialized as an ordered list for stable JSON, but its identity is the **unordered** `frozenset` from the core. Exposing `defense_score`/`expertise_score`/`buddy_score` separately is required for the backtest ablation and explainability (`buddy_score` is a per-person contribution surfaced only in MEMBERS-ONLY, where the buddy edge maps to a single candidate term; in FULL the buddy effect lives in the *pair* objective, so it shows up in the pair's selection, not as a per-person component).
- API-layer search defaults differ from data-layer defaults by design (cf. `ClosestQuestionsSchema` `limit=20`/`threshold=0.5` vs the data layer's `limit=8`/threshold-from-map) — pick `initial_k`/`top_k`/`paper_initial_k` deliberately in `recommendations.py`, don't conflate the two layers.

---

## Ingestion

### Diplomas — admin endpoint (`api/app/api/diplomas.py`)

Behind `api_key_dep = Depends(verify_api_key)` (router carries `db_dep`; auth is per-route on mutating routes, matching every existing router), reusing the data layer so a future script could call the same functions:

- `POST /diplomas/sync` — fetch `settings.DIPLOMAS_API_URL` (httpx via the shared client; clean JSON, no scraping) → filter `status='Одбрана'` (configurable) → parse `DD.MM.YYYY` → `upsert_diploma` keyed on `external_id` (`INSERT … ON CONFLICT (external_id) DO UPDATE`). Returns counts. Handle `httpx.RequestError`/`HTTPStatusError` explicitly (shared client has no retry).
- `POST /diplomas/fill-embeddings` — SSE backfill over `WHERE embedding_<col> IS NULL`, mirroring `stream_fill_chunk_embeddings`'s `response_class=StreamingResponse` + SSE-progress shape. (The existing analogue route is `/documents/fill`, not `/fill-embeddings` — pick whichever, but keep the new diploma/professor routes named consistently with each other.)

### Professor papers — `scripts/` scraper + admin fill

**`scripts/scrape_professor_papers.py`** — a standalone offline entrypoint, the paper analogue of `/diplomas/sync`. It builds its own `Database(dsn=Settings().DATABASE_URL)` exactly like `app/migrations.py` (compose uses the asyncpg-native scheme while the settings default is SQLAlchemy-style — don't hardcode a scheme), and reuses the data-layer `upsert_professor_document`.

Flow: **fetch (two co-primaries) → normalize → dedup → map → upsert → (separately) embed**

```
# --- source 1: OpenAlex (per mapped author) ---
for each author_id in professor_name_map.json:
  canonical = map[author_id]
  works = openalex.works(filter=author.id|orcid, per_page=200, cursor=*)   # polite pool: mailto=
    fetch:     httpx GET, cursor pagination
    normalize: reconstruct abstract_inverted_index -> plaintext; coalesce title/year/doi;
               coauthors = [ map.get(a.author.id, a.author.display_name)        # canonical if mapped, else raw Latin
                             for a in work.authorships ]                          # FULL list, incl. non-FINKI
               topics    = [ t.subfield.display_name for t in (work.topics or []) ]  # NOT concepts[]
# --- source 2: UKIM-repo OAI-PMH (FCSE collections) ---
records = oai.ListRecords(set=com_20.500.12188_5 / col_..._109 / col_..._108)  # https://repository.ukim.mk/server/oai/request
    normalize: <dc:description> is ALREADY plaintext; <dc:creator>* -> coauthors (Latin "Last, First", mapped where possible);
               <dc:subject>* -> topics; <dc:date> -> year; doi/handle -> dedup keys
# --- merge ---
dedup:     union both sources, collapse on DOI -> normalized title (Open Decision 17 picks the winning metadata)
enrich:    abstract still missing -> Semantic Scholar paper lookup (backoff)
map:       professor = canonical            (the row OWNER; unmapped owners never reach here)
upsert:    upsert_professor_document(external_id=sha256(source|source_paper_id),
                                     professor, source, coauthors, topics, ...)   # source in {openalex,ukim_repo,semantic_scholar}
then (separate command, mirrors /diplomas/fill-embeddings):
  POST /professor-documents/fill-embeddings    # SSE backfill over title+abstract only (coauthors/topics NOT embedded)
```

- **Idempotency unchanged:** `external_id = sha256(source|source_paper_id)` is still per (paper, mapped owner). A multi-FINKI-author paper still produces N rows (one per mapped owner), but each now carries the identical full `coauthors`/`topics` arrays — so the buddy intersection is recoverable from *any* retrieved copy, and edge double-counting is de-duped in Python (`frozenset` keys + the retrieved-subset bound), not in SQL. Re-runs UPSERT the arrays in place (`… ON CONFLICT (external_id) DO UPDATE SET … coauthors = EXCLUDED.coauthors, topics = EXCLUDED.topics`); no schema migration on re-ingest.
- **No new endpoint, no new fill stream:** embeddings still backfill via `POST /professor-documents/fill-embeddings` over title+abstract only — `coauthors`/`topics` are not embedded, they are scored in Python. Write/read separation intact.
- **Coverage bound (honesty caveat):** coverage is a **best-effort OpenAlex ∪ UKIM-repo union (deduped), bounded — not exhaustive** (Google Scholar would be the only near-exhaustive source and is unscrapeable). Within that, papers reach only *mapped* professors; the unresolved name-map tail (Open Decision 10) contributes no expertise and no buddy edges until hand-added — the cold-start cohort the feature targets.
- **UKIM-repo is a co-primary *paper* source but a secondary *edge* source.** UKIM-repo papers enter the union (title/abstract/year → expertise) as first-class rows, but for the buddy **edge graph** OpenAlex's stable `author.id` stays the disambiguated source. Mapping each UKIM `<dc:creator>` into `coauthors` for OpenAlex-missing local papers is **optional (Open Decision 16)** — map through the name map the same way, un-disambiguated free-text names stay raw — off by default given the no-ID disambiguation cost.

Rate-limit / retry handling:
- OpenAlex: polite pool (`mailto=`), self-throttle ≤10 req/s, cursor pagination, bounded exponential backoff on `httpx.HTTPStatusError` 429/5xx — reuse the backoff shape of `generate_embeddings` (`EMBEDDING_MAX_ATTEMPTS=3`).
- UKIM-repo OAI-PMH: harvest the FCSE collections (`col_20.500.12188_109`/`_108`) via `ListRecords` + `resumptionToken` pagination; gentle (it is a single institutional server); same bounded backoff on 5xx.
- Semantic Scholar: treat 429 as expected; per-paper backoff, **skip-and-log** rather than fail the run; require API key before bulk use.
- Idempotent by `external_id` UPSERT → safe to re-run / resume after a partial crash without a separate state store (the reason it stays a script, not a service).

The **scrape** has no HTTP surface; only embedding backfill is an endpoint (`POST /professor-documents/fill-embeddings` on the small `professor_documents` admin router) so it matches the diploma path. Embedding is a **separate read-of-write step** — write/read separation preserved.

Note (both ingestors and the scraper): endpoints/scripts that build `Database` themselves must read `Settings().DATABASE_URL` exactly like `app/migrations.py`. `Database` is in `api/app/data/connection.py` (the request-scoped `get_db` is in `api/app/data/db.py`); the lifespan creates the singleton at `app.state.db` and does **not** run migrations (those run only via `python -m app.migrations`).

---

## Backtest harness — the quality gate (`scripts/backtest.py`)

Pure read; the real proof the feature works, run **before** the endpoint is exposed in chat. It imports the pure functions — never reimplements them, or it measures the wrong thing.

- **Population:** defended diplomas with a non-null embedding + present mentor + both members.
- **Leave-one-out, single retrieval, multi-evaluation:** for each held-out diploma, run `get_closest_diplomas(..., exclude_external_id=held_out)` **once**, then evaluate every mode/ablation from that same retrieved set. Papers need no exclusion — the paper corpus is independent of the held-out defense (a leakage guard: a mentor having papers is realistic and fine; only the held-out *defense row* must be excluded). **The query mirrors production: embed the held-out diploma's TITLE ONLY** (E5 `query:`), *not* its stored `title + description` passage embedding — otherwise the backtest measures an input shape users never send.
- **Ablation built into `ScoringWeights` — three nested runs:** `--no-papers --no-buddies` (defenses-only) → `--no-buddies` (+expertise) → full (+buddies), each isolating one signal's delta. Because `expertise_weight=0` and `coauthor_weight=0` make `score_people` numerically identical to the original, the default run is byte-identical to GATE A until a knob is swept. `--no-buddies` sets `coauthor_weight=0` and `coauthor_member_boost=0`; it composes with `--no-papers`.
- **Buddy "change over time / depend on topic" is exercised only when both knobs are on.** The headline buddy run must set `coauthor_weight>0` **and** a non-zero `coauthor_recency_half_life_days` (sweep from ~5 years, `≈1825`); with recency off the edge is topic-conditioned but not time-weighted, so the user's time-varying property would be untested.
- **Modes:**
  - **FULL:** `score_people(retrieved, expertise, w, Mode.FULL)` → `select_committee(..., Mode.FULL, mentor_topk=...)`. Metrics: mentor **hit@1 / hit@3**, member-pair **Jaccard** vs `frozenset({member1, member2})`.
  - **MEMBERS-ONLY:** feed the held-out row's **true mentor** as given — `score_people(retrieved, expertise, w, Mode.MEMBERS_ONLY, given_mentor=true_mentor, coauthors=coauthors)` → `select_committee(..., given_mentor=true_mentor)`. Metric: **member-pair Jaccard** (mentor excluded from both predicted and true sets — the true set is already `{member1, member2}`, so no recomputation). **No mentor hit@k** — the mentor is given, not predicted. **This is the headline buddy result:** `coauthor_member_boost>0` should raise member-pair Jaccard by surfacing the given mentor's topic-recent buddies. The buddy signal is a *pair* signal, so member-pair Jaccard is where it must move.
- **Cold-start slice:** restrict to held-out diplomas whose true mentor had `< cold_start_defense_floor` *other* retrieved defenses — this is where the paper signals must move the needle most. The buddy edge can be nonzero where `pair_score` is zero (co-published but never co-served), so this slice is where buddies should beat defense-pair-affinity outright. If papers/buddies don't improve the cold-start slice, the feature isn't earning its complexity — **if buddies don't lift the cold-start MEMBERS-ONLY pair-Jaccard, keep `coauthor_weight=0`** (same ship-gate as expertise).
- **Leakage:** the paper corpus is independent of the held-out *defense* row, so no paper exclusion is needed for either expertise or buddies — a mentor having co-authored papers is realistic. The only guard remains excluding the held-out **defense** row from `get_closest_diplomas`; the buddy graph is built from papers, untouched by leave-one-out.
- **Reporting:** mean FULL mentor hit@1/@3, mean FULL member-pair Jaccard, mean MEMBERS-ONLY member-pair Jaccard — each against a naive baseline (globally-most-frequent mentor; most-frequent co-members of the given mentor for the MEMBERS-ONLY baseline). MEMBERS-ONLY pair-Jaccard is expected **higher** than FULL's (the mentor constraint removes ambiguity); if not, `mentor_match_boost`/`pair_affinity_weight` are mis-tuned. External bar (see **Prior art & reusable resources**): the GNN thesis reports mentor Hits@3 ≈ 0.97 / Hits@1 ≈ 0.66 — match-or-beat on mentor Hits@k (caveat: their ordered-C2/C3 + edge-masking isn't a like-for-like comparison).
- **Sweep + CLI:** `ScoringWeights` overridable via flags (incl. `expertise_weight`, `expertise_top_papers`, `cold_start_*`, `mentor_match_boost`, `--coauthor-weight`, `--coauthor-member-boost`, `--coauthor-recency-half-life-days`); `--sample-size` (stratified by mentor) for speed, `--full` for the headline; `--no-papers` (sets `expertise_weight=0`); `--no-buddies` (sets `coauthor_weight=0` and `coauthor_member_boost=0`, composable with `--no-papers`); `--mode {full,members,both}` (default `both`) to isolate `mentor_match_boost`/`coauthor_member_boost` against the members metric without recomputing FULL. The gate passes only if **both modes beat their baselines**, the paper signal ships only if it improves hit@k (especially cold-start) — otherwise keep `expertise_weight=0` — and the buddy signal ships only if it improves the cold-start / MEMBERS-ONLY pair-Jaccard, otherwise keep `coauthor_weight=0`. Report buddy-on vs buddy-off per mode, with the cold-start slice broken out.

---

## Temporal collaboration groups (staff groups by time)

**A required capability** (not optional): model groups of professors that **form / persist / dissolve over time** — "as time goes, groups of professors come and go." Distinct from the per-thesis buddy edge: that scores *pairs* for *one* proposal; this detects *communities* across the whole timeline. Both reuse corpora already built.

- **Two professor↔professor graphs, both timestamped, derived from existing corpora:**
  - **co-authorship edges** ← `professor_document` (`coauthors[]` + publication `year`)
  - **co-service edges** ← `diploma` (unordered pairs among `mentor`/`member1`/`member2` + `date_of_submission`)
- **Method:** window the edges (fixed or rolling year buckets) → community detection **per window** → track communities **across** windows by membership overlap (Jaccard) to capture birth / death / merge / split.
- **Tooling (grounded against this repo):** the `api` is deliberately dependency-light — **no numpy/scipy/sklearn/networkx**; Postgres has pgvector only (**no Apache AGE**, no native community detection); `gpu-api` has sklearn transitively via sentence-transformers. At **N≈100–126 professors** detection is trivial. **Implement as an OFFLINE `scripts/` job — pure-Python label-propagation / greedy-modularity (~40 lines, zero new deps)** is the best fit for the codebase's minimal-dependency ethos; `networkx` is an acceptable **single dep in the script ONLY** (never in the `api`). Edge extraction = plain SQL (`unnest` + self-join + `GROUP BY`); cross-window tracking = pure-Python set overlap.
- **Output:** writes the `professor_group` table; the `api` reads precomputed groups via a thin `api/app/data/professor_groups.py` reader → **NO graph libraries in the `api` request path.**
- **Sequencing:** **co-service** temporal groups are buildable **IMMEDIATELY** from `diploma` alone (clean Cyrillic identity, no name map). **co-authorship** temporal groups additionally need Datasets B (papers) + C (name map).
- **Optional recommender hook:** prefer committee members from the mentor's CURRENTLY-ACTIVE group (an enhancement behind a default-off knob; not required for v1).
- **Scope honesty:** only **emergent / inferred** groups — there is **no data source for official FINKI lab/research-group membership.**

---

## Phased sequence

Defense path ships and is backtested first (lower-risk half); papers layer on as an ablatable, default-off add.

0. **Schema** — append the `diploma`, `professor_document`, and `professor_group` blocks (one migration pass); confirm idempotent and leave `question`/`document`/`chunk` untouched.
1. **Data layer** — `recommenders/text.py`; `schemas/diplomas.py` (incl. optional `mentor` (no description), `_strip` validator, `mode`/`mentor_is_given` response) + `data/diplomas.py`; `schemas/professor_documents.py` + `data/professor_documents.py`.
2. **Ingestion (defenses)** — `api/diplomas.py` (`/sync` + `/fill-embeddings`); wire into `main.py`; run to populate (~3,548 rows).
   2a. **Co-service temporal groups (EARLY — no name map, no papers)** — `scripts/compute_professor_groups.py` over `diploma` only (`edge_basis='coservice'`) → `professor_group`; `data/professor_groups.py` reader. Buildable the moment defenses are ingested.
3. **Pure recommender** — `recommenders/config.py` (defense + expertise + cold-start + `mentor_match_boost` fields, expertise/boost defaulting to 0) + `recommenders/recommend.py` (`Mode`, `score_people`, `select_committee`, helpers). Defaults keep step 5 measuring the defenses-only baseline first.
4. **Retrieval orchestrators** — `llms/diploma_retrieval.py`; optional `_post_rerank` → `rerank_client.py` lift.
5. **Backtest (GATE A)** — `scripts/backtest.py`: defenses-only headline (FULL hit@k + both modes' pair-Jaccard); must beat the most-frequent-mentor baseline.
   --- paper corpus layers on here ---
   5a. **Name map** — build `api/resources/professor_name_map.json` (UKIM CRIS → ORCID → OpenAlex; human-verify tail; drop the unresolved).
   5b. **Scraper** — `scripts/scrape_professor_papers.py` UNIONing **OpenAlex + UKIM-repo** (dedup DOI → title) → populate `professor_document`, **also persisting `coauthors`/`topics`** (full authorships); `POST /professor-documents/fill-embeddings`. Re-run is idempotent (arrays UPSERT in place).
   5c. **Expertise retrieval** — `llms/professor_retrieval.py` (expertise aggregate); wire into `score_people`.
   5d. **Buddy index (new)** — `_accumulate_coauthor_edges` + the `_half_life_decay` kernel + the `CoauthorIndex` second return value + the `score_people`/`_best_unordered_pair`/MEMBERS-ONLY wiring, all behind `coauthor_weight=0`/`coauthor_member_boost=0`.
   5e. **Co-authorship temporal groups (after papers + name map)** — re-run `scripts/compute_professor_groups.py` with `edge_basis='coauthorship'` (and optional `'blend'`) over `professor_document`.
6. **Backtest (GATE B)** — three nested ablation runs (defenses-only → +expertise → +buddies), headline + cold-start slice + MEMBERS-ONLY. Ship the paper signal ONLY if it improves hit@k (esp. cold-start); else keep `expertise_weight=0`. Ship the buddy signal ONLY if it lifts the cold-start / MEMBERS-ONLY pair-Jaccard; else keep `coauthor_weight=0`.
7. **Endpoint** — `api/recommendations.py` (BOTH modes); wire into `main.py`.

The `compute_professor_groups.py` job is **offline (`scripts/`), never in the request path** — the `api` only reads `professor_group`.

This keeps blast radius minimal (papers additive, default-off, gated by their own ablation), preserves write/read separation (scrape script ≠ recommend path), honors canonical-Cyrillic-names-as-source-of-truth (map drops, never guesses), and reuses the proven `question`/`chunk` embed→KNN→rerank machinery rather than forking it.

---

## Open decisions (kept flexible by construction)

1. **Corpus embedding model** — default `BGE_M3_LOCAL` (`BAAI/bge-m3`, the local gpu-api model both `BGE_M3`/`BGE_M3_LOCAL` resolve to via the shared `embedding_bge_m3` column; multilingual); resolve in backtest (vs multilingual-e5-large / text-embedding-3-large for Cyrillic). Extra per-model columns are additive on both tables. The same model serves both corpora (shared column name).
2. **Scoring weights** — similarity vs rerank vs recency vs pair-affinity; tuned via backtest sweep.
3. **Expertise blend** — `expertise_weight`, `expertise_top_papers` (top-N sum cap), expertise recency, and whether to rerank papers (`rerank_papers`); all swept. `expertise_weight=0` is the defenses-only baseline.
4. **Cold-start lever** — `cold_start_defense_floor` / `cold_start_expertise_boost`; validated against the cold-start backtest slice.
5. **MEMBERS-ONLY mentor signal** — `mentor_match_boost` magnitude; swept against the MEMBERS-ONLY pair-Jaccard.
6. **Recency decay** on/off; **status filter** (defended-only vs include in-progress as low-weight signal).
7. **Member scoring** — individual vs pair-co-occurrence blend (`pair_affinity_weight`).
8. **`initial_k` / `top_k` / `paper_initial_k`** — start at the question/chunk defaults (30/10) for defenses; pick `paper_initial_k` deliberately (papers are an aggregate, so likely larger).
9. **Mentor exclusion / distinctness rules** — mentor (chosen or given) excluded from the member pair; two distinct members (confirm against any domain committee rules).
10. **Name-map resolution tail** — the threshold for "confidently resolved"; the size of the human-verified tail.
11. **Corpus text vs. query text** — diploma corpus rows embed `title + description` (richer passage); a query is **title-only** (the fixed input contract). The backtest may also try embedding the corpus title-only for query/corpus symmetry — but that only changes *how stored rows are embedded*, never the input shape.
12. **Co-author edge weight & recency** — `coauthor_weight`, `coauthor_recency_half_life_days`; swept (recency from ~5 years); `coauthor_weight=0` is the +expertise baseline.
13. **MEMBERS-ONLY buddy boost** — `coauthor_member_boost` magnitude; swept against the MEMBERS-ONLY pair-Jaccard (the buddy analogue of `mentor_match_boost`, decision #5).
14. **Topic-conditioning metric** — embedding-distance (chosen: reuse `s = 1 - distance`) vs a discrete `topics`-overlap weight; the stored `topics` array keeps the latter open as a topic-faceted backtest slice without re-scraping.
15. **Co-author storage shape** — array columns on `professor_document` (chosen) vs a normalized `paper`+`paper_author` join; revisit only if a *relational* (SQL-side) co-authorship query is ever needed at the read path (it isn't today — the buddy query is pure-Python over the retrieved subset).
16. **UKIM-repo co-author backfill** — whether to map `<dc:creator>` lists into `coauthors` for OpenAlex-missing local papers (free-text names, no IDs → disambiguation cost); off by default.
17. **OpenAlex × UKIM-repo union** — source weighting and the dedup key (DOI → normalized title); which source's metadata wins on a collision.
18. **UKIM-repo CRIS Person format — RESOLVED (probed this session): it does NOT carry Cyrillic or ORCID.** Person entities expose only Latin `crisrp.name`/`dc.title` (e.g. `"Mishkovski, Igor"`), `person.affiliation.name` (`"Faculty of Computer Science and Engineering"` — a clean FCSE filter), `person.email`/`dspace.object.owner` (**FINKI email**, e.g. `igor.mishkovski@finki.ukim.mk`), and sometimes `person.identifier.scopus-author-id` — never ORCID, never Cyrillic (~2,977 Person entities UKIM-wide). Consequence: the name map is built from APIs *without* a CRIS Cyrillic/ORCID bridge (see Professor name map); the **FINKI email is the bridge anchor** (`first.last@finki.ukim.mk` = a deterministic Latin transliteration of the Cyrillic name). Net: **no required HTML scraping** — `finki.ukim.mk` (Dataset D) drops to optional.
19. **Temporal-groups parameters** — window size (fixed vs rolling year buckets), community-detection algorithm (pure-Python label-propagation vs greedy-modularity vs `networkx` Louvain in the script only), and the cross-window matching Jaccard threshold.
20. **Group basis & recommender hook** — `edge_basis` `coservice` vs `coauthorship` vs `blend`; whether to enable the optional "prefer the mentor's currently-active group" member hint (default-off).

---

## Verification (no test suite — manual end-to-end)

Repo uses `uv` + Docker Compose (pgvector/pg17); `api/start.sh` migrates before gunicorn.

1. **Infra:** `docker compose up -d db gpu-api api` (db healthcheck gates api).
2. **Migrate:** `docker compose exec api python -m app.migrations`; verify `docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d diploma"`, `"\d professor_document"`, and `"\d professor_group"`.
3. **Ingest defenses:** `curl -X POST http://localhost:8880/diplomas/sync -H "x-api-key: $API_KEY"` then `curl -N -X POST http://localhost:8880/diplomas/fill-embeddings -H "x-api-key: $API_KEY"`; sanity: `SELECT count(*) FROM diploma WHERE embedding_bge_m3 IS NOT NULL;` → ~3,548.
4. **Scrape + ingest papers:** `cd api && uv run python ../scripts/scrape_professor_papers.py` (idempotent; re-runnable; unions OpenAlex + UKIM-repo, dedup DOI → title) then `curl -N -X POST http://localhost:8880/professor-documents/fill-embeddings -H "x-api-key: $API_KEY"`; sanity: `SELECT count(*), count(DISTINCT professor), count(DISTINCT source) FROM professor_document WHERE embedding_bge_m3 IS NOT NULL;` (expect ≥2 sources). UKIM-repo OAI harvest sanity (before/independent of the scraper): `ListRecords` for `col_20.500.12188_109`/`_108` and confirm `<dc:creator>`/`<dc:description>`/`<dc:subject>` are present per record.
4b. **Temporal-groups job:** `cd api && uv run python ../scripts/compute_professor_groups.py` — **co-service groups runnable before any paper data**; inspect `SELECT window, edge_basis, count(*), count(DISTINCT group_id) FROM professor_group GROUP BY window, edge_basis ORDER BY window;` (sane per-window counts; co-authorship rows appear only after step 4 + name map).
5. **Backtest (gate):** `cd api && uv run python ../scripts/backtest.py --sample-size 300` (GATE A defenses-only); then `--mode both` and (after papers) compare against `--no-papers` for the GATE B ablation incl. the cold-start slice; re-run with `--full` / weight flags for the headline.
6. **Endpoint — both modes (one endpoint):**

```bash
# FULL: title only -> mentor + 2 members  (the title is the ONLY text input — no description)
curl -s -X POST http://localhost:8880/recommendations/ -H 'content-type: application/json' \
  -d '{"title":"Систем за препорака базиран на вградувања"}' | python3 -m json.tool
  # expect mode="full", mentor_is_given=false, mentor + 2-person set

# MEMBERS-ONLY: title + mentor -> only the 2 members; mentor echoed, excluded
curl -s -X POST http://localhost:8880/recommendations/ -H 'content-type: application/json' \
  -d '{"title":"…","mentor":"Соња Гиевска"}' | python3 -m json.tool
  # expect mode="members_only", mentor_is_given=true, mentor echoed, 2 members != mentor

# Edge: whitespace-only mentor must 422 (not silently FULL)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8880/recommendations/ \
  -H 'content-type: application/json' -d '{"title":"…","mentor":"   "}'   # expect 422
```

Confirm the `Recommendations` tag at `/docs` shows the single endpoint with an optional `mentor` (no description field), and the `Professor Documents` admin tag shows the fill route.

7. **Static gates (only automated guardrails):** `cd api && uv run ruff format --check . && uv run ruff check . && uv run mypy app` (new dynamic-column SQL carries the sanctioned `# noqa: S608`; KNN clones carry the file-level `# mypy: disable-error-code="arg-type"`).

---

## Out of scope (documented follow-ups)

- **Chat-agent tool (native, in-process)** — after the backtest passes, expose the recommender as a native LangChain tool, *not* via the external MCP server (unrelated to this feature) and *not* by self-proxying the REST endpoint. `create_agent(llm, tools)` accepts any LangChain tools (each provider passes the raw `get_mcp_tools()` list straight through), so the change is small:
  - New `api/app/llms/agent_tools.py`: define `recommend_committee` as an async `@tool` with `title` (required) and an **optional `mentor`** Pydantic arg (no `description` — the input is title-only), whose body calls the **same shared core** the REST endpoint uses (embed the bare `title` → `retrieve_similar_diplomas` + `retrieve_expert_professors` → `score_people(..., mode, given_mentor=mentor, coauthors=coauthors)` → `select_committee(...)`) and returns a compact, name-first Macedonian string; plus `get_agent_tools()` returning `await get_mcp_tools() + [recommend_committee]` (MCP list may be empty — fine). No new routing — the single endpoint's two-mode shape is exactly what the tool needs, reinforcing "one shared core, two thin entrypoints." Because it calls the shared core, the tool **inherits the buddy signal automatically once `coauthor_weight>0`** — no extra tool change.
  - Swap the one line `tools = await get_mcp_tools()` → `tools = await get_agent_tools()` in the four providers (`anthropic.py:161`, `openai.py:160`, `google.py:226`, `ollama.py:192` — all four verified exact). Each path already logs `tool.name for tool in tools` and falls back to the non-agent `stream_*_response` on any exception, so a new tool's failures are caught by that same `try/except`.
  - **DB access (the one wrinkle — agent path has no `db`):** expose the startup-created `Database` (already a process-wide singleton at `app.state.db`, set in the `main.py:54` lifespan) via a module-level accessor — add `set_db()/get_db_pool()` in `api/app/data/db.py` (import the `Database` type from `api/app/data/connection.py`, where the class actually lives) and call `set_db(db)` in the lifespan. The tool reads `get_db_pool()`; the REST endpoint keeps `Depends(get_db)`. No threading `db` through the agent chain, no per-request state in a cached tool.
  - Note: `create_agent_token_generator` yields only `AIMessageChunk` text (drops tool-result frames), so the LLM paraphrases the tool's output — return a tight, quotable string and instruct the prompt to quote names exactly. (gpu-api Qwen models are non-agentic; they use the REST endpoint, not the tool.)
- **Per-row metadata on the new corpora** — `chunk.metadata JSONB` exists in SQL but is dead (never written/read; absent from `ChunkSchema`/`Chunk`). Neither new table adds a metadata column in v1; if richer paper provenance (venue, citation count, page) is wanted later, it requires new INSERT/SELECT/schema code — additive.
- **Reserved retrieval slots for papers in chat-RAG** — `_select_with_faq_reservation` reserves slots for `source=="faq"` only; the recommender does not flow through that pipeline, so this is moot here, but note that if professor docs were ever surfaced in the general chat RAG they would be just-another-displaceable source unless slots were deliberately reserved.