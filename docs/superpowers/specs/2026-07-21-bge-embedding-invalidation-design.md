# BGE-M3 Embedding Invalidation Design

## Goal

Keep FAQ, document chunk, diploma, and professor-document BGE-M3 vectors correct when rows are inserted or edited through the API, ingestion scripts, pgAdmin, or direct SQL, without a polling queue or general-purpose task framework.

## Decisions

- BGE-M3 is the only required corpus embedding model.
- The source rows themselves are the durable queue: a dirty row has no current BGE vector or has an old embedding spec version.
- PostgreSQL triggers invalidate vectors and send a commit-bound `NOTIFY` whenever embedding inputs change.
- One dedicated worker listens for notifications and performs a full dirty scan only on startup or listener reconnect.
- Existing manual fill endpoints remain compatibility and recovery surfaces, but use the same revision-safe persistence rules.
- Chat-model selection is independent from the corpus retrieval model.

## Schema

Add the following fields to `question`, `chunk`, `diploma`, and `professor_document`:

```sql
embedding_revision BIGINT NOT NULL DEFAULT 1,
embedding_bge_m3_version TEXT,
embedding_bge_m3_updated_at TIMESTAMP
```

The existing `embedding_bge_m3` vector remains the stored data plane.

The application defines one stable BGE embedding spec version. It changes whenever the model checkpoint, canonical text builder, or preprocessing contract changes.

## Invalidation triggers

Before an embedding-relevant update, PostgreSQL:

1. Increments `embedding_revision`.
2. Clears `embedding_bge_m3`.
3. Clears `embedding_bge_m3_version` and `embedding_bge_m3_updated_at`.
4. Updates the row's `updated_at`.

Embedding inputs are:

- `question`: `name`, `content`.
- `chunk`: `section`, `content`.
- `diploma`: `title`, `description`.
- `professor_document`: `title`, `abstract`.
- `document.title`: invalidates every child chunk because chunk canonical text includes the document title.

After a dirty insert or update, PostgreSQL sends `pg_notify('embedding_dirty', '<resource>:<uuid>')`. Notifications wake the worker but do not carry durable work state.

## Worker

The worker is a separate Compose service using the API image and a dedicated command.

On startup or PostgreSQL reconnect it scans every corpus for rows where:

```text
embedding_bge_m3 IS NULL
OR embedding_bge_m3_version != current BGE spec version
```

After startup it blocks on `LISTEN embedding_dirty` and drains dirty rows whenever notified. It does not periodically poll.

The worker processes one BGE batch at a time. For each row it captures the row ID, canonical text, and `embedding_revision`, validates batch cardinality and 1024-dimensional vectors, then persists with a revision guard. Vector, spec version, and embedding timestamp are written together. If the source revision changed during inference, the update affects zero rows and the newer dirty notification wins.

Transient generation failures retain the dirty row. The worker uses the existing bounded provider retries; an authenticated manual fill or worker restart provides recovery after a persistent failure.

## Scripts with supplied embeddings

Scripts use shared revision-safe persistence functions. Within one transaction they:

1. Insert or update source fields, triggering invalidation and revision advancement.
2. Read the resulting revision.
3. Validate a supplied BGE vector's dimensions.
4. Write the vector, current spec version, and timestamp with the same revision guard.
5. Commit.

The row is never externally visible with new source text and an old vector.

## Retrieval

All BGE retrieval SQL requires both:

```sql
embedding_bge_m3 IS NOT NULL
AND embedding_bge_m3_version = <current spec version>
```

Dirty rows are excluded immediately after the source transaction commits.

## Administration

Authenticated administration provides:

- health counts by corpus: ready and dirty;
- fill-dirty: notify the worker and return current dirty counts;
- rebuild: clear BGE version/vector metadata for all corpora and notify the worker.

Existing resource-specific fill endpoints continue to work during migration and are changed to revision-safe writes.

## Migration

Existing vectors have no trustworthy spec version. The new fields therefore begin with a NULL version, which classifies every existing row as dirty. The worker's first startup scan rebuilds BGE-M3 coverage. Retrieval switches to the version predicate only with the worker deployed so the corpus can converge immediately.

## Failure guarantees

- A pgAdmin edit fires the database trigger and invalidates the old vector in the same transaction.
- A missed notification cannot lose work because dirty state is stored on the row and startup/reconnect scans recover it.
- A source change during inference prevents the stale vector write through `embedding_revision`.
- A malformed provider batch writes no vectors from that batch.
- Worker downtime leaves rows dirty and unsearchable, never falsely clean.

## Verification

- Migration tests inspect all columns, trigger functions, relevant-field behavior, child-chunk invalidation, and notification DDL.
- Unit tests prove dirty selection, canonical text, batch validation, and revision-guarded writes.
- API tests prove authentication and health/rebuild contracts.
- Integration QA uses real PostgreSQL to update FAQ content as pgAdmin would, observes vector invalidation and notification, runs the worker drain, and confirms retrieval only after the current spec is written.
- Full API tests, Ruff, mypy, Compose rendering, and post-implementation review must pass before the PR opens.
