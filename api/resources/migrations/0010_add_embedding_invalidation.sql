ALTER TABLE question
    ADD COLUMN IF NOT EXISTS embedding_revision BIGINT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS embedding_bge_m3_version TEXT,
    ADD COLUMN IF NOT EXISTS embedding_bge_m3_updated_at TIMESTAMP;

ALTER TABLE chunk
    ADD COLUMN IF NOT EXISTS embedding_revision BIGINT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS embedding_bge_m3_version TEXT,
    ADD COLUMN IF NOT EXISTS embedding_bge_m3_updated_at TIMESTAMP;

ALTER TABLE diploma
    ADD COLUMN IF NOT EXISTS embedding_revision BIGINT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS embedding_bge_m3_version TEXT,
    ADD COLUMN IF NOT EXISTS embedding_bge_m3_updated_at TIMESTAMP;

ALTER TABLE professor_document
    ADD COLUMN IF NOT EXISTS embedding_revision BIGINT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS embedding_bge_m3_version TEXT,
    ADD COLUMN IF NOT EXISTS embedding_bge_m3_updated_at TIMESTAMP;

CREATE OR REPLACE FUNCTION embedding_notify_dirty()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM pg_notify('embedding_dirty', TG_ARGV[0] || ':' || NEW.id::TEXT);
    ELSIF OLD.embedding_revision IS DISTINCT FROM NEW.embedding_revision THEN
        PERFORM pg_notify('embedding_dirty', TG_ARGV[0] || ':' || NEW.id::TEXT);
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION question_invalidate_embedding()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.name IS NOT DISTINCT FROM OLD.name
       AND NEW.content IS NOT DISTINCT FROM OLD.content THEN
        RETURN NEW;
    END IF;
    NEW.embedding_revision := OLD.embedding_revision + 1;
    NEW.embedding_bge_m3 := NULL;
    NEW.embedding_bge_m3_version := NULL;
    NEW.embedding_bge_m3_updated_at := NULL;
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION chunk_invalidate_embedding()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.section IS NOT DISTINCT FROM OLD.section
       AND NEW.content IS NOT DISTINCT FROM OLD.content THEN
        RETURN NEW;
    END IF;
    NEW.embedding_revision := OLD.embedding_revision + 1;
    NEW.embedding_bge_m3 := NULL;
    NEW.embedding_bge_m3_version := NULL;
    NEW.embedding_bge_m3_updated_at := NULL;
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION diploma_invalidate_embedding()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.title IS NOT DISTINCT FROM OLD.title
       AND NEW.description IS NOT DISTINCT FROM OLD.description THEN
        RETURN NEW;
    END IF;
    NEW.embedding_revision := OLD.embedding_revision + 1;
    NEW.embedding_bge_m3 := NULL;
    NEW.embedding_bge_m3_version := NULL;
    NEW.embedding_bge_m3_updated_at := NULL;
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION professor_document_invalidate_embedding()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.title IS NOT DISTINCT FROM OLD.title
       AND NEW.abstract IS NOT DISTINCT FROM OLD.abstract THEN
        RETURN NEW;
    END IF;
    NEW.embedding_revision := OLD.embedding_revision + 1;
    NEW.embedding_bge_m3 := NULL;
    NEW.embedding_bge_m3_version := NULL;
    NEW.embedding_bge_m3_updated_at := NULL;
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION document_title_touch()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.title IS DISTINCT FROM OLD.title THEN
        NEW.updated_at := NOW();
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION document_title_invalidate_chunks()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    child_id UUID;
BEGIN
    IF NEW.title IS NOT DISTINCT FROM OLD.title THEN
        RETURN NULL;
    END IF;
    FOR child_id IN
        UPDATE chunk
        SET embedding_revision = embedding_revision + 1,
            embedding_bge_m3 = NULL,
            embedding_bge_m3_version = NULL,
            embedding_bge_m3_updated_at = NULL,
            updated_at = NOW()
        WHERE document_id = NEW.id
        RETURNING id
    LOOP
        PERFORM pg_notify('embedding_dirty', 'chunk:' || child_id::TEXT);
    END LOOP;
    RETURN NULL;
END;
$$;

CREATE TRIGGER question_embedding_invalidate_before_update
BEFORE UPDATE OF name, content ON question
FOR EACH ROW EXECUTE FUNCTION question_invalidate_embedding();

CREATE TRIGGER question_embedding_dirty_after_insert
AFTER INSERT ON question
FOR EACH ROW EXECUTE FUNCTION embedding_notify_dirty('question');

CREATE TRIGGER question_embedding_dirty_after_update
AFTER UPDATE OF name, content ON question
FOR EACH ROW EXECUTE FUNCTION embedding_notify_dirty('question');

CREATE TRIGGER chunk_embedding_invalidate_before_update
BEFORE UPDATE OF section, content ON chunk
FOR EACH ROW EXECUTE FUNCTION chunk_invalidate_embedding();

CREATE TRIGGER chunk_embedding_dirty_after_insert
AFTER INSERT ON chunk
FOR EACH ROW EXECUTE FUNCTION embedding_notify_dirty('chunk');

CREATE TRIGGER chunk_embedding_dirty_after_update
AFTER UPDATE OF section, content ON chunk
FOR EACH ROW EXECUTE FUNCTION embedding_notify_dirty('chunk');

CREATE TRIGGER diploma_embedding_invalidate_before_update
BEFORE UPDATE OF title, description ON diploma
FOR EACH ROW EXECUTE FUNCTION diploma_invalidate_embedding();

CREATE TRIGGER diploma_embedding_dirty_after_insert
AFTER INSERT ON diploma
FOR EACH ROW EXECUTE FUNCTION embedding_notify_dirty('diploma');

CREATE TRIGGER diploma_embedding_dirty_after_update
AFTER UPDATE OF title, description ON diploma
FOR EACH ROW EXECUTE FUNCTION embedding_notify_dirty('diploma');

CREATE TRIGGER professor_document_embedding_invalidate_before_update
BEFORE UPDATE OF title, abstract ON professor_document
FOR EACH ROW EXECUTE FUNCTION professor_document_invalidate_embedding();

CREATE TRIGGER professor_document_embedding_dirty_after_insert
AFTER INSERT ON professor_document
FOR EACH ROW EXECUTE FUNCTION embedding_notify_dirty('professor_document');

CREATE TRIGGER professor_document_embedding_dirty_after_update
AFTER UPDATE OF title, abstract ON professor_document
FOR EACH ROW EXECUTE FUNCTION embedding_notify_dirty('professor_document');

CREATE TRIGGER document_title_touch_before_update
BEFORE UPDATE OF title ON document
FOR EACH ROW EXECUTE FUNCTION document_title_touch();

CREATE TRIGGER document_title_invalidate_chunks_after_update
AFTER UPDATE OF title ON document
FOR EACH ROW EXECUTE FUNCTION document_title_invalidate_chunks();
