-- Warehouse layout
--   ops             what the pipeline did (runs, per-document status)
--   raw_restricted  unredacted OCR output, never exposed to analysts
--   raw             redacted OCR tokens + extracted fields, the input to dbt
--   staging, marts  built by dbt

CREATE SCHEMA ops;
CREATE SCHEMA raw_restricted;
CREATE SCHEMA raw;

CREATE TABLE ops.pipeline_runs (
    run_id       text PRIMARY KEY,
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    status       text NOT NULL DEFAULT 'running',  -- running | success | failed
    failed_step  text,
    error        text,
    params       jsonb,
    stats        jsonb
);

-- Each step moves a document forward: pending -> ocr_done -> redacted -> extracted.
-- A failure parks it in <step>_failed with the error, without blocking the others.
CREATE TABLE ops.documents (
    doc_id       serial PRIMARY KEY,
    file_name    text NOT NULL,
    file_hash    text NOT NULL UNIQUE,  -- re-ingesting the same image is a no-op
    image_path   text NOT NULL,
    status       text NOT NULL DEFAULT 'pending',
    error        text,
    ocr_source   text,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    last_run_id  text REFERENCES ops.pipeline_runs (run_id)
);

CREATE TABLE raw_restricted.ocr_tokens (
    doc_id      int  NOT NULL REFERENCES ops.documents (doc_id) ON DELETE CASCADE,
    token_idx   int  NOT NULL,
    text        text NOT NULL,
    polygon     jsonb NOT NULL,  -- [[x, y] x 4], image pixels
    confidence  real,
    PRIMARY KEY (doc_id, token_idx)
);

CREATE TABLE raw.ocr_tokens (
    doc_id      int  NOT NULL REFERENCES ops.documents (doc_id) ON DELETE CASCADE,
    token_idx   int  NOT NULL,
    text        text NOT NULL,   -- PII replaced by a [TYPE] placeholder
    pii_type    text,            -- null when the token was kept as is
    polygon     jsonb NOT NULL,
    confidence  real,
    PRIMARY KEY (doc_id, token_idx)
);

-- One row per text line. token_idxs links every value back to its boxes in the image.
CREATE TABLE raw.extracted_lines (
    doc_id             int  NOT NULL REFERENCES ops.documents (doc_id) ON DELETE CASCADE,
    line_idx           int  NOT NULL,
    kind               text NOT NULL,  -- item | subtotal | tax | service | discount | rounding
                                       -- | total | cash | card | change | count | note | unknown
    name               text,
    qty                int,
    unit_price         numeric,
    amount             numeric,
    line_text          text NOT NULL,
    token_idxs         int[] NOT NULL,
    extractor_version  text NOT NULL,
    PRIMARY KEY (doc_id, line_idx)
);
