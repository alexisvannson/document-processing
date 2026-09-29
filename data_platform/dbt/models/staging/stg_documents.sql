select
    doc_id,
    file_name,
    status,
    error,
    ocr_source,
    ingested_at,
    updated_at,
    last_run_id
from {{ source('ops', 'documents') }}
