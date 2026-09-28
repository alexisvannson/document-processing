-- Lines of receipts that made it through extraction. A failed receipt has no lines left
-- (the pipeline deletes them), but the status filter keeps this true whatever the source holds.
select
    l.doc_id,
    l.line_idx,
    l.kind,
    nullif(trim(l.name), '') as name,
    l.qty,
    l.unit_price,
    l.amount,
    l.line_text,
    l.token_idxs,
    l.extractor_version
from {{ source('raw', 'extracted_lines') }} as l
join {{ source('ops', 'documents') }} as d using (doc_id)
where d.status = 'extracted'
