-- Receipts that failed a check, with the reasons, for review (by a person or the review agent).
select
    doc_id,
    file_name,
    ingested_at,
    failed_checks,
    n_items,
    items_total,
    discount_total,
    subtotal,
    total,
    cash,
    change
from {{ ref('int_receipt_checks') }}
where cardinality(failed_checks) > 0
