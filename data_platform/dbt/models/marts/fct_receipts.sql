-- Receipts that passed every check: the trusted table for dashboards and the agent.
select
    doc_id,
    file_name,
    ingested_at,
    ocr_source,
    n_items,
    items_total,
    discount_total,
    subtotal,
    tax,
    service,
    total,
    cash,
    change,
    card,
    payment_method
from {{ ref('int_receipt_checks') }}
where cardinality(failed_checks) = 0
