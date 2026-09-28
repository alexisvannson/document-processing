-- No receipt is lost or counted twice: each extracted one is in exactly one of the two marts.
with extracted as (
    select doc_id from {{ ref('stg_documents') }} where status = 'extracted'
),
landed as (
    select doc_id from {{ ref('fct_receipts') }}
    union all
    select doc_id from {{ ref('receipt_quarantine') }}
)
select e.doc_id, count(l.doc_id) as times_landed
from extracted as e
left join landed as l using (doc_id)
group by e.doc_id
having count(l.doc_id) <> 1
