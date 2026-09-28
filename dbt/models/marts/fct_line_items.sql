-- Items of published receipts only, so item-level numbers add up to fct_receipts.
select
    l.doc_id,
    l.line_idx,
    l.name,
    upper(regexp_replace(l.name, '\s+', ' ', 'g')) as name_normalized,
    l.qty,
    coalesce(l.unit_price, case when l.qty > 0 then l.amount / l.qty end) as unit_price,
    l.amount,
    l.token_idxs
from {{ ref('stg_extracted_lines') }} as l
join {{ ref('fct_receipts') }} as r using (doc_id)
where l.kind = 'item'
