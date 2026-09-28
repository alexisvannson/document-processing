-- One row per extracted receipt: its key amounts and the checks that decide whether it is
-- published (fct_receipts) or quarantined (receipt_quarantine).
--
-- Receipts often print several totals ("Net Total" before service and tax, then "TOTAL"),
-- so the items may match any of them; the amount paid is the last one.

{% set tolerance = 1 %}  -- rounding on per-item tax

with lines as (
    select * from {{ ref('stg_extracted_lines') }}
),

per_receipt as (
    select
        doc_id,
        count(*) filter (where kind = 'item') as n_items,
        sum(amount) filter (where kind = 'item') as items_total,
        -- discounts are printed signed or unsigned ("-Rp 34,363", "DISC 5,000")
        coalesce(-sum(abs(amount)) filter (where kind = 'discount'), 0) as discount_total,
        (array_agg(amount order by line_idx) filter (where kind = 'subtotal' and amount is not null))[1] as subtotal,
        array_agg(amount order by line_idx) filter (where kind = 'total' and amount is not null) as totals,
        sum(amount) filter (where kind = 'tax') as tax,
        sum(amount) filter (where kind = 'service') as service,
        (array_agg(amount order by line_idx) filter (where kind = 'cash' and amount is not null))[1] as cash,
        -- change is sometimes printed negative ("CHANGE -4,500")
        abs((array_agg(amount order by line_idx) filter (where kind = 'change' and amount is not null))[1]) as change,
        coalesce(sum(amount) filter (where kind = 'card'), 0) as card
    from lines
    group by doc_id
),

checks as (
    select
        *,
        totals[array_length(totals, 1)] as total,
        n_items > 0 as has_items,
        abs(items_total - subtotal) <= {{ tolerance }}
            or exists (
                select 1 from unnest(totals) as t
                where abs(items_total - t) <= {{ tolerance }}
                   or abs(items_total + discount_total - t) <= {{ tolerance }}
            ) as items_reconcile,
        -- only checkable when cash was handed over and change printed
        case
            when cash > 0 and change is not null then exists (
                select 1 from unnest(totals) as t where abs(cash - t - change) <= {{ tolerance }}
            )
        end as change_consistent
    from per_receipt
)

select
    c.doc_id,
    d.file_name,
    d.ingested_at,
    d.ocr_source,
    c.n_items,
    c.items_total,
    c.discount_total,
    c.subtotal,
    c.tax,
    c.service,
    c.total,
    c.cash,
    c.change,
    c.card,
    case
        when c.card > 0 and c.cash > 0 then 'mixed'
        when c.card > 0 then 'card'
        when c.cash > 0 then 'cash'
        else 'unknown'
    end as payment_method,
    c.has_items,
    coalesce(c.items_reconcile, false) as items_reconcile,
    c.change_consistent,
    array_remove(array[
        case when not c.has_items then 'no_items' end,
        case when c.has_items and not coalesce(c.items_reconcile, false) then 'items_do_not_reconcile' end,
        case when c.change_consistent = false then 'change_inconsistent' end
    ], null) as failed_checks
from checks as c
join {{ ref('stg_documents') }} as d using (doc_id)
