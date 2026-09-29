"""
Exports what the warehouse currently holds, one row per receipt, to a CSV, so that runs with
different OCR engines can be compared (notebooks/eval_system.ipynb).

    python -m data_platform.pipeline.snapshot results/system_ground_truth.csv
"""
import argparse
import json
import os

import pandas as pd

from data_platform.pipeline.db import connect

QUERY = """
SELECT d.doc_id, d.file_name, d.status, d.error, d.ocr_source,
       CASE WHEN r.doc_id IS NOT NULL THEN 'published'
            WHEN q.doc_id IS NOT NULL THEN 'quarantined'
            ELSE d.status END AS outcome,
       array_to_string(q.failed_checks, ',') AS failed_checks,
       coalesce(r.total, q.total) AS total, coalesce(r.subtotal, q.subtotal) AS subtotal,
       coalesce(r.cash, q.cash) AS cash, coalesce(r.change, q.change) AS change,
       coalesce(r.n_items, q.n_items) AS n_items, coalesce(r.items_total, q.items_total) AS items_total,
       r.payment_method,
       (SELECT count(*) FROM raw.ocr_tokens t WHERE t.doc_id = d.doc_id) AS n_tokens,
       (SELECT avg(confidence) FROM raw.ocr_tokens t WHERE t.doc_id = d.doc_id) AS mean_confidence,
       (SELECT count(*) FROM raw.ocr_tokens t WHERE t.doc_id = d.doc_id AND t.pii_type IS NOT NULL) AS n_pii,
       (SELECT json_agg(json_build_array(e.name, e.qty, e.amount) ORDER BY e.line_idx)
          FROM raw.extracted_lines e WHERE e.doc_id = d.doc_id AND e.kind = 'item') AS items
FROM ops.documents d
LEFT JOIN marts.fct_receipts r USING (doc_id)
LEFT JOIN marts.receipt_quarantine q USING (doc_id)
ORDER BY d.file_name
"""


def main():
    parser = argparse.ArgumentParser(description="Export per-receipt pipeline results to a CSV")
    parser.add_argument("out", help="CSV path, e.g. results/system_ground_truth.csv")
    parser.add_argument("--metadata", default="dataset_receipt/metadata.pkl")
    args = parser.parse_args()

    with connect() as conn:
        cur = conn.execute(QUERY)
        df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
    df["items"] = df["items"].map(lambda v: json.dumps(v) if v is not None else "[]")

    # the same train / val / test split as model training (seed 42), to report held-out receipts separately
    from ocr.dataset import split_receipts

    train, val, test = split_receipts(pd.read_pickle(args.metadata), seed=42)
    split = {**{f: "train" for f in train.file_name}, **{f: "val" for f in val.file_name},
             **{f: "test" for f in test.file_name}}
    df.insert(2, "split", df.file_name.map(split))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} receipts -> {args.out} | {df.outcome.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
