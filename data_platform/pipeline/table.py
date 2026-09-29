"""
The receipts as one DataFrame, without the database: OCR -> redact -> extract -> the dbt checks
(dbt/models/intermediate/int_receipt_checks.sql), ported to pandas. One row per receipt, with
`items` as a list of (amount, name) tuples.

    python -m data_platform.pipeline.table results/receipts_ground_truth
    python -m data_platform.pipeline.table results/receipts_model_test --ocr model --split test

writes <out>.pkl (the tuples kept as tuples) and <out>.csv.
"""
import argparse
import os

import pandas as pd

from data_platform.pipeline.extract import extract_lines
from data_platform.pipeline.ocr import GroundTruthOCR, ModelOCR
from data_platform.pipeline.redact import redact_tokens

TOLERANCE = 1  # rounding on per-item tax, as in dbt


def first(values):
    return values[0] if values else None


def receipt_row(tokens):
    """Items, key amounts, checks and outcome of one receipt, from its OCR tokens."""
    pii = redact_tokens(tokens)
    tokens = [{**t, "text": f"[{p}]" if p else t["text"]} for t, p in zip(tokens, pii)]
    lines = extract_lines(tokens)

    def amounts(kind):
        return [r["amount"] for r in lines if r["kind"] == kind and r["amount"] is not None]

    items = [(r["amount"], r["name"]) for r in lines if r["kind"] == "item" and r["amount"] is not None]
    items_total = sum(a for a, _ in items) if items else None
    discount_total = -sum(abs(a) for a in amounts("discount"))
    subtotal, totals = first(amounts("subtotal")), amounts("total")
    cash, change = first(amounts("cash")), first(amounts("change"))
    change = abs(change) if change is not None else None
    card = sum(amounts("card"))
    total = totals[-1] if totals else None

    near = lambda a, b: a is not None and b is not None and abs(a - b) <= TOLERANCE  # noqa: E731
    items_reconcile = bool(items) and (
        near(items_total, subtotal)
        or any(near(items_total, t) or near(items_total + discount_total, t) for t in totals)
    )
    change_consistent = (any(near(cash - t, change) for t in totals)
                         if cash and cash > 0 and change is not None else None)

    failed = []
    if not items:
        failed.append("no_items")
    elif not items_reconcile:
        failed.append("items_do_not_reconcile")
    if change_consistent is False:
        failed.append("change_inconsistent")

    if total is None:
        outcome = "extract_failed"  # the pipeline's extract step rejects a receipt with no TOTAL amount
    else:
        outcome = "quarantined" if failed else "published"
    confidences = [t["confidence"] for t in tokens if "confidence" in t]
    return {
        "items": items,
        "total": total,
        "n_items": len(items),
        "items_total": items_total,
        "subtotal": subtotal,
        "tax": sum(amounts("tax")) or None,
        "cash": cash,
        "change": change,
        "payment_method": ("mixed" if card > 0 and cash and cash > 0 else "card" if card > 0
                           else "cash" if cash and cash > 0 else "unknown"),
        "failed_checks": failed,
        "outcome": outcome,
        "n_tokens": len(tokens),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
    }


def build_table(engine, metadata, img_dir):
    rows = []
    for i, name in enumerate(metadata.file_name, 1):
        doc = {"file_name": name, "image_path": os.path.join(img_dir, name)}
        rows.append({"file_name": name, **receipt_row(engine(doc))})
        if engine.name == "model":
            print(f"  {i}/{len(metadata)} {name}", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Receipts -> one DataFrame of items, totals and checks")
    parser.add_argument("out", help="Output path without extension, e.g. results/receipts_ground_truth")
    parser.add_argument("--ocr", default="ground_truth", choices=["ground_truth", "model"])
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--img-dir", default="dataset_receipt/images")
    parser.add_argument("--metadata", default="dataset_receipt/metadata.pkl")
    parser.add_argument("--dbnet", default="checkpoints/dbnet_best.pt")
    parser.add_argument("--recognizer", default="checkpoints/trocr-printed/best")
    args = parser.parse_args()

    from ocr.dataset import split_receipts

    metadata = pd.read_pickle(args.metadata)
    train, val, test = split_receipts(metadata, seed=42)  # the split used to train the models
    split = {**dict.fromkeys(train.file_name, "train"), **dict.fromkeys(val.file_name, "val"),
             **dict.fromkeys(test.file_name, "test")}
    if args.split != "all":
        metadata = metadata[metadata.file_name.map(split) == args.split]

    engine = (ModelOCR(args.dbnet, args.recognizer) if args.ocr == "model"
              else GroundTruthOCR(args.metadata))
    table = build_table(engine, metadata, args.img_dir)
    table.insert(1, "split", table.file_name.map(split))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_pickle(f"{args.out}.pkl")
    table.to_csv(f"{args.out}.csv", index=False)
    print(f"{len(table)} receipts -> {args.out}.pkl/.csv | {table.outcome.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
