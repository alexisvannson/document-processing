import argparse
import json
from datetime import datetime, timezone

from data_platform.pipeline.db import connect
from data_platform.pipeline.extract import run_extract
from data_platform.pipeline.ingest import ingest
from data_platform.pipeline.ocr import GroundTruthOCR, ModelOCR, run_ocr
from data_platform.pipeline.redact import run_redact
from data_platform.pipeline.runs import fail_run, finish_run, record_step, start_run

STEPS = ["ingest", "ocr", "redact", "extract"]


def parse_args():
    parser = argparse.ArgumentParser(description="Receipts pipeline: ingest -> OCR -> redact -> extract into Postgres")
    parser.add_argument("--steps", default=",".join(STEPS), help=f"Comma-separated subset of {STEPS}")
    parser.add_argument("--img-dir", default="dataset_receipt/images")
    parser.add_argument("--metadata", default="dataset_receipt/metadata.pkl")
    parser.add_argument("--ocr", default="ground_truth", choices=["ground_truth", "model"],
                        help="ground_truth: annotated words from metadata.pkl; model: DBNet + the recognizer")
    parser.add_argument("--dbnet", default="checkpoints/dbnet_best.pt")
    parser.add_argument("--recognizer", default="checkpoints/trocr-printed/best")
    parser.add_argument("--corrupt", type=float, default=0.0,
                        help="Fraction of numeric tokens to corrupt with a swapped digit (simulated OCR errors)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true",
                        help="Send every document back to `pending`, to reprocess them all")
    parser.add_argument("--run-id", default=None,
                        help="Record the steps under this run, owned by the caller (a scheduler starts, fails and"
                             " finishes it). Without it the CLI creates and closes its own run.")
    return parser.parse_args()


def main():
    args = parse_args()
    steps = args.steps.split(",")
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(f"Unknown steps: {sorted(unknown)}")

    owns_run = args.run_id is None
    run_id = args.run_id or datetime.now(timezone.utc).strftime("manual__%Y%m%dT%H%M%S")
    conn = connect()
    start_run(conn, run_id, vars(args))
    if args.reset:
        conn.execute("UPDATE ops.documents SET status = 'pending', error = NULL, updated_at = now()")

    step = None
    try:
        for step in [s for s in STEPS if s in steps]:
            if step == "ingest":
                stats = ingest(conn, args.img_dir)
            elif step == "ocr":
                engine = (ModelOCR(args.dbnet, args.recognizer) if args.ocr == "model"
                          else GroundTruthOCR(args.metadata, args.corrupt, args.seed))
                stats = run_ocr(conn, run_id, engine)
            elif step == "redact":
                stats = run_redact(conn, run_id)
            elif step == "extract":
                stats = run_extract(conn, run_id)
            record_step(conn, run_id, step, stats)
            print(f"{step:>8}: {stats}")
    except Exception as e:
        if owns_run:
            fail_run(conn, run_id, step, f"{type(e).__name__}: {e}")
        raise
    if owns_run:
        finish_run(conn, run_id)

    counts = conn.execute("SELECT status, count(*) FROM ops.documents GROUP BY status ORDER BY status").fetchall()
    print(f"Run {run_id} | documents by status: {json.dumps(dict(counts))}")


if __name__ == "__main__":
    main()
