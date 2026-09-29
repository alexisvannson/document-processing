# Data platform (optional extra)

This isn't part of the OCR brief: see the [main README](../README.md) for the OCR itself. It runs the OCR inside
a small pipeline, to measure what OCR errors cost downstream: a receipt is published only if its items add up to
its total, so running the same pipeline on perfect OCR and on the trained models isolates the error OCR adds
(`notebooks/eval_system.ipynb`).

The design rule: **deterministic steps produce trusted data.** Every step is repeatable and logged, a receipt that
fails a check is quarantined rather than published, and nothing downstream of the redaction step can read
unredacted text.

## Architecture

```
 dataset_receipt/        pipeline/ (Python CLI)                          dbt/
   images/*.png  ───►  ingest ──► ocr ──► redact ──► extract ───────►  checks ──► publish or quarantine
   metadata.pkl            │        │        │          │                 │
                           ▼        ▼        ▼          ▼                 ▼
                 ┌─────────────────────── Postgres: warehouse ───────────────────────┐
                 │ ops             documents (status per receipt), pipeline_runs     │
                 │ raw_restricted  ocr_tokens, unredacted (no analyst access)        │
                 │ raw             ocr_tokens (PII masked), extracted_lines          │
                 │ staging         dbt views + per-receipt checks                    │
                 │ marts           fct_receipts, fct_line_items, receipt_quarantine  │
                 └───────────────────────────────────────────────────────────────────┘
```

## Layout

```
pipeline/                receipts pipeline (plain Python CLI)
  run.py                 python -m data_platform.pipeline.run [--steps ...] [--ocr model] [--corrupt 0.1] [--reset]
  ingest.py ocr.py redact.py extract.py   the four steps
  lines.py               deskew + group word boxes into text lines
  db.py runs.py          per-receipt transactions and statuses; run bookkeeping in ops.pipeline_runs
  snapshot.py            export one row per receipt to a CSV (results/, read by notebooks/eval_system.ipynb)
  table.py               the same steps and checks in pandas, no database: one DataFrame of (amount, item) tuples
dbt/                     staging -> checks -> marts, with tests
infra/
  docker-compose.yml     Postgres
  init/                  database setup on first start: warehouse schemas, read-only `analyst` role
```

## Quick start

Run every command from the repo root.

**Prerequisites**
- Docker Desktop.
- `make setup` (see the main README) and the data in `dataset_receipt/`.
- For `OCR=model`: `checkpoints/dbnet_best.pt` and `checkpoints/trocr-printed/best`.

```bash
cp .env.example .env     # optional: passwords and port
make up                  # Postgres on :5433
make pipeline            # load the 200 receipts: ingest -> OCR -> redact -> extract (OCR=model for the trained models)
make dbt                 # build and test the marts
make psql                # look at the results
```

**Comparing OCR engines** (`notebooks/eval_system.ipynb`):

```bash
make pipeline && make dbt
.venv/bin/python -m data_platform.pipeline.snapshot results/system_ground_truth.csv
make pipeline OCR=model && make dbt
.venv/bin/python -m data_platform.pipeline.snapshot results/system_models.csv
```

## Components

### Pipeline (`pipeline/`)

Four steps. Each reads its input from Postgres and writes its output there, one transaction per receipt. The
status in `ops.documents` decides which step picks a receipt up next:

```
pending ─ocr─► ocr_done ─redact─► redacted ─extract─► extracted
                   └─► ocr_failed      └─► redact_failed    └─► extract_failed   (error recorded)
```

- **Idempotent:** images are keyed by file hash, so re-ingesting is a no-op, and a step only processes receipts
  still waiting for it. `--reset` reprocesses everything.
- **Isolated failures:** a failing receipt is rolled back, parked with its error, and its stale output is
  deleted; the other receipts continue.
- **OCR engine:** `--ocr model` runs DBNet and the fine-tuned recognizer (`ocr/inference.py`). The default,
  `--ocr ground_truth`, reads the annotated words from `metadata.pkl`. It's fast, and it's the "perfect OCR"
  baseline. `--corrupt 0.1` swaps a digit in 10% of the numbers to simulate recognition errors.
- **Redaction** (`redact.py`) runs before extraction, so extracted fields can never carry PII. It masks emails,
  phone numbers, masked card numbers and 13–19 digit numbers that pass the Luhn checksum (product barcodes
  usually don't), plus numbers and names after keywords such as TELP, CARD or KASIR. Unredacted tokens stay in
  `raw_restricted`.
- **Extraction** (`lines.py`, `extract.py`) is rule-based, so it's repeatable and easy to audit. It deskews the
  page using the median slope of the words, groups words into lines, and labels each line by keyword, in English
  and Indonesian: `item`, `subtotal`, `tax`, `service`, `discount`, `total`, `cash`, `card`, `change`… It parses
  `16,500` / `23.000` / `Rp 45.000`, quantities and unit prices. Every extracted line keeps `token_idxs`, pointing
  back to its boxes in the image. `extract_lines(tokens)` needs no database.

On the ground-truth OCR: all 200 receipts are ingested, 184 are extracted, and 16 fail with no readable TOTAL
amount.

### dbt (`dbt/`)

`dbt build` turns `raw.extracted_lines` into published tables. `int_receipt_checks` computes, per receipt:

- `no_items`: no item line was found
- `items_do_not_reconcile`: the items (with discounts) don't add up to the subtotal or any TOTAL line, within 1
- `change_inconsistent`: `cash − total ≠ change`, checked when both are printed

Each extracted receipt lands in exactly one mart:

| Mart | Contents |
| --- | --- |
| `marts.fct_receipts` | receipts that passed: total (the last TOTAL line, i.e. the amount paid), subtotal, tax, service, cash, change, payment method |
| `marts.fct_line_items` | their items: name, normalized name, qty, unit price, amount, source boxes |
| `marts.receipt_quarantine` | the others, with `failed_checks` |

On ground truth: 146 published and 38 quarantined (`make table` reproduces these numbers without the database).
The CSVs in `results/system_*.csv` were exported with the earlier rules (139 published). The 25
tests include two of my own: no receipt is lost between the marts, and no published item name looks like a phone
number, card number or email. A failing test stops the marts from being rebuilt, so downstream readers keep the
last good data.

## How I'd deploy it

Not built here: with 200 fixed receipts, a scheduler, a dashboard and an agent would add moving parts without
adding evidence about the OCR. Once receipts arrive continuously, I'd add them in this order:

1. **Airflow, to run the pipeline.** One DAG, triggered when photos land in object storage (or hourly):
   `ingest → ocr → redact → extract → dbt build`. The OCR task runs the models in their own GPU container
   (`KubernetesPodOperator`), so Airflow only orchestrates and never installs torch. Each task retries, and a
   failure is recorded in `ops.pipeline_runs`. When a new model version ships, a backfill reprocesses old receipts
   and the two versions are compared on the same checks.
2. **Metabase, to watch quality.** Connected as the read-only `analyst` role: published vs quarantined receipts
   over time, the reasons for quarantine, mean OCR confidence per day (a drop signals a new kind of photo), and
   the last run's status. An alert fires when the quarantine rate jumps.
3. **An agent (LangGraph), last and only if people ask questions the dashboard can't answer.** Read-only SQL
   tools under the `analyst` role, a statement timeout, the data's freshness stated in every answer, and PII
   masked in the output. LangGraph earns its place over a plain loop when a person must approve a step, such as a
   reviewer accepting a corrected total for a quarantined receipt. Before it ships: a fixed set of questions with
   expected answers.

## Configuration

Settings live in `.env` at the repo root, which is gitignored; start with `cp .env.example .env`. The Makefile
exports it to every target and passes it to docker compose. Without a `.env`, everything falls back to the same
local defaults.

| Variable | Used for |
| --- | --- |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` | the warehouse container |
| `ANALYST_PASSWORD` | the read-only role; only applied when the Postgres volume is first created |
| `DATABASE_URL`, `DBT_*` | connections from the host, derived from the values above |

Keep `.env` to plain `KEY=value` lines, without quotes, because the Makefile includes it directly.

## Limitations

- **The model run in `results/` covers 164 of the 200 receipts.** It was stopped at about 18 s per receipt on CPU.
  `python -m data_platform.pipeline.run --ocr model` (without `--reset`) finishes the rest.
- **Extraction is rule-based and tuned on these 200 receipts.**
- **The receipts carry almost no PII.** 1 phone number and 1 card number are masked across the dataset.
- **Everything is set up for local development:** default passwords, one Postgres container. On AWS this would
  map to RDS and Secrets Manager.
