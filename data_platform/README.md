# Data platform (optional extra)

This isn't part of the OCR brief: see the [main README](../README.md) for the OCR itself. It runs the OCR inside
a small data platform, to show how it would be used on real documents. Receipts stand in for medical reports:
both are unstructured, photographed in bad conditions, contain personal data, and need specific values extracted
reliably.

The main design rule: **deterministic pipelines produce trusted data; the agent only reasons over it.** Every step
is repeatable and logged, a receipt that fails a check is quarantined rather than published, and nothing
downstream of the redaction step can read unredacted text.

## Architecture

```
                    ┌──────────────── Airflow: receipts_daily (daily) ────────────────┐
                    │                                                                   │
 dataset_receipt/   │  ingest ──► ocr ──► redact ──► extract ──► dbt build ──► finish   │
   images/*.png ────┼─►  │         │        │          │            │                   │
   metadata.pkl     └────┼─────────┼────────┼──────────┼────────────┼───────────────────┘
                         ▼         ▼        ▼          ▼            ▼
                 ┌─────────────────────── Postgres: warehouse ───────────────────────┐
                 │ ops             documents (status per receipt), pipeline_runs     │
                 │ raw_restricted  ocr_tokens, unredacted (no analyst access)        │
                 │ raw             ocr_tokens (PII masked), extracted_lines          │
                 │ staging         dbt views + per-receipt checks                    │
                 │ marts           fct_receipts, fct_line_items, receipt_quarantine  │
                 └──────────────────────────────────┬────────────────────────────────┘
                                                    │ read-only role `analyst`
                                     ┌──────────────┴──────────────┐
                                     ▼                             ▼
                              Metabase dashboard            LangGraph agent (Gemini)
```

## Layout

```
pipeline/                receipts pipeline (plain Python, runs from the CLI or Airflow)
  run.py                 CLI: python -m data_platform.pipeline.run [--steps ...] [--ocr model] [--corrupt 0.1] [--reset]
  ingest.py ocr.py redact.py extract.py   the four steps
  lines.py               deskew + group word boxes into text lines
  db.py runs.py          per-receipt transactions and statuses; run bookkeeping in ops.pipeline_runs
  snapshot.py            export one row per receipt to a CSV (results/, read by notebooks/eval_system.ipynb)
dbt/                     staging -> checks -> marts, with tests
dags/receipts_daily.py   Airflow DAG
agent/                   LangGraph agent: graph.py (nodes, CLI), tools.py (read-only SQL, receipt lookup)
infra/
  docker-compose.yml     Postgres, Metabase, Airflow
  init/                  database setup on first start: databases, warehouse schemas, `analyst` role
  airflow/Dockerfile     Airflow image with separate venvs for the pipeline and dbt
  metabase/setup.py      provisions the Metabase admin, warehouse connection and dashboard
```

## Quick start

Run every command from the repo root.

**Prerequisites**
- Docker Desktop. Give it 6–8 GB of memory: Postgres, Metabase and Airflow use about 2.5 GB together.
- `make setup` (see the main README) and the data in `dataset_receipt/`.
- For `OCR=model`: `checkpoints/dbnet_best.pt` and `checkpoints/trocr-printed/best`.
- For the agent only: a Gemini API key (the free tier works, with Flash models).

```bash
cp .env.example .env     # optional: passwords, ports, GEMINI_API_KEY
make up                  # Postgres :5433, Metabase :3000, Airflow :8080
make pipeline            # load the 200 receipts: ingest -> OCR -> redact -> extract (OCR=model for the trained models)
make dbt                 # build and test the marts
make dashboard           # create the Metabase dashboard (prints its URL and login)
make ask Q="What's the average receipt total by payment method?"
```

After that, Airflow runs the same steps every day. `make trigger` starts a run immediately. The Airflow UI is at
http://localhost:8080 (no login locally).

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
  back to its boxes in the image.

On the ground-truth OCR: all 200 receipts are ingested, 177 are extracted, and 23 fail with no readable TOTAL
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

On ground truth: 139 published and 38 quarantined. With `CORRUPT=0.1`: 92 published and 85 quarantined. The 25
tests include two of my own: no receipt is lost between the marts, and no published item name looks like a phone
number, card number or email. A failing test stops the marts from being rebuilt, so downstream readers keep the
last good data. The `analyst` role gets SELECT on every mart that is rebuilt.

### Airflow (`dags/receipts_daily.py`)

`start → ingest → ocr → redact → extract → dbt_build → finish`, scheduled daily with one retry per task.

- Each step runs the pipeline CLI under the DAG run's id, so `ops.pipeline_runs` has one row per run, with every
  step's stats.
- A task that still fails after its retry marks the run `failed` and names the task. The dashboard and the agent
  read this.
- The image keeps the pipeline and dbt in their own venvs, isolated from Airflow's packages. The repo is mounted
  read-only.

### Metabase

`make dashboard` provisions everything through Metabase's API and can be re-run:

- the admin account, on first run
- a connection to the warehouse as the read-only `analyst` role
- a "Receipts" dashboard: published, quarantined and failed counts, last run status, receipt totals, payment
  methods, top items, quarantine reasons and recent pipeline runs

### Agent (`agent/`)

A LangGraph graph that answers questions about the receipts, using Gemini (`gemini-3.8-flash` by default) through
Google's `google-genai` SDK. The SDK's automatic function calling is off, so the graph owns the loop:

```
START → check_freshness → agent ⇄ tools → guard → END
```

- **check_freshness** reads the last pipeline run and the receipt counts by status before the model sees the
  question. Answers can then say how fresh the data is and what they leave out.
- **tools**: `run_sql` (one SELECT, at most 100 rows) and `get_receipt(doc_id)`.
  - Both connect as `analyst`, in read-only transactions with a 5 s statement timeout.
  - Postgres itself rejects writes and access to `raw` or `raw_restricted`.
- **guard** masks anything that looks like a phone number, card number or email in the final answer.
- At most 8 tool rounds per question. Overloaded-model errors (503) are retried with backoff, then the question
  goes to `GEMINI_FALLBACK_MODEL`.

```bash
make ask Q="Why is receipt 57 quarantined?"     # --verbose output shows the data status and tool calls
```

## Configuration

Settings live in `.env` at the repo root, which is gitignored; start with `cp .env.example .env`. The Makefile
exports it to every target and passes it to docker compose. Without a `.env`, everything falls back to the same
local defaults.

| Variable | Used for |
| --- | --- |
| `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL` | the agent |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` | the warehouse container |
| `ANALYST_PASSWORD` | the read-only role; only applied when the Postgres volume is first created |
| `METABASE_PORT`, `AIRFLOW_PORT` | host ports |
| `DATABASE_URL`, `AGENT_DATABASE_URL`, `DBT_*` | connections from the host, derived from the values above |
| `MB_URL`, `MB_ADMIN_EMAIL`, `MB_ADMIN_PASSWORD` | Metabase provisioning |

Keep `.env` to plain `KEY=value` lines, without quotes, because the Makefile includes it directly.

## Commands

| Command | What it does |
| --- | --- |
| `make up` / `make down` | start / stop Postgres, Metabase and Airflow; data persists in the `pgdata` volume |
| `make pipeline [OCR=model] [CORRUPT=0.1]` | reprocess every receipt, with ground-truth OCR or the trained models |
| `make dbt` | build and test the dbt models |
| `make dashboard` | create or refresh the Metabase dashboard |
| `make trigger` | run the Airflow DAG now |
| `make ask Q="..."` | ask the agent a question |
| `make psql` | psql shell on the warehouse |

## Limitations

- **Airflow runs the ground-truth OCR only.** Its image doesn't install torch, to stay small. Running the models
  on a schedule would need a GPU worker, or a separate OCR service the DAG calls.
- **The model run in `results/` covers 164 of the 200 receipts.** It was stopped at about 18 s per receipt on CPU.
  `python -m data_platform.pipeline.run --ocr model` (without `--reset`) finishes the rest.
- **Extraction is rule-based and tuned on these 200 receipts.**
- **The receipts carry almost no PII.** 1 phone number and 1 card number are masked across the dataset. Medical
  reports would need a NER model behind the rules, with recall measured on a hand-labelled set.
- **The agent has no evaluation set yet.** It answers real questions correctly (checked against SQL), but it
  still needs a fixed set of questions with expected answers to measure changes.
- **Everything is set up for local development:** default passwords, no Airflow login, a single-container Airflow
  (`standalone`). On AWS this would map to RDS, managed Airflow and Secrets Manager.
