# document-processing

Turning photos of messy documents into data you can trust and question. Receipts (the CORD dataset) stand
in for medical reports: both are unstructured, photographed in bad conditions, contain personal data, and
need specific values extracted reliably.

The repo has two halves:

- **OCR models**: a text detector (DBNet) and a text recognizer (ViT encoder + BERT decoder), trained on
  the receipts.
- **Data platform**: a pipeline that loads receipts into Postgres, redacts PII, extracts fields, validates
  them with dbt, runs daily under Airflow, and is exposed through a Metabase dashboard and a LangGraph agent.

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

The main design rule: **deterministic pipelines produce trusted data; the agent only reasons over it.**
Every step is repeatable and logged, a receipt that fails a check is quarantined rather than published,
and nothing downstream of the redaction step can read unredacted text.

## Repository layout

```
ocr/                     OCR models and their data code
  models/                DBNET.py (ResNet-18 + FPN + DB head), VIT.py + BERT.py + TrOCR.py (recognizer), CRAFT.py
  dataset.py             detection and recognition datasets, augmentations, 70/15/15 split by receipt
  DBLoss.py              DBNet loss
  getDBprobMap.py        ground-truth probability and threshold maps from word boxes
  getDBbboxes.py         probability map -> text boxes (post-processing)
training/                train_dbnet.py, train_trocr.py (W&B logging, best checkpoint on val loss)
notebooks/               Colab notebooks: train / evaluate DBNet, train the recognizer (from scratch, or
                         fine-tune trocr-base-printed), and eda_words (word and character imbalance)

pipeline/                receipts pipeline (plain Python, runs from the CLI or Airflow)
  run.py                 CLI: python -m pipeline.run [--steps ...] [--corrupt 0.1] [--reset] [--run-id ...]
  ingest.py ocr.py redact.py extract.py   the four steps
  lines.py               deskew + group word boxes into text lines
  db.py runs.py          per-receipt transactions and statuses; run bookkeeping in ops.pipeline_runs
dbt/                     staging -> checks -> marts, with tests
dags/receipts_daily.py   Airflow DAG
agent/                   LangGraph agent: graph.py (nodes, CLI), tools.py (read-only SQL, receipt lookup)

infra/
  docker-compose.yml     Postgres, Metabase, Airflow
  init/                  database setup on first start: databases, warehouse schemas, `analyst` role
  airflow/Dockerfile     Airflow image with separate venvs for the pipeline and dbt
  metabase/setup.py      provisions the Metabase admin, warehouse connection and dashboard
Makefile                 every command below (`make help` lists them)
.env.example             settings template
```

Not in git: `dataset_receipt/` (the data), `checkpoints/` (trained weights), `.env`.

## Quick start

**Prerequisites**
- Docker Desktop. Give it 6–8 GB of memory: Postgres, Metabase and Airflow use about 2.5 GB together.
- Python 3.12. `make setup` creates `.venv` from `requirements.txt`.
- The data in `dataset_receipt/`: `images/*.png` plus `metadata.pkl`, a DataFrame with one row per receipt:
  `file_name`, `split_origin`, `words` (list of strings) and `bboxes` (list of 4-point boxes `x1..y4`).
- For the agent only: a Gemini API key (the free tier works, with Flash models).

**Run the data platform**

```bash
cp .env.example .env     # optional: passwords, ports, GEMINI_API_KEY
make setup               # .venv with all Python dependencies
make up                  # Postgres :5433, Metabase :3000, Airflow :8080
make pipeline            # load the 200 receipts: ingest -> OCR -> redact -> extract
make dbt                 # build and test the marts
make dashboard           # create the Metabase dashboard (prints its URL and login)
make ask Q="What's the average receipt total by payment method?"
```

After that, Airflow runs the same steps every day. `make trigger` starts a run immediately. The Airflow UI
is at http://localhost:8080 (no login locally).

**Train the OCR models**

```bash
make traindbnet EPOCHS=50 BATCH_SIZE=4     # -> checkpoints/dbnet_{best,last}.pt
make traintrocr TROCR_EPOCHS=30            # -> checkpoints/trocr/{best,last}
```

On a GPU, use the Colab notebooks in `notebooks/`. They copy the data from Google Drive, run the same
training scripts, and save checkpoints back to Drive.

## Components

### OCR models (`ocr/`, `training/`)

OCR is done in two steps: find the text, then read it.

**Detection: DBNet.** A ResNet-18 backbone with a feature pyramid predicts, for every pixel, the
probability that it is text, plus a threshold map. "Differentiable binarization" learns where to cut
between neighbouring lines, which suits dense receipt text. `getDBbboxes.py` turns the probability map
back into boxes. Evaluation reports precision, recall and F1 of the boxes at IoU ≥ 0.5. I chose DBNet over
the alternatives I considered:

| Approach | Why not (for receipts) |
| --- | --- |
| Faster R-CNN | slow and memory-hungry |
| YOLO | struggles with long, thin text lines and dense packing |
| UNet segmentation | merges close lines; heavy post-processing |
| CRAFT (`ocr/models/CRAFT.py`) | character-level; kept as an alternative, no training script |

**Recognition: ViT → BERT.** A TrOCR-style encoder-decoder: a pretrained ViT encodes each word crop, and
a BERT decoder with cross-attention generates the text one character at a time. The tokenizer is
character-level, because WordPiece can't round-trip strings like `16,500`. The training script reports
character error rate and exact word accuracy.

**Robustness.** Detector training uses augmentations for the physical defects of real photos: shadows and
lighting gradients, creases (elastic and grid distortion, rotations), and contrast changes. The recognizer
uses lighter affine ones (rotation, shear, scale) on each word crop. Both scripts split the
receipts 70/15/15 into train, validation and test with a fixed seed, pick the checkpoint on validation
loss, and score the test set once at the end.

### Pipeline (`pipeline/`)

Four steps. Each reads its input from Postgres and writes its output there, one transaction per receipt.
The status in `ops.documents` decides which step picks a receipt up next:

```
pending ─ocr─► ocr_done ─redact─► redacted ─extract─► extracted
                   └─► ocr_failed      └─► redact_failed    └─► extract_failed   (error recorded)
```

- **Idempotent:** images are keyed by file hash, so re-ingesting is a no-op, and a step only processes
  receipts still waiting for it. `--reset` reprocesses everything.
- **Isolated failures:** a failing receipt is rolled back, parked with its error, and its stale output
  is deleted; the other receipts continue.
- **OCR engine:** for now the words and boxes come from the ground truth in `metadata.pkl`, so the rest
  of the platform could be built before the models were ready. `--corrupt 0.1` swaps a digit in 10% of
  the numbers to simulate recognition errors.
- **Redaction** (`redact.py`) runs before extraction, so extracted fields can never carry PII. It
  masks emails, phone numbers, masked card numbers and 13–19 digit numbers that pass the Luhn checksum
  (product barcodes usually don't), plus numbers and names after keywords such as TELP, CARD or KASIR.
  Unredacted tokens stay in `raw_restricted`.
- **Extraction** (`lines.py`, `extract.py`) is rule-based, so it's repeatable and easy to audit. It
  deskews the page using the median slope of the words, groups words into lines, and labels each line
  by keyword, in English and Indonesian: `item`, `subtotal`, `tax`, `service`, `discount`, `total`,
  `cash`, `card`, `change`… It parses `16,500` / `23.000` / `Rp 45.000`, quantities and unit prices.
  Every extracted line keeps `token_idxs`, pointing back to its boxes in the image.

On the ground-truth OCR: all 200 receipts are ingested, 177 are extracted, and 23 fail with no readable
TOTAL amount.

### dbt (`dbt/`)

`dbt build` turns `raw.extracted_lines` into published tables. `int_receipt_checks` computes, per receipt:

- `no_items`: no item line was found
- `items_do_not_reconcile`: the items (with discounts) don't add up to the subtotal or any TOTAL line,
  within 1
- `change_inconsistent`: `cash − total ≠ change`, checked when both are printed

Each extracted receipt lands in exactly one mart:

| Mart | Contents |
| --- | --- |
| `marts.fct_receipts` | receipts that passed: total (the last TOTAL line, i.e. the amount paid), subtotal, tax, service, cash, change, payment method |
| `marts.fct_line_items` | their items: name, normalized name, qty, unit price, amount, source boxes |
| `marts.receipt_quarantine` | the others, with `failed_checks` |

On ground truth: 139 published and 38 quarantined. With `CORRUPT=0.1`: 92 published and 85 quarantined.
The 25 tests include two of my own: no receipt is lost between the marts, and no published item name
looks like a phone number, card number or email. A failing test stops the marts from being rebuilt, so
downstream readers keep the last good data. The `analyst` role gets SELECT on every mart that is rebuilt.

### Airflow (`dags/receipts_daily.py`)

`start → ingest → ocr → redact → extract → dbt_build → finish`, scheduled daily with one retry per task.

- Each step runs the pipeline CLI under the DAG run's id, so `ops.pipeline_runs` has one row per run,
  with every step's stats.
- A task that still fails after its retry marks the run `failed` and names the task. The dashboard and
  the agent read this.
- The image keeps the pipeline and dbt in their own venvs, isolated from Airflow's packages. The repo is
  mounted read-only.

### Metabase

`make dashboard` provisions everything through Metabase's API and can be re-run:

- the admin account, on first run
- a connection to the warehouse as the read-only `analyst` role
- a "Receipts" dashboard: published, quarantined and failed counts, last run status, receipt totals,
  payment methods, top items, quarantine reasons and recent pipeline runs

### Agent (`agent/`)

A LangGraph graph that answers questions about the receipts, using Gemini (`gemini-3.8-flash` by default)
through Google's `google-genai` SDK. The SDK's automatic function calling is off, so the graph owns the loop:

```
START → check_freshness → agent ⇄ tools → guard → END
```

- **check_freshness** reads the last pipeline run and the receipt counts by status before the model sees
  the question. Answers can then say how fresh the data is and what they leave out.
- **tools**: `run_sql` (one SELECT, at most 100 rows) and `get_receipt(doc_id)`.
  - Both connect as `analyst`, in read-only transactions with a 5 s statement timeout.
  - Postgres itself rejects writes and access to `raw` or `raw_restricted`.
- **guard** masks anything that looks like a phone number, card number or email in the final answer.
- At most 8 tool rounds per question. Overloaded-model errors (503) are retried with backoff, then the
  question goes to `GEMINI_FALLBACK_MODEL`.

```bash
make ask Q="Why is receipt 57 quarantined?"     # --verbose output shows the data status and tool calls
```

## Configuration

Settings live in `.env`, which is gitignored; start with `cp .env.example .env`. The Makefile exports it
to every target and passes it to docker compose. Without a `.env`, everything falls back to the same
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
| `make setup` | create `.venv` and install `requirements.txt` |
| `make up` / `make down` | start / stop Postgres, Metabase and Airflow; data persists in the `pgdata` volume |
| `make pipeline [CORRUPT=0.1] [OCR=model RECOGNIZER=...]` | reprocess every receipt through the pipeline, with ground-truth OCR or the trained models |
| `make dbt` | build and test the dbt models |
| `make dashboard` | create or refresh the Metabase dashboard |
| `make trigger` | run the Airflow DAG now |
| `make ask Q="..."` | ask the agent a question |
| `make psql` | psql shell on the warehouse |
| `make traindbnet` / `make traintrocr` | train the detector / recognizer (`make help` lists the variables) |

## Current limitations

- **The pipeline doesn't use the trained models yet.** Its OCR step reads the ground truth. The next
  step is an OCR engine that runs DBNet + the recognizer, then measuring how much field accuracy drops
  compared with ground truth.
- **Extraction is rule-based and tuned on these 200 receipts.** It reconciles 142 of 200 on ground truth.
- **The receipts carry almost no PII.** 1 phone number and 1 card number are masked across the dataset.
  Medical reports would need a NER model behind the rules, with recall measured on a hand-labelled set.
- **The agent has no evaluation set yet.** It answers real questions correctly (checked against SQL),
  but it still needs a fixed set of questions with expected answers to measure changes.
- **Everything is set up for local development:** default passwords, no Airflow login, a single-container
  Airflow (`standalone`). On AWS this would map to RDS, managed Airflow and Secrets Manager.
