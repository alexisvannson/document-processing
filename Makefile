# Local settings (cp .env.example .env). Optional: every value has a default in the code.
# Its keys are exported to every recipe, so Python, dbt and docker compose all see them.
-include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env 2>/dev/null)

# torch has no wheels for the newest Python yet, so prefer 3.12 when available
PYTHON ?= $(shell command -v python3.12 || command -v python3)
VENV := .venv
PY := $(VENV)/bin/python

EPOCHS ?= 300
BATCH_SIZE ?= 8
LR ?= 1e-3
NUM_WORKERS ?= 4
VAL_EVERY ?= 10
TROCR_EPOCHS ?= 30
TROCR_BATCH_SIZE ?= 16
TROCR_LR ?= 5e-5
CORRUPT ?= 0.0
OCR ?= ground_truth
Q ?= How many receipts were published, and how fresh is the data?
COMPOSE := docker compose $(if $(wildcard .env),--env-file .env) -f infra/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help setup traindbnet traintrocr up down psql pipeline dbt dashboard trigger ask

help: ## Show this help
	@echo "Usage: make <target> [VAR=value]"
	@echo ""
	@echo "Targets:"
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (current value):"
	@echo "  PYTHON       $(PYTHON)  (interpreter used to create $(VENV))"
	@echo "  EPOCHS       $(EPOCHS)"
	@echo "  BATCH_SIZE   $(BATCH_SIZE)"
	@echo "  LR           $(LR)"
	@echo "  NUM_WORKERS  $(NUM_WORKERS)"
	@echo "  VAL_EVERY    $(VAL_EVERY)"
	@echo "  CORRUPT      $(CORRUPT)  (fraction of amounts the pipeline corrupts, simulating OCR errors)"
	@echo "  TROCR_EPOCHS $(TROCR_EPOCHS)  TROCR_BATCH_SIZE $(TROCR_BATCH_SIZE)  TROCR_LR $(TROCR_LR)"
	@echo ""
	@echo "Example: make traindbnet EPOCHS=50 BATCH_SIZE=4"

setup: $(VENV)/.installed ## Create .venv and install requirements.txt

# The venv is only created once; a changed requirements.txt just reruns pip install
# (re-running `venv` on an existing .venv with another interpreter breaks it).
$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

# --prefer-binary: take the newest version with a prebuilt wheel rather than compiling one
# (the latest cryptography, pulled in by google-genai, has no Intel-Mac wheel).
$(VENV)/.installed: requirements.txt | $(PY)
	$(PY) -m pip install --prefer-binary -r requirements.txt
	@touch $@

traindbnet: setup ## Train DBNet on dataset_receipt (checkpoints -> checkpoints/)
	$(PY) -m training.train_dbnet --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --lr $(LR) \
		--num-workers $(NUM_WORKERS) --val-every $(VAL_EVERY)

traintrocr: setup ## Train the ViT -> BERT text recognizer on word crops (checkpoints -> checkpoints/trocr/)
	$(PY) -m training.train_trocr --epochs $(TROCR_EPOCHS) --batch-size $(TROCR_BATCH_SIZE) --lr $(TROCR_LR) \
		--num-workers $(NUM_WORKERS)

up: ## Start Postgres, Metabase and Airflow (ports in .env: 5433, 3000, 8080)
	$(COMPOSE) up -d --wait

down: ## Stop the containers (data is kept in the pgdata volume)
	$(COMPOSE) down

psql: ## Open a psql shell on the warehouse
	$(COMPOSE) exec postgres psql -U pipeline -d warehouse

pipeline: setup ## Run ingest -> OCR -> redact -> extract into Postgres (OCR=model for the trained models)
	$(PY) -m pipeline.run --reset --ocr $(OCR) --corrupt $(CORRUPT)

dbt: setup ## Build the dbt models and run their tests (staging -> marts, quarantine)
	$(VENV)/bin/dbt build --project-dir dbt --profiles-dir dbt

dashboard: ## Create/refresh the Metabase "Receipts" dashboard (after `make up`)
	$(PY) infra/metabase/setup.py

trigger: ## Trigger a run of the receipts_daily Airflow DAG
	$(COMPOSE) exec airflow airflow dags trigger receipts_daily

ask: setup ## Ask the LangGraph agent a question: make ask Q="..." (needs GEMINI_API_KEY)
	$(PY) -m agent.graph --verbose "$(Q)"
