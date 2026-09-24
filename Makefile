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

.DEFAULT_GOAL := help
.PHONY: help setup traindbnet traintrocr

help: ## Show this help
	@echo "Usage: make <target> [VAR=value]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (current value):"
	@echo "  PYTHON       $(PYTHON)  (interpreter used to create $(VENV))"
	@echo "  EPOCHS       $(EPOCHS)"
	@echo "  BATCH_SIZE   $(BATCH_SIZE)"
	@echo "  LR           $(LR)"
	@echo "  NUM_WORKERS  $(NUM_WORKERS)"
	@echo "  VAL_EVERY    $(VAL_EVERY)"
	@echo "  TROCR_EPOCHS $(TROCR_EPOCHS)  TROCR_BATCH_SIZE $(TROCR_BATCH_SIZE)  TROCR_LR $(TROCR_LR)"
	@echo ""
	@echo "Example: make traindbnet EPOCHS=50 BATCH_SIZE=4"

setup: $(VENV)/.installed ## Create .venv and install requirements.txt

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@touch $@

traindbnet: setup ## Train DBNet on dataset_receipt (checkpoints -> checkpoints/)
	$(PY) train_dbnet.py --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --lr $(LR) \
		--num-workers $(NUM_WORKERS) --val-every $(VAL_EVERY)

traintrocr: setup ## Train the ViT -> BERT text recognizer on word crops (checkpoints -> checkpoints/trocr/)
	$(PY) train_trocr.py --epochs $(TROCR_EPOCHS) --batch-size $(TROCR_BATCH_SIZE) --lr $(TROCR_LR) \
		--num-workers $(NUM_WORKERS)
