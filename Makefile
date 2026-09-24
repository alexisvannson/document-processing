# torch has no wheels for the newest Python yet, so prefer 3.12 when available
PYTHON ?= $(shell command -v python3.12 || command -v python3)
VENV := .venv
PY := $(VENV)/bin/python

EPOCHS ?= 300
BATCH_SIZE ?= 8
LR ?= 1e-3
NUM_WORKERS ?= 4
VAL_EVERY ?= 10

.DEFAULT_GOAL := help
.PHONY: help setup traindbnet

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
