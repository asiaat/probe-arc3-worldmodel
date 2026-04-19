.PHONY: help setup test fetch-baselines collect-data buffer-stats train-wm lint format clean

# Load .env if present
-include .env
export

PYTHON := python3.12
VENV := .venv
BIN := $(VENV)/bin

help:
	@echo "arcwm — ARC-AGI-3 World Model Agent"
	@echo ""
	@echo "Setup:"
	@echo "  make setup            Create venv, install package (editable), install dev deps"
	@echo "  make test             Run pytest"
	@echo ""
	@echo "Week 1 — Data collection:"
	@echo "  make fetch-baselines  Fetch real human baselines from SDK scorecard"
	@echo "  make collect-data     Collect random-policy transitions into replay buffer"
	@echo "  make buffer-stats     Show statistics of the replay buffer"
	@echo ""
	@echo "Week 2 — World model:"
	@echo "  make train-wm         Train VAE + transition model"
	@echo ""
	@echo "Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Format code with black"
	@echo ""
	@echo "  make clean            Remove build artifacts"

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	@echo ""
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"
	@echo "Remember to: cp .env.example .env  and fill in your ARC_API_KEY"

test:
	$(BIN)/pytest tests/ -v

fetch-baselines:
	$(BIN)/python scripts/fetch_baselines.py

collect-data:
	$(BIN)/python scripts/collect_data.py \
	    --games $(TRAIN_GAMES) \
	    --target $(COLLECTION_TARGET_TRANSITIONS) \
	    --max-steps $(COLLECTION_MAX_STEPS_PER_EPISODE) \
	    --output-dir $(DATA_DIR)/replay

buffer-stats:
	$(BIN)/python -m arcwm.env.replay_buffer stats --path $(DATA_DIR)/replay

train-wm:
	$(BIN)/python scripts/train_world_model.py \
	    --buffer-path $(DATA_DIR)/replay \
	    --epochs 10 \
	    --batch-size 64 \
	    --latent-dim 64

lint:
	$(BIN)/ruff check arcwm/ scripts/ tests/

format:
	$(BIN)/black arcwm/ scripts/ tests/
	$(BIN)/ruff check --fix arcwm/ scripts/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
