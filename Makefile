.PHONY: install seed ingest eval test run docker-up docker-down clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

seed-data:
	$(PYTHON) scripts/generate_synthetic_data.py
	$(PYTHON) scripts/seed_database.py

ingest:
	$(PYTHON) scripts/ingest_docs.py

eval:
	$(PYTHON) scripts/run_eval.py --queries data/eval/test_queries.jsonl

test:
	DATABASE_URL='sqlite:///:memory:' $(PYTHON) -m pytest tests/ -v

run:
	uvicorn app.main:app --reload --port 8000

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".venv" -exec rm -rf {} +
