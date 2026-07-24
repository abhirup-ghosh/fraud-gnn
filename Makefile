.PHONY: setup features train eval baseline test up seed loadgen down

export PYTHONPATH := src

setup:
	uv sync

features:
	uv run python -m scripts.load_featurestore

train:
	uv run python -m fraud_gnn.train

eval:
	uv run python -m fraud_gnn.evaluate

baseline:
	uv run python -m fraud_gnn.baseline

test:
	uv run pytest -q

up:
	docker compose up --build -d

seed:
	uv run python -m scripts.load_featurestore

loadgen:
	uv run python scripts/loadgen.py --n 5000

down:
	docker compose down -v
