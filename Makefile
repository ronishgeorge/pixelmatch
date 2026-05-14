# PixelMatch — developer convenience targets
.PHONY: install test lint format serve docker index train data clean coverage

PY ?= python3
PIP ?= pip3

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PY) data/generate_catalog.py --num 5000
	$(PY) data/generate_interactions.py --num 50000

test:
	PYTHONPATH=src $(PY) -m pytest tests/ -v

coverage:
	PYTHONPATH=src $(PY) -m pytest tests/ --cov=pixelmatch --cov-report=term-missing

lint:
	ruff check src/ tests/ data/

format:
	black src/ tests/ data/
	ruff check --fix src/ tests/ data/

serve:
	PYTHONPATH=src uvicorn pixelmatch.serving.server:app --host 0.0.0.0 --port 8080 --reload

index:
	PYTHONPATH=src $(PY) -m pixelmatch.retrieval.faiss_index --build

train:
	PYTHONPATH=src $(PY) -m pixelmatch.recommendation.two_tower --train

docker:
	docker build -t pixelmatch:latest .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
