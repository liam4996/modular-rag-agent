.PHONY: help dashboard mcp test test-unit test-quick lint typecheck ingest clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dashboard: ## Start the Streamlit Dashboard
	streamlit run src/observability/dashboard/app.py

mcp: ## Start the MCP Server
	python -m main

test: ## Run all tests (except slow/external)
	pytest tests/ -q -m "not slow"

test-unit: ## Run unit tests only
	pytest tests/unit/ -q

test-quick: ## Run quick tests (no LLM, no external)
	pytest tests/ -q -m "not (llm or slow)"

test-all: ## Run all tests (may require external services)
	pytest tests/ -q

lint: ## Lint code with ruff
	ruff check src/

typecheck: ## Type check with mypy
	mypy src/

ingest: ## Ingest documents: make ingest path=./docs/file.pdf
	python scripts/ingest.py --path $(path)

query: ## Query: make query q="your question"
	python scripts/query.py --query "$(q)"

docker-build: ## Build Docker image
	docker compose build

docker-up: ## Start all services with Docker
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

clean: ## Clean temporary files
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
