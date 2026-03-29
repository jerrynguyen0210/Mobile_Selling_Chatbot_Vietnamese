.DEFAULT_GOAL := help
SHELL         := /bin/bash
PYTHON        := python3
PIP           := $(PYTHON) -m pip

# Colour helpers
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "$(BOLD)Mobile Selling Chatbot Vietnamese$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { \
		printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
.PHONY: install
install: ## Install all dependencies (dev included)
	$(PIP) install -e ".[dev]"

.PHONY: env
env: ## Copy .env.example → .env (skips if .env already exists)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env from .env.example — fill in your secrets$(RESET)"; \
	else \
		echo ".env already exists, skipping"; \
	fi

# ---------------------------------------------------------------------------
# Development servers
# ---------------------------------------------------------------------------
.PHONY: run
run: ## Start all services via docker-compose (infra + backend + frontend)
	docker compose up --build

.PHONY: run-detach
run-detach: ## Start all services in detached mode
	docker compose up --build -d

.PHONY: run-infra
run-infra: ## Start only infrastructure (Postgres, Redis, Qdrant)
	docker compose up postgres redis qdrant -d

.PHONY: run-backend
run-backend: ## Run the FastAPI backend locally (requires infra running)
	cd back-end && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: run-frontend
run-frontend: ## Run the Streamlit frontend locally
	cd front-end && streamlit run chatting_bot.py

.PHONY: stop
stop: ## Stop and remove all docker-compose containers
	docker compose down

.PHONY: stop-volumes
stop-volumes: ## Stop containers AND remove all persistent volumes (destructive!)
	docker compose down -v

# ---------------------------------------------------------------------------
# Database migrations (Alembic)
# ---------------------------------------------------------------------------
.PHONY: migrate
migrate: ## Apply all pending Alembic migrations
	alembic upgrade head

.PHONY: migrate-create
migrate-create: ## Create a new migration (usage: make migrate-create MSG="add products table")
	@[ -n "$(MSG)" ] || (echo "$(BOLD)Usage: make migrate-create MSG=\"description\"$(RESET)" && exit 1)
	alembic revision --autogenerate -m "$(MSG)"

.PHONY: migrate-down
migrate-down: ## Rollback the last migration
	alembic downgrade -1

.PHONY: migrate-history
migrate-history: ## Show Alembic migration history
	alembic history --verbose

# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
.PHONY: ingest
ingest: ## Ingest product catalogue into Qdrant vector store
	$(PYTHON) -m back_end.scripts.ingest

.PHONY: ingest-reset
ingest-reset: ## Drop & re-create the Qdrant collection, then re-ingest
	$(PYTHON) -m back_end.scripts.ingest --reset

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run all tests with coverage
	pytest

.PHONY: test-unit
test-unit: ## Run unit tests only
	pytest tests/unit -v

.PHONY: test-integration
test-integration: ## Run integration tests only (requires infra)
	pytest tests/integration -v

.PHONY: test-watch
test-watch: ## Re-run tests on file changes (requires pytest-watch)
	ptw -- -v

# ---------------------------------------------------------------------------
# Linting & formatting
# ---------------------------------------------------------------------------
.PHONY: lint
lint: ## Run ruff linter
	ruff check back-end front-end

.PHONY: lint-fix
lint-fix: ## Run ruff linter and auto-fix issues
	ruff check --fix back-end front-end

.PHONY: format
format: ## Format code with ruff formatter
	ruff format back-end front-end

.PHONY: typecheck
typecheck: ## Run mypy type checker
	mypy back-end/app

.PHONY: check
check: lint typecheck ## Run all static analysis checks

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
.PHONY: logs
logs: ## Tail docker-compose logs for all services
	docker compose logs -f

.PHONY: logs-backend
logs-backend: ## Tail backend logs only
	docker compose logs -f backend

.PHONY: shell-backend
shell-backend: ## Open a shell in the backend container
	docker compose exec backend /bin/bash

.PHONY: shell-db
shell-db: ## Open a psql shell in the Postgres container
	docker compose exec postgres psql -U $${POSTGRES_USER:-chatbot} -d $${POSTGRES_DB:-chatbot}

.PHONY: clean
clean: ## Remove Python cache files and build artefacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info htmlcov .coverage coverage.xml
