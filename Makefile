# Convenience targets (Linux / macOS / WSL). See README.md for Windows equivalents.
.PHONY: help up down logs install backend frontend test test-backend test-frontend lint eval eval-real migrate

help:            ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

up:              ## Build and start the full Docker stack
	@test -f .env || cp .env.example .env
	docker compose up -d --build
	@echo "UI: http://localhost:3000   API docs: http://localhost:8000/docs"

down:            ## Stop the stack (keeps volumes)
	docker compose down

logs:            ## Tail backend + model download logs
	docker compose logs -f backend ollama-init

install:         ## Install backend (with ML extras) and frontend dependencies
	cd backend && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[ml,dev]"
	cd frontend && npm ci

backend:         ## Run the API locally with hot reload
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:        ## Run the Vite dev server (proxies /api to :8000)
	cd frontend && npm run dev

migrate:         ## Apply database migrations
	cd backend && alembic upgrade head

lint:            ## Lint + type-check everything
	cd backend && ruff check app tests evaluation && ruff format --check app tests evaluation && mypy app
	cd frontend && npm run typecheck

test: test-backend test-frontend  ## Run all tests

test-backend:    ## Backend unit + integration + evaluation tests
	cd backend && pytest

test-frontend:   ## Frontend component & unit tests
	cd frontend && npm test

eval:            ## Retrieval/hallucination evaluation with fast fakes
	cd backend && python -m evaluation.run

eval-real:       ## Evaluation with the real embedding + reranker models
	cd backend && python -m evaluation.run --models real
