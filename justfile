set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Show available commands
default:
    @just --list

# Install Python and project dependencies
install:
    uv python install
    uv sync --extra dev

# Run type checking with pyright
typecheck:
    uv run pyright

# Run tests with pytest
test:
    uv run pytest

# Lint code with ruff
lint:
    uv run ruff check src tests

# Format code with ruff
format:
    uv run ruff format src tests

# Check code formatting without modifying
format-check:
    uv run ruff format --check src tests

# Run all checks (lint, format-check, typecheck, test)
check: lint format-check typecheck test

# Run the main application
run:
    uv run python src/main.py

# Ingest sources using rag-ingest
ingest +sources:
    uv run rag-ingest {{sources}}

# Compare Dense vs. Hybrid vs. Reranked retrieval quality
validate-retrieval *args:
    uv run python scripts/validate_retrieval.py {{args}}

# Start Qdrant vector database
qdrant-up:
    docker compose up -d --wait qdrant

# Stop Qdrant vector database
qdrant-down:
    docker compose down

# View Qdrant logs
qdrant-logs:
    docker compose logs -f qdrant