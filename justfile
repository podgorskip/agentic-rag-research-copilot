default:
    @just --list

install:
    uv sync

typecheck:
    uv run pyright

test:
    uv run pytest tests/

lint:
    uv run ruff check src/ tests/

format:
    uv run ruff format src/ tests/

run:
    uv run python src/main.py