default:
    @just --list

install:
    uv python install
    uv sync --extra dev

typecheck:
    uv run pyright

test:
    uv run pytest

lint:
    uv run ruff check src tests

format:
    uv run ruff format src tests

format-check:
    uv run ruff format --check src tests

check: lint format-check typecheck test

run:
    uv run python src/main.py