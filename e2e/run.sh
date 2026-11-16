#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

uv run python e2e_ingestion.py \
    data/paper-01.pdf data/paper-02.pdf data/paper-03.pdf \
    "https://en.wikipedia.org/wiki/Python_(programming_language)" \
    "https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)" \
    "https://en.wikipedia.org/wiki/ChatGPT" \
    --collection e2e_ingestion \
    --reingest \
    --queries-file queries.json \
    --output output/report.md
