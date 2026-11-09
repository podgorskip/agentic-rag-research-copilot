# Quality Assurance & Code Conventions

This document defines the coding standards and conventions that must be enforced in this repository.

## Logging

- **Always use the logging module**, never `print()` statements
- All scripts must call `setup_logging()` at the entry point (typically in `main()`)
- Import loggers from `agentic_rag_copilot.logging`:
  ```python
  from agentic_rag_copilot.logging import get_logger, setup_logging
  
  logger = get_logger(__name__)
  ```
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- All log messages follow the standard format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

## Documentation

- **Every function and method must have a docstring** that includes:
  - Description of what the function does
  - Arguments (Args section with parameter names, types, and descriptions)
  - Return value (Returns section with type and description)
  - Exceptions raised (Raises section if applicable)

Example:
```python
def ingest_source(uri: str, source_type: SourceType) -> IngestResponse:
    """Ingest a single source and upsert it into the vector store.

    Args:
        uri: The source URI (file path or URL).
        source_type: The type of source (PDF, HTML, etc.).

    Returns:
        An IngestResponse containing the document ID and chunks created.

    Raises:
        IngestionError: If the source cannot be processed.
    """
```

## Code Comments

- **Avoid comments in the code** — code should be self-documenting through clear naming
- Comments are only acceptable for:
  - Very complex logic that is not immediately obvious
  - Non-obvious workarounds or hacks for specific bugs
  - Subtle invariants or constraints that could surprise a reader
- Remove comments that just restate what the code does (e.g., `# increment counter` above `count += 1`)

## Constants

- File-level constants that are only used within a single file must:
  - Use a leading underscore prefix (`_CONSTANT_NAME`)
  - Be annotated with `Final[]` type hint from `typing`
  - Be defined at the top of the file after imports

Example:
```python
from typing import Final

_EXIT_OK: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1
_MAX_RETRIES: Final[int] = 3
```

- Public constants (used across files) should not have a leading underscore

## Just Commands

- **Every new command added to the justfile must have a description**
- Descriptions are added as comments on the line immediately before the command
- Keep descriptions concise (one short line)

Example:
```justfile
# Run all checks (lint, format-check, typecheck, test)
check: lint format-check typecheck test

# Install Python and project dependencies
install:
    uv python install
    uv sync --extra dev
```

## Summary

When contributing to this repository:

✅ **Do:**
- Use logging instead of print
- Document all functions with docstrings
- Use clear, self-documenting code
- Use `_CONSTANT_NAME: Final[type]` for file-local constants
- Add descriptions to all justfile commands

❌ **Don't:**
- Use print() for output
- Skip docstrings
- Write unnecessary comments
- Use bare constants without Final[] annotations (for file-local constants)
- Add undocumented commands to justfile
