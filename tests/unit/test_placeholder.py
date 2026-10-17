"""Initial unit tests to verify package setup and imports."""

from agentic_rag_copilot import __version__


def test_package_version() -> None:
    """Verifies that the package version is properly exposed."""
    assert __version__ == "0.1.0"
