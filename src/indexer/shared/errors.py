"""Shared exception types for the indexing pipeline."""


class IndexerError(Exception):
    """Base exception for indexing pipeline failures."""


class InputValidationError(IndexerError):
    """Raised when the configured inputs are invalid."""


class DependencyUnavailableError(IndexerError):
    """Raised when an optional runtime dependency is unavailable."""


class IndexingRuntimeError(IndexerError):
    """Raised when indexing fails after input validation succeeds."""
