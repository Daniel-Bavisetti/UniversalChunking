"""Document Worker: wraps ingest_document for distributed worker dispatch."""

from __future__ import annotations

from pathlib import Path

from ..ingest_document import IngestResult, ingest_document


def process_document_file(path: str | Path) -> IngestResult:
    """Process a document file via Docling/openpyxl."""
    return ingest_document(path)
