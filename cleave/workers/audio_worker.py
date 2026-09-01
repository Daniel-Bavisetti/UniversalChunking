"""Audio Worker: wraps ingest_audio for distributed worker dispatch."""

from __future__ import annotations

from pathlib import Path

from ..ingest_audio import ingest_audio
from ..ingest_document import IngestResult


def process_audio_file(path: str | Path) -> IngestResult:
    """Process an audio file via the STT ingestion pipeline."""
    return ingest_audio(path)
