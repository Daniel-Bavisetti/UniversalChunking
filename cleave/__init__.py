"""Cleave — universal chunking and information extraction.

Understand information before you cut it: profile the input, map the
relationships inside it, split along the grain, and spend AI only where
structure cannot supply the missing context.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load the project's .env at import time, before any module reads os.environ.
# Every entry point — the web app, `python -m cleave.evaluate`, the tests —
# goes through this package, so configuration lands in exactly one place.
# Real environment variables already set take precedence (override=False), so
# `CLEAVE_LLM=none uv run …` still wins over the file.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

__version__ = "0.1.0"
