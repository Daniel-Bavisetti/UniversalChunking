"""Paths and the Jinja environment, in one place.

Its own module so routers can render without importing ``cleave.app`` — which
imports them — and creating a cycle.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = ROOT / "cleave" / "static"
TEMPLATE_DIR = ROOT / "cleave" / "templates"

templates = Jinja2Templates(directory=TEMPLATE_DIR)
