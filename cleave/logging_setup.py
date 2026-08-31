"""Log configuration, installed however the app was started.

``logging.basicConfig`` used to live under ``if __name__ == "__main__"``, so it
ran for ``python -m cleave.app`` and for nothing else. The documented
alternative, ``uvicorn cleave.app:app``, therefore discarded every application
log line — including the ones explaining why a job failed. Configuration now
happens in the FastAPI lifespan, so both entry points get it.

Records also carry the job they belong to. A multi-file job logs ingest,
chunking and enrichment for several files through the same module loggers, and
without the id those lines interleave into something unattributable.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

FORMAT = "%(asctime)s %(levelname)-7s [%(job_id)s] %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

#: The job being processed on this task/thread. ``-`` outside a job.
current_job_id: ContextVar[str] = ContextVar("job_id", default="-")

_configured = False


class JobIdFilter(logging.Filter):
    """Stamp every record with the job it belongs to."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = current_job_id.get()
        return True


def configure_logging(level: str | None = None) -> None:
    """Install Cleave's format and level once, idempotently.

    Uvicorn installs its own root handler before the lifespan runs. Rather than
    fight it, the filter and format are applied to whatever handlers exist, and
    the ``cleave`` logger's level is set directly.
    """
    global _configured

    if level is None:
        from .config import settings  # noqa: PLC0415 — avoids an import cycle

        try:
            level = settings().log_level
        except Exception:  # pragma: no cover - a bad level must not stop startup
            level = "INFO"

    resolved = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()

    if not root.handlers:
        handler: logging.Handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(resolved)
    elif not _configured:
        # Uvicorn got here first: adopt its handlers, add our context.
        for handler in root.handlers:
            handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))

    for handler in root.handlers:
        if not any(isinstance(f, JobIdFilter) for f in handler.filters):
            handler.addFilter(JobIdFilter())

    logging.getLogger("cleave").setLevel(resolved)
    _configured = True
