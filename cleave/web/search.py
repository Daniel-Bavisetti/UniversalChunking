"""Retrieval over a job's units — the demo that proves the context-first design.

What is cached, and what is not, is deliberate. The vectors cost a model
inference to produce, so they are kept; the units are a local JSON read of a few
milliseconds, so they are re-read per query. The old cache held both, which meant
eight fully deserialized documents resident at once, and it was keyed by job id
with no invalidation — reprocessing a job served its old vectors forever.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict

from ..models import KnowledgeUnit  # noqa: F401 — documents what units.json holds
from .jobs import job_artifact

log = logging.getLogger(__name__)

#: job_id → (units.json mtime, vectors). Bounded: a long demo session would
#: otherwise hold every job's matrix for good. Vectors only — a 500-unit job at
#: 384 dims is well under a megabyte.
_VECTOR_CACHE: OrderedDict[str, tuple[float, object]] = OrderedDict()
_VECTOR_CACHE_MAX = 8
_CACHE_LOCK = threading.Lock()

TOP_K = 5


def clear_cache() -> None:
    """Forget every cached matrix. For tests."""
    with _CACHE_LOCK:
        _VECTOR_CACHE.clear()


def search_units(job_id: str, query: str) -> list[tuple[float, dict]]:
    """Top-``TOP_K`` units for ``query``, by cosine similarity of ``embed_text``.

    Blocking: it runs sentence-transformer inference. Callers on the event loop
    must hand it to a threadpool — this used to run inline in an ``async`` route
    and froze every other connection, including the status poller, for the
    duration of every search.
    """
    from ..semantic import embed  # noqa: PLC0415 — torch import stays lazy

    path = job_artifact(job_id, "units.json")
    if not path.exists():
        return []
    units = json.loads(path.read_text())
    if not units:
        return []
    mtime = path.stat().st_mtime

    with _CACHE_LOCK:
        cached = _VECTOR_CACHE.get(job_id)
        vecs = cached[1] if cached and cached[0] == mtime else None

    if vecs is None:
        vecs = embed([u["embed_text"] for u in units])
        if vecs is None:
            return []
        with _CACHE_LOCK:
            _VECTOR_CACHE[job_id] = (mtime, vecs)
            while len(_VECTOR_CACHE) > _VECTOR_CACHE_MAX:
                _VECTOR_CACHE.popitem(last=False)

    with _CACHE_LOCK:
        if job_id in _VECTOR_CACHE:
            _VECTOR_CACHE.move_to_end(job_id)

    qv = embed([query])[0]
    # One matrix-vector product rather than N Python-level dot products.
    scores = vecs @ qv
    order = scores.argsort()[::-1][:TOP_K]
    return [(float(scores[i]), units[i]) for i in order]


def embedding_available() -> bool:
    """Whether the MiniLM model can be loaded at all."""
    from ..semantic import available  # noqa: PLC0415

    return bool(available())
