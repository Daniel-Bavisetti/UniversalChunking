"""Semantic boundaries (stretch): embedding drift between adjacent elements.

Upgrades the flat-prose path from paragraph packing to topic-aware grouping —
same veto rules, same element-aligned cuts, just better-chosen groups. If the
model is unavailable the caller keeps paragraph_fallback; per the evidence
(NAACL 2025, arXiv:2410.13070) that is not much of a loss, which is exactly why
this is a stretch item and not core.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .models import ContentElement

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        log.warning("MiniLM unavailable (%s) — flat prose stays on paragraph_fallback", exc)
        return None


def available() -> bool:
    return _model() is not None


def embed(texts: list[str]):
    m = _model()
    if m is None:
        return None
    return m.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def semantic_groups(stream: list[ContentElement]) -> list[list[ContentElement]] | None:
    """Split a flat element stream where adjacent-element similarity dips below
    mean − 1σ. Returns None when the model is missing or the stream is tiny."""
    texts = [e.text for e in stream]
    if len(texts) < 4:
        return None
    vecs = embed(texts)
    if vecs is None:
        return None
    sims = [float(vecs[i] @ vecs[i + 1]) for i in range(len(vecs) - 1)]
    mean = sum(sims) / len(sims)
    var = sum((s - mean) ** 2 for s in sims) / len(sims)
    threshold = mean - var ** 0.5
    cuts = [i + 1 for i, s in enumerate(sims) if s < threshold]
    if not cuts:
        return [stream]
    groups, prev = [], 0
    for c in cuts:
        groups.append(stream[prev:c])
        prev = c
    groups.append(stream[prev:])
    return [g for g in groups if g]
