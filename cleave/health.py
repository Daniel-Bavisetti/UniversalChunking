"""System status: what actually works right now, checked rather than assumed.

The failure this module exists to prevent is discovering in front of an
audience that the flagship feature quietly did nothing. Config presence is not
health: a key can be valid while the model it names is retired, which is
exactly how ``gemini-2.5-flash`` fails — the key authenticates, ``ListModels``
still advertises the model, and only a real call returns 404.

So the LLM check issues a real schema-constrained call through the same
``complete_json`` path enrichment uses. If the probe answers, the feature works;
if it does not, the UI says so before a job is ever run.

Probe cost is a few tokens and is deliberately NOT recorded in the usage ledger:
that ledger answers "what did processing this content cost", and a health check
is not processing content.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Health is cached rather than recomputed per page load — the LLM probe is a
#: network round trip, and the answer does not change second to second.
TTL_S = 300.0

_cache: dict[str, Any] = {"at": 0.0, "checks": None}

_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "string"}},
    "required": ["ok"],
}


@dataclass(slots=True)
class Check:
    key: str
    label: str
    ok: bool
    state: str            # active | unavailable | not_configured
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_parser() -> Check:
    try:
        import importlib.metadata as md  # noqa: PLC0415

        import docling  # noqa: F401, PLC0415

        try:
            version = md.version("docling")
        except md.PackageNotFoundError:
            version = "installed"
        return Check("parser", "Document parser", True, "active",
                     f"Docling {version} — PDF · DOCX · PPTX · XLSX · CSV · MD · HTML")
    except Exception as exc:
        return Check("parser", "Document parser", False, "unavailable",
                     f"Docling failed to import ({type(exc).__name__}) — no document ingestion")


def _check_embeddings() -> Check:
    try:
        from .semantic import available  # noqa: PLC0415

        if available():
            return Check("embeddings", "Embeddings", True, "active",
                         "all-MiniLM-L6-v2 — semantic boundaries and retrieval")
        return Check("embeddings", "Embeddings", False, "unavailable",
                     "MiniLM did not load — flat prose falls back to paragraph packing")
    except Exception as exc:
        return Check("embeddings", "Embeddings", False, "unavailable",
                     f"embedding model unavailable ({type(exc).__name__})")


def _check_retrieval(embeddings: Check) -> Check:
    if embeddings.ok:
        return Check("retrieval", "Retrieval", True, "active",
                     "cosine search over unit embed_text")
    return Check("retrieval", "Retrieval", False, "unavailable",
                 "needs the embedding model — search is disabled")


def _check_vision() -> Check:
    """Vision is not built yet. Reported honestly rather than omitted, so the
    panel shows the whole intended surface and its real state."""
    return Check("vision", "Vision model", False, "not_configured",
                 "not implemented — figures carry caption text only")


def _check_llm() -> Check:
    from .llm import get_provider  # noqa: PLC0415

    provider = get_provider()
    if provider.name == "none":
        return Check("llm", "LLM enrichment", False, "not_configured",
                     "no provider configured — deterministic mode, chunks stay at tier 0")

    text, usage = provider.complete_json(
        'Reply with {"ok":"yes"}.',
        system="You are a health probe. Reply with the JSON object only.",
        schema=_PROBE_SCHEMA,
    )
    if text:
        return Check("llm", "LLM enrichment", True, "active",
                     f"{provider.model} answered a live probe — situating summaries enabled")
    return Check("llm", "LLM enrichment", False, "unavailable",
                 f"{provider.model} is configured but did not answer — "
                 "check the key and that the model is still served")


def system_status(refresh: bool = False) -> list[dict[str, Any]]:
    """All subsystem checks, cached for ``TTL_S``."""
    now = time.time()
    if not refresh and _cache["checks"] is not None and now - _cache["at"] < TTL_S:
        return _cache["checks"]

    parser = _check_parser()
    embeddings = _check_embeddings()
    checks = [
        parser,
        embeddings,
        _check_retrieval(embeddings),
        _check_vision(),
        _check_llm(),
    ]
    out = [c.to_dict() for c in checks]
    _cache["at"], _cache["checks"] = now, out
    log.info("system status: %s", ", ".join(
        f"{c.key}={'ok' if c.ok else c.state}" for c in checks))
    return out


def enrichment_banner(checks: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """The one line that must never be ambiguous during a demo."""
    checks = checks if checks is not None else system_status()
    llm = next((c for c in checks if c["key"] == "llm"), None)
    if llm and llm["ok"]:
        return {"state": "active", "text": "LLM Enrichment: Active", "detail": llm["detail"]}
    detail = llm["detail"] if llm else "no provider"
    return {"state": "inactive",
            "text": "LLM Enrichment unavailable — deterministic mode",
            "detail": detail}
