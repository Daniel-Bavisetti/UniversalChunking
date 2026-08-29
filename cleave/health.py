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
    """The visual stack is three independent producers, so it degrades in
    stages rather than on/off: OCR and object detection can carry a figure on
    their own when no vision model is configured."""
    try:
        from .vision import available  # noqa: PLC0415

        state = available()
    except Exception as exc:
        return Check("vision", "Visual understanding", False, "unavailable",
                     f"visual stack failed to load ({type(exc).__name__}: {exc})")

    live = [name for name, key in (("OCR", "ocr"), ("objects", "objects"),
                                   ("vision model", "vision_model")) if state.get(key)]
    if not live:
        why = "; ".join(f"{k}: {v}" for k, v in state.get("reasons", {}).items())
        return Check("vision", "Visual understanding", False, "not_configured",
                     f"no visual producer available — figures keep caption text only"
                     + (f" ({why})" if why else ""))

    detail_bits = []
    if state.get("ocr"):
        detail_bits.append(f"OCR {state.get('ocr_model', '')}".strip())
    if state.get("objects"):
        detail_bits.append(f"objects {state.get('object_model', '')}".strip())
    if state.get("vision_model"):
        detail_bits.append(f"vision {state.get('vision_model_name', '')}".strip())
    missing = [k for k in ("ocr", "objects", "vision_model") if not state.get(k)]
    detail = " · ".join(detail_bits)
    if missing:
        detail += f" — {', '.join(missing)} unavailable"
    # Partial capability is real capability here, so it reports active with the
    # gap named rather than failing the whole check.
    return Check("vision", "Visual understanding", True, "active", detail)


def _check_video() -> Check:
    """The video engine is several parts; name whichever one is missing."""
    missing: list[str] = []
    for module, label in (("faster_whisper", "ASR (faster-whisper)"),
                          ("scenedetect", "scene detection"),
                          ("cv2", "frame decode (opencv)")):
        try:
            __import__(module)
        except Exception:
            missing.append(label)
    try:
        import vke.pipeline  # noqa: F401, PLC0415
    except Exception as exc:
        return Check("video", "Video engine", False, "unavailable",
                     f"vke did not import ({type(exc).__name__}: {exc})")
    if missing:
        return Check("video", "Video engine", False, "unavailable",
                     "missing " + ", ".join(missing) + " — install the 'video' extra")

    from .ingest_video import BOUNDARY_ENGINE  # noqa: PLC0415

    how = ("multimodal boundaries (speech · scene cuts · visual novelty · topic drift)"
           if BOUNDARY_ENGINE == "vke" else "speaker-turn boundaries via the shared chunker")
    return Check("video", "Video engine", True, "active", how)


def _check_web() -> Check:
    """Static extraction is the fast path; the browser is the escalation."""
    static = rendered = False
    try:
        import trafilatura  # noqa: F401, PLC0415

        static = True
    except Exception:
        pass
    try:
        import crawl4ai  # noqa: F401, PLC0415

        rendered = True
    except Exception:
        pass

    if not static and not rendered:
        return Check("web", "Web ingestion", False, "not_configured",
                     "no fetcher installed — install the 'web' extra for URLs")
    if static and not rendered:
        return Check("web", "Web ingestion", True, "active",
                     "trafilatura — static pages; JavaScript-rendered pages need "
                     "the 'web-rendered' extra")
    if rendered and not static:
        return Check("web", "Web ingestion", True, "active",
                     "crawl4ai — every page pays for a headless browser")
    return Check("web", "Web ingestion", True, "active",
                 "trafilatura for static pages, crawl4ai when a page needs rendering")


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
        _check_video(),
        _check_web(),
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
