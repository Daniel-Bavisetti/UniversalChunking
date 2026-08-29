"""Video ingestion — the VKE engine, joined at the element seam.

The video work is a full pipeline of its own (``vke/``): ASR, scene detection,
diarization, object detection, OCR, and a boundary model that scores speech,
visual and topical signals together. It has its own store, its own evaluation
and its own notion of a knowledge unit.

None of that needs to be rebuilt here, and none of it is. The seam is exactly
the one CONTRACT.md already defines for external modality workers: VKE produces
timestamped, speaker-attributed, visually-annotated segments, and this module
normalizes them into ``ContentElement``s. From there the ordinary graph, router
and chunker take over, and the video routes ``temporal`` for the same reason a
podcast does — it carries timestamps.

Two integration levels, and which one runs is a real decision rather than a
preference:

  * **elements** (default) — VKE's extraction, Cleave's boundaries. The unit
    receipts, cut vetoes and enrichment all work exactly as they do for a PDF,
    and the two modalities are genuinely comparable because the same chunker
    made both.
  * **units** (``CLEAVE_VIDEO_BOUNDARIES=vke``) — VKE's own multimodal
    boundaries, imported whole. Its boundary model is better informed than the
    generic temporal chunker, because it can see scene cuts and visual novelty
    that never reach an element stream.

Speech and what was on screen while it was said travel together, so a unit can
answer "what did they say while the dashboard was showing the revenue panel" —
which is the thing a transcript-only chunker structurally cannot do.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .ingest_document import IngestResult
from .models import (
    ChunkingDecision,
    ContentElement,
    Context,
    KnowledgeUnit,
    Modality,
    Provenance,
    Temporal,
    count_tokens,
    sha256_of,
)

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

#: Which engine draws the boundaries.
#:
#: "vke" is the default, and the reason is measurable rather than aesthetic: the
#: generic temporal chunker cuts on speaker turns, so a single-presenter video
#: has no turns and collapses into one enormous unit — on the fixture, four
#: distinct topics (auth → sessions → migrations → deployment) became one
#: 105-second chunk. The video engine scores speech pauses, scene cuts, visual
#: novelty and topic drift together, and finds those four. Flattening to an
#: element stream throws away exactly the signals that make video chunkable.
#:
#: "cleave" is kept because it is the honest comparison: one chunker over every
#: modality, with directly comparable receipts.
BOUNDARY_ENGINE = os.environ.get("CLEAVE_VIDEO_BOUNDARIES", "vke").lower()

#: VKE produces three chunkings side by side for comparison. This is the one
#: that uses all the signals; the others exist as its baselines.
VKE_CONFIG = os.environ.get("CLEAVE_VKE_CONFIG", "vke_multimodal")


def _video_id(path: Path) -> str:
    """Stable per-file id, so VKE's expensive extraction cache actually hits."""
    return f"{path.stem[:32]}_{sha256_of(str(path))[:10]}"


def ingest_video(path: str | Path, progress=None) -> tuple[IngestResult | None,
                                                           list[KnowledgeUnit]]:
    """Run the video engine.

    → ``(ingest, [])`` when Cleave will chunk the elements, or
      ``(None, units)`` when VKE's own boundaries are imported whole.
    """
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"unsupported video type: {path.suffix}")

    from vke.pipeline import process_video  # noqa: PLC0415

    def report(stage: str, percent: int, message: str) -> None:
        if progress:
            progress(percent / 100.0, f"{message} ({path.name})")

    meta, units_by_config, _traces = process_video(
        path, _video_id(path), progress=report)
    vke_units = units_by_config.get(VKE_CONFIG) or next(iter(units_by_config.values()), [])
    log.info("video %s: %.0fs, %d unit(s) from config %r",
             path.name, meta.duration, len(vke_units), VKE_CONFIG)

    if BOUNDARY_ENGINE == "vke":
        units = [_as_cleave_unit(u, path) for u in vke_units]
        _annotate_unit_semantics(units)
        return None, units

    return _as_elements(vke_units, meta, path), []


# ───────── level 1: elements in, Cleave chunks ─────────

def _as_elements(vke_units, meta, path: Path) -> IngestResult:
    """Flatten VKE units back to a timed element stream.

    A VKE unit already merged several utterances; splitting it back apart would
    invent detail. So one unit becomes one ``speech_segment``, carrying what was
    on screen during it — Cleave's temporal chunker then groups by speaker and
    time exactly as it does for audio.
    """
    elements: list[ContentElement] = []
    warnings: list[str] = []

    for i, u in enumerate(vke_units):
        speakers = getattr(u, "speakers", []) or []
        visual_bits = [b for b in (getattr(u, "visual_context", ""),
                                   ", ".join(getattr(u, "objects", [])[:6]),
                                   " · ".join(getattr(u, "ocr_text", [])[:8])) if b]
        meta_out: dict = {}
        if visual_bits:
            meta_out["visual_summary"] = " · ".join(visual_bits)
        if getattr(u, "ocr_text", None):
            meta_out["ocr_text"] = list(u.ocr_text)
        if getattr(u, "objects", None):
            meta_out["objects"] = list(u.objects)
        if getattr(u, "observations", None):
            # Every visual claim keeps the model that made it and its score.
            meta_out["observations"] = [o.model_dump() for o in u.observations]
        if getattr(u, "keyframe_url", ""):
            meta_out["keyframe"] = u.keyframe_url

        text = (u.transcript or "").strip()
        if not text and visual_bits:
            # A stretch with no speech is still content when something happened
            # on screen; dropping it would lose the visual-only moments.
            text = visual_bits[0]
        if not text:
            continue

        elements.append(ContentElement(
            id=f"el_{i:04d}",
            kind="speech_segment",
            text=text,
            t0=float(u.span.start),
            t1=float(u.span.end),
            speaker=(speakers[0] if speakers else None),
            meta=meta_out,
        ))

    if not elements:
        warnings.append("the video produced no speech or visual content")

    from .cleaning import clean_elements  # noqa: PLC0415

    report = clean_elements(elements)
    elements = [e for e in elements if e.text]

    from .meeting import annotate_elements  # noqa: PLC0415

    annotate_elements(elements)

    return IngestResult(
        elements=elements,
        title=path.stem,
        source_uri=str(path),
        sha256=sha256_of(str(path)),
        warnings=warnings,
        cleaning=report.to_dict(),
    )


# ───────── level 2: VKE's own boundaries, imported whole ─────────

def _as_cleave_unit(u, path: Path) -> KnowledgeUnit:
    """Map one VKE KnowledgeUnit onto Cleave's.

    The boundary explanation survives the crossing: VKE scores each cut against
    speech, visual and topical signals, and that receipt is exactly what
    Cleave's ``ChunkingDecision`` is for. A unit that cannot say why it exists
    is the thing both engines refuse to emit.
    """
    # Each signal keeps both halves of its story: what was measured, and how
    # much it actually moved the boundary score once weighted.
    signals: dict[str, float] = {}
    for s in getattr(u.boundary, "signals", []):
        signals[s.name] = float(s.contribution)
        signals[f"{s.name}_raw"] = float(s.raw)
    if getattr(u.boundary, "score", None) is not None:
        signals["boundary_score"] = float(u.boundary.score)
    if getattr(u.boundary, "threshold", None) is not None:
        signals["threshold"] = float(u.boundary.threshold)
    content_parts = [u.transcript or ""]
    if getattr(u, "ocr_text", None):
        content_parts.append("Text on screen: " + " · ".join(u.ocr_text[:20]))
    if getattr(u, "objects", None):
        content_parts.append("Visible: " + ", ".join(sorted(set(u.objects))[:12]))
    content = "\n\n".join(p for p in content_parts if p.strip())

    return KnowledgeUnit(
        id=u.id,
        content=content,
        modality=Modality.VIDEO,
        context=Context(
            document_title=getattr(u, "title", "") or path.stem,
            situating_summary=getattr(u, "summary", "") or None,
            leading=getattr(u, "prev_summary", None),
            trailing=getattr(u, "next_summary", None),
            # A summary written by a model is tier 2 wherever it came from.
            tier=2 if getattr(u, "summary", "") else 0,
        ),
        provenance=Provenance(source_uri=str(path), source_sha256=sha256_of(str(path))),
        decision=ChunkingDecision(
            strategy="multimodal",
            reason=(getattr(u.boundary, "summary", "")
                    or "boundary scored on speech, visual and topical signals together"),
            signals=signals,
            escalation_flags=list(getattr(u, "flags", [])),
        ),
        temporal=Temporal(start_s=float(u.span.start), end_s=float(u.span.end),
                          speaker=(u.speakers[0] if getattr(u, "speakers", None) else None)),
        entities=list(getattr(u, "entities", []))[:8],
        metadata={
            "element_kind": "video_segment",
            "scene_ids": list(getattr(u, "scene_ids", [])),
            "keyframe": getattr(u, "keyframe_url", ""),
            "speakers": list(getattr(u, "speakers", [])),
            "ocr_text": list(getattr(u, "ocr_text", [])),
            "objects": list(getattr(u, "objects", [])),
            "visual_sources": list(getattr(u, "visual_sources", [])),
            "observations": [o.model_dump() for o in getattr(u, "observations", [])],
            "quality": getattr(u, "quality", None),
            "boundary_engine": "vke",
        },
        token_count=count_tokens(content),
    )


def _annotate_unit_semantics(units: list[KnowledgeUnit]) -> None:
    """Meeting semantics for VKE-boundary units.

    These units never pass through the element stream, so the per-utterance
    annotation cannot reach them. Sentences are classified instead; timestamps
    fall back to the unit's span, which is the finest anchor that survives
    VKE's merging.
    """
    import re as _re  # noqa: PLC0415

    from .meeting import classify_utterance  # noqa: PLC0415

    sent_split = _re.compile(r"(?<=[.!?])\s+")
    for u in units:
        found: list[dict] = []
        for sent in sent_split.split(u.content):
            sent = sent.strip()
            if len(sent) < 8:
                continue
            sem = classify_utterance(sent)
            if sem is None:
                continue
            sem.update({
                "text": sent[:280],
                "speaker": u.temporal.speaker if u.temporal else None,
                "timestamp_start": u.temporal.start_s if u.temporal else None,
                "timestamp_end": u.temporal.end_s if u.temporal else None,
                "ambiguous": sem["confidence"] < 0.75,
            })
            found.append(sem)
        if found:
            u.metadata["semantics"] = found
