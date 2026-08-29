"""Video Knowledge Units -> Universal Knowledge Units.

VKE is the video arm of a future multi-modality platform (PDF, DOCX, slides, web,
audio). This is the whole integration investment: a pure mapping function, not a
parallel type hierarchy. A schema is only load-bearing once something consumes
it, and nothing will until a second modality exists - but writing the mapping now
proves it actually works, and it answers the "Universal" in the problem statement.

Two decisions carry all the future compatibility, and both are free at export time:

  * `evidence` is a LIST of typed Locators, not a bare time span. A PDF chunk may
    span two pages; a video unit happens to need one locator.
  * `metadata` is an open dict, so video puts scene_ids/keyframe there and a PDF
    puts page_count/font_stats without any schema churn.

A future PDF pipeline supplies its own extraction and its own boundary signals,
emits Locator(kind="page_region", ...), and reuses everything downstream.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .config import PIPELINE_VERSION
from .schemas import KnowledgeUnit

UNIVERSAL_SCHEMA_VERSION = "0.1.0"


class Locator(BaseModel):
    """Where a unit came from inside its source. The modality-portable idea.

    kind          ref
    ------------  --------------------------------------------------
    time_span     {"start": 25.4, "end": 52.4}          (video, audio)
    page_region   {"page": 3, "bbox": [x0,y0,x1,y1]}    (pdf, slides)
    char_range    {"start": 1204, "end": 1890}          (docx, html)
    """

    kind: str
    ref: dict[str, Any]
    preview: str | None = None


class UniversalKnowledgeUnit(BaseModel):
    id: str
    source: dict[str, Any]
    content: dict[str, Any]
    context: dict[str, Any]
    evidence: list[Locator]
    relationships: dict[str, Any]
    metadata: dict[str, Any]
    confidence: float
    provenance: dict[str, Any]


def to_universal(unit: KnowledgeUnit, source_reference: str = "") -> UniversalKnowledgeUnit:
    """Pure mapping. No I/O, trivially testable against a fixture."""
    return UniversalKnowledgeUnit(
        id=unit.id,
        source={
            "source_id": unit.video_id,
            "source_type": "video",
            "source_reference": source_reference or f"video://{unit.video_id}",
        },
        content={
            "primary": unit.transcript,
            "structured": {
                "title": unit.title,
                "summary": unit.summary,
                "visual_context": unit.visual_context,
                "entities": unit.entities,
                # Visual evidence is content, not decoration: a unit whose screen
                # read "Deploy Production" is findable by that phrase even though
                # nobody ever said it. Dropping these on export would throw away
                # the one thing a transcript-only chunker cannot produce.
                "objects": unit.objects,
                "ocr_text": unit.ocr_text,
                "actions": unit.actions,
                "speakers": unit.speakers,
            },
        },
        context={
            "preceding": unit.prev_summary,
            "following": unit.next_summary,
            "carried_entities": unit.carried_entities,
        },
        evidence=[
            Locator(
                kind="time_span",
                ref={"start": unit.span.start, "end": unit.span.end},
                preview=unit.keyframe_url or None,
            )
        ],
        relationships={
            "previous": unit.prev_unit_id,
            "next": unit.next_unit_id,
            "related": unit.related_unit_ids,
        },
        metadata={
            "modality": "video",
            "duration_seconds": unit.span.duration,
            "scene_ids": unit.scene_ids,
            "keyframe": unit.keyframe_url,
            "chunking_config": unit.config,
            # Boundary explanation generalizes across modalities once signal names
            # are free-form, so it travels with the unit rather than being dropped.
            "boundary": {
                "ts": unit.boundary.ts,
                "score": unit.boundary.score,
                "threshold": unit.boundary.threshold,
                "snapped_from": unit.boundary.snapped_from,
                "signals": [s.model_dump() for s in unit.boundary.signals],
                "summary": unit.boundary.summary,
            },
            "quality": unit.quality_parts,
            "flags": unit.flags,
            # Provenance for every visual claim above: which model said it, when,
            # and how confident it was (None where the producer ships no score).
            "visual_source": unit.visual_source,
            "visual_sources": unit.visual_sources,
            "observations": [o.model_dump() for o in unit.observations],
        },
        confidence=unit.quality if unit.quality is not None else 0.0,
        provenance={
            **unit.provenance,
            "schema": "UniversalKnowledgeUnit",
            "schema_version": UNIVERSAL_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
        },
    )


def export_units(
    units: list[KnowledgeUnit], source_reference: str = ""
) -> list[dict[str, Any]]:
    return [to_universal(u, source_reference).model_dump() for u in units]


def export_jsonl(units: list[KnowledgeUnit], source_reference: str = "") -> str:
    import json

    return "\n".join(
        json.dumps(to_universal(u, source_reference).model_dump(), ensure_ascii=False)
        for u in units
    )
