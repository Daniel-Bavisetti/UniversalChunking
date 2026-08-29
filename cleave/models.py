"""Cleave data contract.

Plain dataclasses (not pydantic) so the pipeline stays importable without the
web stack; the server serializes via ``to_dict()``. This module is the frozen
contract shared with external modality workers (see CONTRACT.md) — change it
deliberately or not at all.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any


# ───────── enums ─────────

class Modality(str, Enum):
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class RelationType(str, Enum):
    """Only the types the MVP actually creates. Hierarchy is deliberately NOT
    an edge on units — it is denormalized once into ``Context.heading_path``."""

    CAPTIONS = "captions"
    CAPTIONED_BY = "captioned_by"
    REFERENCES = "references"
    NEXT = "next"
    PREVIOUS = "previous"
    SPOKEN_BY = "spoken_by"
    SCHEMA_OF = "schema_of"           # schema card → the row groups it describes
    HAS_SCHEMA = "has_schema"         # row group → its schema card
    # Meeting semantics (additive, contract-compatible: unknown types are
    # filtered on import, so old consumers simply ignore these).
    ANSWERS = "answers"               # answer unit → the question unit it resolves
    ANSWERED_BY = "answered_by"       # question unit → where its answer lives


# ───────── intermediate representation ─────────

#: Element kinds every modality normalizes into. Documents use the first row;
#: audio/video workers use the second.
ELEMENT_KINDS = (
    "heading", "paragraph", "list_item", "table", "figure", "caption", "code",
    "speech_segment", "visual_event", "other",
)


@dataclass(slots=True)
class ContentElement:
    """One normalized piece of source content — the universal intermediate
    representation. Kept intentionally thin: confidence/evidence live on
    relationships (where they earn their keep), not on every element."""

    id: str
    kind: str
    text: str
    level: int | None = None          # heading depth (1-based)
    parent_id: str | None = None      # hierarchy pointer (heading ancestry)
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    t0: float | None = None           # temporal span, seconds
    t1: float | None = None
    speaker: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. table grid rows


# ───────── knowledge unit parts ─────────

@dataclass(slots=True)
class Relationship:
    type: RelationType
    target_id: str
    confidence: float = 1.0           # 1.0 deterministic · 0.7-0.9 heuristic · ≤0.6 inferred
    evidence: str | None = None       # human-readable mechanism, e.g. bbox adjacency


@dataclass(slots=True)
class Provenance:
    source_uri: str
    source_sha256: str | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class Temporal:
    start_s: float
    end_s: float
    speaker: str | None = None


@dataclass(slots=True)
class Context:
    """What a reader needs to understand ``content`` standing alone."""

    document_title: str | None = None
    heading_path: list[str] = field(default_factory=list)
    situating_summary: str | None = None   # LLM-written; None until enrichment runs
    leading: str | None = None             # surrounding-text windows (optional)
    trailing: str | None = None
    tier: int = 0                          # 0 deterministic · 1 local model · 2 LLM


@dataclass(slots=True)
class ChunkingDecision:
    """Why this chunk exists — the receipt shown in the demo. Every unit is
    explainable: strategy, reason, the cuts we refused to make, and cost."""

    strategy: str                          # structural | hybrid | paragraph_fallback |
                                           # semantic | temporal | atomic
    reason: str
    signals: dict[str, float] = field(default_factory=dict)
    vetoed_cuts: list[str] = field(default_factory=list)
    severed_refs: int = 0
    escalation_flags: list[str] = field(default_factory=list)  # why it WOULD deserve an LLM
    llm_calls: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class KnowledgeUnit:
    id: str
    content: str
    modality: Modality
    context: Context
    provenance: Provenance
    decision: ChunkingDecision
    relationships: list[Relationship] = field(default_factory=list)
    temporal: Temporal | None = None
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def embed_text(self) -> str:
        """Exactly what a downstream system should embed — context first, so a
        chunk retrieved alone still knows where it lives.

        Surrounding prose is included because it is often the only thing that
        says what a table or figure is *for*; it stays out of `content` so a
        consumer can still display the chunk clean.

        Visual evidence is embedded for the same reason it is extracted: a
        video moment where a person is running, or a screen that reads "Deploy
        Production", must be findable by those words even though the transcript
        never says them. Dropping metadata evidence here would make everything
        the vision stack produced unretrievable — extraction that retrieval
        cannot see may as well not have run.
        """
        parts: list[str] = []
        if self.context.document_title:
            # The source's own name, so "what happened in <video X>" lands on
            # that video's units rather than everyone's.
            parts.append(self.context.document_title)
        if self.context.heading_path:
            parts.append(" > ".join(self.context.heading_path))
        if self.context.situating_summary:
            parts.append(self.context.situating_summary)
        if self.temporal and self.temporal.speaker:
            parts.append(f"speaker: {self.temporal.speaker}")
        if self.context.leading:
            parts.append(f"[before] {self.context.leading}")
        parts.append(self.content)
        if self.context.trailing:
            parts.append(f"[after] {self.context.trailing}")

        # ── visual + semantic evidence from metadata ──
        md = self.metadata or {}
        if md.get("visual_summary"):
            parts.append(f"[on screen] {md['visual_summary']}")
        ocr = md.get("ocr_text") or []
        if ocr and "Text on screen:" not in self.content and "Text in image:" not in self.content:
            parts.append("[text on screen] " + " · ".join(str(x) for x in ocr[:20]))
        objects = md.get("objects") or []
        if objects and "Visible:" not in self.content and "Objects detected:" not in self.content:
            parts.append("[visible] " + ", ".join(sorted({str(x) for x in objects})[:12]))
        if md.get("actions"):
            parts.append("[actions] " + ", ".join(str(x) for x in md["actions"][:8]))
        for sem in md.get("semantics", []):
            kind = str(sem.get("type", "")).replace("_", " ")
            if kind and kind != "statement":
                bits = [f"[{kind}]", str(sem.get("text", ""))[:160]]
                if sem.get("owner"):
                    bits.append(f"owner: {sem['owner']}")
                if sem.get("deadline"):
                    bits.append(f"due: {sem['deadline']}")
                parts.append(" ".join(bits))
        return "\n\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality"] = self.modality.value
        for rel in d["relationships"]:
            rel["type"] = rel["type"].value if isinstance(rel["type"], RelationType) else rel["type"]
            rel["confidence"] = round(rel["confidence"], 3)
        d["decision"]["cost_usd"] = round(d["decision"]["cost_usd"], 6)
        d["decision"]["signals"] = {k: round(v, 4) for k, v in d["decision"]["signals"].items()}
        if d["temporal"]:
            d["temporal"]["start_s"] = round(d["temporal"]["start_s"], 3)
            d["temporal"]["end_s"] = round(d["temporal"]["end_s"], 3)
        d["embed_text"] = self.embed_text()
        return d


# ───────── document profile ─────────

@dataclass(slots=True)
class Profile:
    """Cheap deterministic signals the router reads. Milliseconds, no models."""

    element_count: int = 0
    text_element_count: int = 0
    heading_count: int = 0
    heading_density: float = 0.0
    table_count: int = 0
    figure_count: int = 0
    caption_count: int = 0
    has_timestamps: bool = False
    is_tabular: bool = False           # the input IS a dataset, not prose containing tables
    row_count: int = 0
    column_count: int = 0
    total_tokens: int = 0
    route: str = ""                        # strategy chosen for the non-atomic remainder
    route_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["heading_density"] = round(self.heading_density, 4)
        return d


# ───────── shared helpers ─────────

@lru_cache(maxsize=1)
def _encoder():
    import tiktoken  # noqa: PLC0415 — lazy: keep import time sane

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoder().encode(text, disallowed_special=()))


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
