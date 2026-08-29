"""The VKE data contract.

`KnowledgeUnit` is the stable interface (plan sec.20). Modules upstream of
`chunker.py` may assume video freely; everything downstream sees only the types
in this file. Add fields here, but do not rename or remove them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, computed_field

SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
class Span(BaseModel):
    """An absolute time interval in seconds from the start of the video.

    Absolute is the whole point: the VideoRAG baseline stores segment times in a
    filename and recovers them with eval(), and never converts its per-window ASR
    timestamps to video time. Every timestamp in VKE is absolute.
    """

    start: float
    end: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


# --------------------------------------------------------------------------- #
# extraction outputs (video-specific, consumed by signals + chunker)
# --------------------------------------------------------------------------- #
class Word(BaseModel):
    text: str
    start: float
    end: float


class Utterance(BaseModel):
    """One ASR segment with absolute timestamps."""

    id: str
    span: Span
    text: str
    confidence: float = 1.0
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)


class FrameFeature(BaseModel):
    """Cheap per-sample visual measurements from the single decode pass."""

    ts: float
    hsv_hist: list[float]      # normalized, used for the visual-shift signal
    edge_density: float        # proxy for on-screen text / UI density
    motion: float              # mean abs difference against the previous sample
    brightness: float


class SceneCut(BaseModel):
    ts: float
    confidence: float = 1.0


class SpeakerTurn(BaseModel):
    """A contiguous stretch attributed to one speaker.

    Heuristic, not a neural diarizer: speaker *changes* are reasonably reliable,
    speaker *identity* is not, which is why ids are opaque (speaker_00) and a
    confidence travels with every turn.
    """

    span: Span
    speaker: str
    confidence: float = 0.5
    method: str = "energy_centroid"


class VisionResult(BaseModel):
    """What a vision provider saw in one keyframe.

    `source` records how it was produced so an offline heuristic can never be
    mistaken for a model's description.
    """

    description: str = ""
    ocr_text: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    source: str = "heuristic"  # heuristic | vlm | failed | budget_exceeded


class VisualObservation(BaseModel):
    """One thing a MODEL inferred from one frame, with the provenance to defend it.

    An observation exists only when real inference produced it, which is why
    `source` has no "heuristic" member. Measured signals (edge density, motion,
    brightness) stay in `FrameFeature` and `visual_context` and never become
    observations: a pixel-difference measurement is not a semantic claim, and
    dressing one up as an action is the single failure this project exists to
    avoid.

    `confidence` is None rather than a defaulted 1.0. A detector and an OCR line
    ship real scores; a VLM ships none. Inventing 1.0 for an unscored producer
    would make a guess indistinguishable from a measured 0.95.
    """

    kind: str                        # object | text | action | description
    value: str
    source: str                      # object_detector | ocr | vlm
    ts: float                        # absolute video time of the frame it came from
    model: str                       # yolov10n-onnx | PP-OCRv4 | gpt-4o-mini
    confidence: float | None = None  # the producer's own score, or None if it has none
    box: list[float] | None = None   # x1,y1,x2,y2 normalized 0..1 of the frame


# --------------------------------------------------------------------------- #
# boundaries - the innovation, made inspectable
# --------------------------------------------------------------------------- #
class SignalContribution(BaseModel):
    """One signal's contribution to a boundary score.

    `name` is deliberately `str`, not a Literal: a future PDF chunker emits
    "heading" / "font_change" / "whitespace_gap" through this same type
    (plan sec.20). Free now, expensive to retrofit.
    """

    name: str
    raw: float           # the measured value, in the signal's own units
    normalized: float    # 0..1 within this video
    weight: float
    contribution: float  # normalized * weight


class BoundaryExplanation(BaseModel):
    """Why a chunk starts where it does. Rendered verbatim in the UI."""

    ts: float
    score: float
    threshold: float
    signals: list[SignalContribution] = Field(default_factory=list)
    snapped_from: float | None = None  # pre-utterance-snap position
    summary: str = ""

    @property
    def dominant_signal(self) -> str | None:
        if not self.signals:
            return None
        return max(self.signals, key=lambda s: s.contribution).name


# --------------------------------------------------------------------------- #
# the product artifact
# --------------------------------------------------------------------------- #
class KnowledgeUnit(BaseModel):
    """A temporally-grounded, context-preserving unit of video knowledge.

    In the MVP an Event and a KnowledgeUnit are 1:1; the distinction earns its
    keep only once merge/split exists, so we keep one type (plan sec.6).
    """

    # --- P1 ---
    id: str
    video_id: str
    span: Span
    title: str
    transcript: str
    visual_context: str = ""
    scene_ids: list[int] = Field(default_factory=list)
    keyframe_url: str = ""
    boundary: BoundaryExplanation
    prev_unit_id: str | None = None
    next_unit_id: str | None = None
    config: str = "vke_multimodal"  # fixed_30s | audio_only | vke_multimodal

    # --- P2 ---
    entities: list[str] = Field(default_factory=list)
    summary: str = ""
    prev_summary: str | None = None
    next_summary: str | None = None
    carried_entities: list[str] = Field(default_factory=list)
    # --- P5 ---
    speakers: list[str] = Field(default_factory=list)
    ocr_text: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    # Provenance of `visual_context` ALONE: heuristic | vlm | failed | not_requested.
    visual_source: str = "heuristic"
    # Every producer that contributed an observation, e.g. ["object_detector", "ocr"].
    visual_sources: list[str] = Field(default_factory=list)
    # The evidence behind objects/ocr_text/actions: those three lists are pure
    # projections of this one (see chunker.attach_observations), so nothing can
    # populate them without leaving a source, a timestamp and a model behind.
    observations: list[VisualObservation] = Field(default_factory=list)

    quality: float | None = None
    quality_parts: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    related_unit_ids: list[str] = Field(default_factory=list)

    # --- P4 ---
    provenance: dict[str, Any] = Field(default_factory=dict)

    def to_embedding_text(self) -> str:
        """Everything a retriever should be able to match on.

        ocr_text and objects are included so a query can reach a moment by what
        was SHOWN rather than by what was said - the one thing transcript-only
        chunking structurally cannot do.
        """
        parts = [self.title, self.transcript, self.visual_context,
                 " ".join(self.entities),
                 " ".join(self.ocr_text),
                 " ".join(self.objects)]
        return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# retrieval (P4)
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    unit_id: str
    video_id: str
    span: Span
    score: float
    reason: str
    snippet: str


# --------------------------------------------------------------------------- #
# processing state
# --------------------------------------------------------------------------- #
class StageTrace(BaseModel):
    stage: str
    seconds: float
    detail: dict[str, Any] = Field(default_factory=dict)


class VideoMeta(BaseModel):
    video_id: str
    filename: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool = True


class Job(BaseModel):
    job_id: str
    video_id: str
    status: str = "queued"     # queued | running | done | error
    stage: str = ""
    percent: int = 0
    message: str = ""
    error: str | None = None
    traces: list[StageTrace] = Field(default_factory=list)
