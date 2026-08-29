"""Configuration and the chunking weight vectors.

The three configs are the experiment (plan sec.18): A, B and C run the *same*
code path and differ only by a weight vector. B->C isolates the visual
contribution exactly, which is the proof that extraction drives chunking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PIPELINE_VERSION = "0.1.0"
CHUNKER_VERSION = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORE_DIR = DATA_DIR / "store"
UPLOAD_DIR = DATA_DIR / "uploads"

# --- extraction -------------------------------------------------------------
FRAME_SAMPLE_FPS = 2.0        # frames per second in the single decode pass
HSV_HIST_BINS = (8, 8, 4)     # H, S, V -> 256-dim histogram
ASR_MODEL = "base"            # 10x realtime on CPU int8, verified in the spike
ASR_COMPUTE = "int8"

# --- boundary detection -----------------------------------------------------
# TextTiling block size, expressed in SECONDS OF SPEECH rather than a token
# count. A block must span roughly the length of topic you want to resolve: too
# short and every sentence boundary reads as a topic shift; too long and real
# shifts get averaged away. Seconds keep this scale-invariant across videos of
# very different speech density.
SEMANTIC_BLOCK_SECONDS = 25.0
SILENCE_FULL_SCORE = 2.0      # a pause this long scores 1.0
SCENE_KERNEL_SIGMA = 1.5      # seconds; how far a scene cut's influence spreads
SPEAKER_KERNEL_SIGMA = 1.0    # seconds; how far a speaker change's influence spreads
ENABLE_DIARIZATION = True     # heuristic; contributes nothing on single-speaker audio

MIN_EVENT_SECONDS = 15.0      # non-maximum suppression radius / hard floor
MAX_EVENT_SECONDS = 180.0     # force a split beyond this
SNAP_WINDOW = 2.0             # snap a boundary to an utterance edge within +-this
THRESHOLD_K = 1.0             # tau = mean(s) + k * std(s)

# --- semantic enrichment (Pass 2) -------------------------------------------
# ENRICHMENT ONLY. Nothing below this line may reach signals.py or influence a
# boundary: model output exists only near candidates the score already found, so
# a score that depended on it would be circular (docs/ARCHITECTURE.md sec.5).
# Every knob here can be off, and every model here can be missing, without
# changing a single boundary timestamp.


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() not in ("0", "false", "no", "")


ENABLE_OBJECT_DETECTION = _flag("VKE_ENABLE_OBJECT_DETECTION", True)
ENABLE_OCR = _flag("VKE_ENABLE_OCR", True)

# Frame selection. We observe a bounded set of representative frames for the
# WHOLE video, not per unit: the detections are then shared by all three configs,
# so the comparison stays an honest ablation (identical visual evidence, only the
# boundaries differ) and we pay for one detection pass instead of three.
OBSERVE_FRAMES_PER_MINUTE = 4.0
OBSERVE_MIN_FRAMES = 8
OBSERVE_MAX_FRAMES = 40
OBSERVE_MIN_GAP = 2.0          # seconds; keeps the budget off near-duplicate frames

# Object detection. YOLOv10 is NMS-free (output [1,300,6], pre-sorted), which is
# why postprocessing is a loop and not a suppression algorithm.
OBJECT_MODEL_REPO = os.getenv("VKE_OBJECT_MODEL_REPO", "onnx-community/yolov10n")
OBJECT_MODEL_FILE = "onnx/model.onnx"
OBJECT_MODEL_NAME = "yolov10n-onnx"
OBJECT_INPUT_SIZE = 640
OBJECT_CONFIDENCE = 0.35

# OCR. Gated on measured edge density: a talking-head frame has almost no edges,
# so running OCR there costs ~0.7s to return nothing. Using the cheap measurement
# to decide where to spend the expensive one is the whole selectivity story.
# An ABSOLUTE threshold does not survive contact with real footage: a clean UI
# with a few lines of large text measures ~0.007 here, while a photograph of a
# street measures ~0.08. A fixed cut would skip the screenshot and OCR the
# photo - precisely backwards. So the floor only rejects near-featureless frames
# and the real selection is RELATIVE: the most edge-dense frames of this video,
# which is the same video-relative philosophy as signals.robust_normalize.
OCR_EDGE_FLOOR = 0.002
OCR_MAX_FRAMES = 24
OCR_MIN_CONFIDENCE = 0.5
OCR_MIN_CHARS = 2              # a 1-char hit is an artifact of an edge, not text

# Where downloaded ONNX weights live. Set VKE_MODEL_DIR to pre-stage them and
# guarantee the demo never downloads anything on stage.
MODEL_DIR = os.getenv("VKE_MODEL_DIR") or None


# --- refinement pass --------------------------------------------------------
ENABLE_REFINE = True
MERGE_MAX_SECONDS = 22.0      # only short neighbours are merge candidates
MERGE_COHESION = 0.30         # ... and only if they share this much vocabulary
SPLIT_MIN_SECONDS = 150.0     # only long units are split candidates
SPLIT_COHESION = 0.12         # ... and only if their halves share less than this


@dataclass(frozen=True)
class ChunkConfig:
    """A named chunking strategy. `weights` drives the boundary scorer."""

    key: str
    label: str
    description: str
    weights: dict[str, float] = field(default_factory=dict)
    fixed_window: float | None = None  # set => ignore signals, cut every N seconds

    @property
    def is_fixed(self) -> bool:
        return self.fixed_window is not None


CONFIG_A = ChunkConfig(
    key="fixed_30s",
    label="Fixed 30s",
    description=(
        "The VideoRAG baseline, reimplemented: fixed 30-second windows packed to a "
        "token budget. No extraction influences the boundaries at all."
    ),
    fixed_window=30.0,
)

# B and C are IDENTICAL except for the visual weight. That is what makes the
# comparison a controlled ablation: any boundary that appears in C but not B is
# attributable to vision and nothing else. (An earlier version also varied the
# semantic and silence weights between the two, which quietly invalidated that
# claim - three variables changed, so no single one could be credited.)
_SHARED = {"semantic": 0.40, "silence": 0.20, "speaker": 0.15}

CONFIG_B = ChunkConfig(
    key="audio_only",
    label="Audio-only",
    description=(
        "Everything VKE does except look at the picture: lexical topic shift, "
        "speech pauses and speaker changes. The visual weight is zero."
    ),
    weights={**_SHARED, "visual": 0.00},
)

CONFIG_C = ChunkConfig(
    key="vke_multimodal",
    label="VKE Multimodal",
    description=(
        "Speech, vision and timing fused into one boundary score. The ONLY "
        "difference from Audio-only is that the visual weight is non-zero."
    ),
    weights={**_SHARED, "visual": 0.40},
)

CONFIGS: dict[str, ChunkConfig] = {c.key: c for c in (CONFIG_A, CONFIG_B, CONFIG_C)}
DEFAULT_CONFIG = CONFIG_C.key
SIGNAL_NAMES = ("semantic", "visual", "silence", "speaker")


def ensure_dirs() -> None:
    for d in (DATA_DIR, STORE_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)
