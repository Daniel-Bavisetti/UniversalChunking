"""Semantic visual enrichment: what a MODEL saw, never what we measured.

This is Pass 2. It is the only place in the tree that runs real vision inference,
and it obeys three rules without exception:

  1. It never raises. A missing package, a failed download, a corrupt model and an
     inference error all return an empty list plus a reason string. The caller
     carries on. Boundaries are computed independently of this module and are
     never revisited because of it.

  2. It never invents. `objects` come from a detector, `ocr_text` from an OCR
     engine, and `actions` from nothing at all here - action recognition is
     deferred, so `actions` is populated only by a VLM in providers.py and is
     empty in every offline run. Motion, edge density and scene cuts are
     MEASUREMENTS; turning one into "a person is typing" would be the exact
     dishonesty this project exists to avoid.

  3. It never hides a gap. "found nothing" and "never ran" are different facts and
     get different status strings, so an empty `objects` list can always be
     explained.

Inference runs on onnxruntime, already a hard dependency of faster-whisper, so
object detection adds no new package to requirements.txt. It runs on CPU
(~120ms/frame for yolov10n): the GPU is never touched, so there is no CUDA,
cuDNN or VRAM risk anywhere in the pipeline.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .config import (
    ENABLE_OBJECT_DETECTION,
    ENABLE_OCR,
    MODEL_DIR,
    OBJECT_CONFIDENCE,
    OBJECT_INPUT_SIZE,
    OBJECT_MODEL_FILE,
    OBJECT_MODEL_NAME,
    OBJECT_MODEL_REPO,
    OBSERVE_FRAMES_PER_MINUTE,
    OBSERVE_MAX_FRAMES,
    OBSERVE_MIN_FRAMES,
    OCR_EDGE_FLOOR,
    OCR_MAX_FRAMES,
    OCR_MIN_CHARS,
    OCR_MIN_CONFIDENCE,
)
from .media import grab_frames, select_observation_frames
from .schemas import FrameFeature, SceneCut, VisualObservation

# Status vocabulary. Distinguishing these is what stops an empty list from being
# readable as either "the model looked and saw nothing" or "no model ever ran".
NOT_REQUESTED = "not_requested"
UNAVAILABLE = "unavailable"


def _reason(exc: Exception, limit: int = 140) -> str:
    """A one-line failure reason. This is stamped into every unit's provenance,
    so a provider's multi-paragraph error would bloat the whole artifact."""
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _quiet(logger_name: str) -> None:
    """Drop below-WARNING records from a chatty third-party logger.

    A filter rather than setLevel: rapidocr re-sets its own level while
    constructing each of its three sub-models, so a level we set would be
    overwritten, while a filter persists. It logs a path per sub-model per
    video, which would bury the pipeline's own progress output.
    """
    logger = logging.getLogger(logger_name)
    if not any(getattr(f, "_vke", False) for f in logger.filters):
        drop = lambda record: record.levelno >= logging.WARNING  # noqa: E731
        drop._vke = True
        logger.addFilter(drop)


def _download(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, cache_dir=MODEL_DIR))


# --------------------------------------------------------------------------- #
# object detection - YOLOv10n via onnxruntime
# --------------------------------------------------------------------------- #
class ObjectDetector:
    """COCO object detection on single frames.

    YOLOv10 is end-to-end NMS-free: the model emits [1, 300, 6] rows of
    (x1, y1, x2, y2, score, class_id) already sorted by score. That removes the
    whole non-maximum-suppression step other YOLO exports need, which is most of
    why this class is short.
    """

    def __init__(self) -> None:
        self.model = OBJECT_MODEL_NAME
        self._session = None
        self._labels: dict[int, str] = {}
        self._input = "images"

    def load(self) -> str:
        """Return "" on success, or a human reason the detector is unavailable."""
        if self._session is not None:
            return ""
        try:
            import onnxruntime as ort

            weights = _download(OBJECT_MODEL_REPO, OBJECT_MODEL_FILE)
            config = _download(OBJECT_MODEL_REPO, "config.json")
            labels = json.loads(config.read_text(encoding="utf-8")).get("id2label", {})
            self._labels = {int(k): str(v) for k, v in labels.items()}
            self._session = ort.InferenceSession(
                str(weights), providers=["CPUExecutionProvider"])
            self._input = self._session.get_inputs()[0].name
            return ""
        except Exception as exc:
            return f"{UNAVAILABLE}: {_reason(exc)}"

    @staticmethod
    def _letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, float]:
        """Resize longest edge to `size`, pad bottom/right, scale to 0..1 RGB CHW.

        Padding only bottom/right keeps the origin at (0, 0), so un-scaling a box
        is a single divide with no offset to get wrong. Per the model's
        preprocessor_config.json there is no mean/std normalization, just 1/255.
        """
        h, w = bgr.shape[:2]
        scale = size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), scale

    def detect(self, bgr: np.ndarray, ts: float) -> list[VisualObservation]:
        if self._session is None:
            return []
        try:
            blob, scale = self._letterbox(bgr, OBJECT_INPUT_SIZE)
            rows = self._session.run(None, {self._input: blob})[0][0]
        except Exception as exc:
            print(f"[detect] object detection failed at {ts:.1f}s "
                  f"({type(exc).__name__}: {exc})")
            return []

        h, w = bgr.shape[:2]
        out: list[VisualObservation] = []
        for x1, y1, x2, y2, score, cls in rows:
            if float(score) < OBJECT_CONFIDENCE:
                continue
            label = self._labels.get(int(cls))
            if not label:
                continue
            out.append(VisualObservation(
                kind="object",
                value=label,
                source="object_detector",
                ts=round(float(ts), 3),
                model=self.model,
                confidence=round(float(score), 3),
                box=[round(float(x1) / scale / w, 4), round(float(y1) / scale / h, 4),
                     round(float(x2) / scale / w, 4), round(float(y2) / scale / h, 4)],
            ))
        return out


# --------------------------------------------------------------------------- #
# OCR - PP-OCR via onnxruntime (optional; absent means absent, never silent)
# --------------------------------------------------------------------------- #
class TextReader:
    """On-screen text. Optional: OCR is skipped entirely when nothing is installed.

    Two package generations return different shapes, so both are normalized here
    rather than pinning the demo to a single release line.
    """

    def __init__(self) -> None:
        self.model = ""
        self._engine = None

    @staticmethod
    def _version(package: str) -> str:
        """Name the engine by its installed version rather than by a guessed
        model family. rapidocr ships different PP-OCR generations across
        releases, and stamping a wrong model name into provenance would be its
        own small dishonesty."""
        try:
            from importlib.metadata import version

            return f"{package}-{version(package)}"
        except Exception:
            return package

    def load(self) -> str:
        if self._engine is not None:
            return ""
        for package, module in (("rapidocr", "rapidocr"),
                                ("rapidocr-onnxruntime", "rapidocr_onnxruntime")):
            try:
                engine_cls = importlib.import_module(module).RapidOCR
                _quiet("RapidOCR")
                self._engine, self.model = engine_cls(), self._version(package)
                return ""
            except Exception as exc:
                last = exc
        return f"{UNAVAILABLE}: rapidocr not installed ({type(last).__name__})"

    @staticmethod
    def _lines(raw) -> list[tuple[str, float | None]]:
        """Normalize either package's return shape to [(text, score)]."""
        if raw is None:
            return []

        # rapidocr 3.x: an object carrying parallel .txts / .scores sequences.
        txts = getattr(raw, "txts", None)
        if txts is not None:
            scores = list(getattr(raw, "scores", None) or [])
            return [(str(t), float(scores[i]) if i < len(scores) else None)
                    for i, t in enumerate(txts)]

        # rapidocr_onnxruntime 1.x: (results, elapse), results = [[box, text, score]].
        results = raw[0] if isinstance(raw, tuple) else raw
        out: list[tuple[str, float | None]] = []
        for row in results or []:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                out.append((str(row[1]), float(row[2])))
        return out

    def read(self, bgr: np.ndarray, ts: float) -> list[VisualObservation]:
        if self._engine is None:
            return []
        try:
            raw = self._engine(bgr)
        except Exception as exc:
            print(f"[detect] OCR failed at {ts:.1f}s ({type(exc).__name__}: {exc})")
            return []

        out: list[VisualObservation] = []
        for text, score in self._lines(raw):
            value = " ".join(str(text).split())
            # A single character is what OCR returns for a button edge or a UI
            # divider, not for text anybody put on screen.
            if len(value) < OCR_MIN_CHARS:
                continue
            if score is not None and score < OCR_MIN_CONFIDENCE:
                continue
            out.append(VisualObservation(
                kind="text",
                value=value,
                source="ocr",
                ts=round(float(ts), 3),
                model=self.model,
                confidence=round(score, 3) if score is not None else None,
            ))
        return out


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def frame_budget(duration: float) -> int:
    per_minute = duration / 60.0 * OBSERVE_FRAMES_PER_MINUTE
    return int(max(OBSERVE_MIN_FRAMES, min(OBSERVE_MAX_FRAMES, round(per_minute))))


def observe(
    video_path: Path,
    features: list[FrameFeature],
    cuts: list[SceneCut],
    duration: float,
) -> tuple[list[VisualObservation], dict[str, str]]:
    """Run the enrichment models over a bounded set of representative frames.

    Frames are chosen for the WHOLE video, not per unit, so one detection pass
    serves all three chunking configs. That keeps the headline comparison an
    honest ablation - identical visual evidence, only the boundaries differ - and
    costs a third of what per-config enrichment would.
    """
    status: dict[str, str] = {
        # Deferred outright (see the ROI table). A VLM may add actions later in
        # pipeline.py; nothing in this module ever will.
        "actions": f"{NOT_REQUESTED}: action recognition deferred, no VLM in this stage",
    }
    observations: list[VisualObservation] = []

    if not ENABLE_OBJECT_DETECTION and not ENABLE_OCR:
        status["objects"] = f"{NOT_REQUESTED}: disabled by config"
        status["ocr"] = f"{NOT_REQUESTED}: disabled by config"
        return [], status

    budget = frame_budget(duration)
    timestamps = select_observation_frames(features, cuts, duration, budget)
    frames = grab_frames(video_path, timestamps)
    if not frames:
        reason = f"{UNAVAILABLE}: no frames could be decoded"
        status["objects"] = status["ocr"] = reason
        return [], status

    edges = {round(f.ts, 3): f.edge_density for f in features}

    # --- objects ----------------------------------------------------------- #
    if ENABLE_OBJECT_DETECTION:
        detector = ObjectDetector()
        problem = detector.load()
        if problem:
            status["objects"] = problem
        else:
            found = 0
            for ts, frame in frames.items():
                hits = detector.detect(frame, ts)
                observations.extend(hits)
                found += len(hits)
            status["objects"] = (f"object_detector:{detector.model} "
                                 f"({found} detections over {len(frames)} frames)")
    else:
        status["objects"] = f"{NOT_REQUESTED}: disabled by config"

    # --- on-screen text ----------------------------------------------------- #
    if ENABLE_OCR:
        # Spend the OCR budget only where the cheap measurement says there is text
        # to read. A talking-head frame has almost no edges, so OCR there costs
        # most of a second to return nothing. Using the measurement we already
        # have to place the expensive model is the whole selectivity story.
        dense = [ts for ts in frames if edges.get(round(ts, 3), 0.0) >= OCR_EDGE_FLOOR]
        candidates = sorted(dense, key=lambda t: -edges.get(round(t, 3), 0.0))[:OCR_MAX_FRAMES]
        if not candidates:
            status["ocr"] = (f"{NOT_REQUESTED}: every frame was below the "
                             f"edge-density floor ({OCR_EDGE_FLOOR}); nothing on "
                             f"screen had enough structure to be text")
        else:
            reader = TextReader()
            problem = reader.load()
            if problem:
                status["ocr"] = problem
            else:
                found = 0
                for ts in sorted(candidates):
                    hits = reader.read(frames[ts], ts)
                    observations.extend(hits)
                    found += len(hits)
                status["ocr"] = (f"ocr:{reader.model} ({found} lines over "
                                 f"{len(candidates)} text-dense frames of {len(frames)})")
    else:
        status["ocr"] = f"{NOT_REQUESTED}: disabled by config"

    observations.sort(key=lambda o: (o.ts, o.kind, -(o.confidence or 0.0)))
    return observations, status


def warmup() -> dict[str, str]:
    """Pre-stage every model so the demo never downloads anything on stage."""
    out = {"objects": ObjectDetector().load() or "ready",
           "ocr": TextReader().load() or "ready"}
    for name, state in out.items():
        print(f"  {name:<10} {state}")
    return out
