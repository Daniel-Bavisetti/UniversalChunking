"""Video-specific extraction: metadata, frame features, keyframes, scene cuts.

One decode pass produces every visual measurement the boundary scorer needs, so
the expensive part of "look at the video" happens exactly once.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .config import (
    FRAME_SAMPLE_FPS,
    HSV_HIST_BINS,
    OBSERVE_MIN_GAP,
)
from .schemas import FrameFeature, SceneCut, VideoMeta


def probe(path: Path, video_id: str) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    duration = frames / fps if fps > 0 else 0.0
    return VideoMeta(
        video_id=video_id,
        filename=path.name,
        duration=round(duration, 3),
        fps=round(fps, 3),
        width=width,
        height=height,
        has_audio=_has_audio(path),
    )


def _has_audio(path: Path) -> bool:
    """PyAV ships with faster-whisper, so this costs us no extra dependency."""
    try:
        import av

        with av.open(str(path)) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return False


def _hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HSV_HIST_BINS),
                        [0, 180, 0, 256, 0, 256])
    hist = hist.flatten()
    total = hist.sum()
    return hist / total if total > 0 else hist


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels on an edge. High on slides, UI and text; low on faces."""
    edges = cv2.Canny(gray, 80, 200)
    return float(np.count_nonzero(edges)) / edges.size


def extract_frames(
    path: Path,
    out_dir: Path,
    sample_fps: float = FRAME_SAMPLE_FPS,
) -> list[FrameFeature]:
    """Single decode pass: sample frames and measure each one.

    Sequential reads with frame skipping rather than per-timestamp seeks, which
    is both faster and more accurate than CAP_PROP_POS_MSEC seeking.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    features: list[FrameFeature] = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        step = max(1, int(round(fps / sample_fps)))

        prev_gray: np.ndarray | None = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts = idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

                if prev_gray is None:
                    motion = 0.0
                else:
                    motion = float(np.abs(
                        small.astype(np.int16) - prev_gray.astype(np.int16)
                    ).mean()) / 255.0
                prev_gray = small

                features.append(FrameFeature(
                    ts=round(ts, 3),
                    hsv_hist=[float(v) for v in _hsv_hist(frame)],
                    edge_density=round(_edge_density(gray), 5),
                    motion=round(motion, 5),
                    brightness=round(float(gray.mean()) / 255.0, 5),
                ))
            idx += 1
    finally:
        cap.release()

    return features


def write_keyframe(frame: np.ndarray, out_file: Path, max_width: int = 480) -> bool:
    """Write an already-decoded frame as a display JPEG.

    Split out from `save_keyframe` so a caller holding frames from `grab_frames`
    can write all of them without reopening the video once per frame.
    """
    if frame is None or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)),
                           interpolation=cv2.INTER_AREA)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 82]))


def save_keyframe(path: Path, ts: float, out_file: Path, max_width: int = 480) -> bool:
    """Grab one frame at `ts` and write it as a JPEG."""
    frames = grab_frames(path, [ts])
    if not frames:
        return False
    return write_keyframe(next(iter(frames.values())), out_file, max_width)


def grab_frames(path: Path, timestamps: list[float]) -> dict[float, np.ndarray]:
    """Full-resolution BGR frames at the requested times. One capture, sorted seeks.

    Full resolution is the point. `save_keyframe` writes 480px JPEGs for display,
    and at that width the UI text in a 1080p screen recording is about four pixels
    tall - unreadable to OCR. Enrichment reads these arrays instead, in memory.

    Opening the capture once also fixes the keyframe stage, which previously
    opened and released one VideoCapture per unit per config.
    """
    if not timestamps:
        return {}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}

    out: dict[float, np.ndarray] = {}
    try:
        for ts in sorted(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(ts, 0.0) * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                out[ts] = frame
    finally:
        cap.release()
    return out


def select_observation_frames(
    features: list[FrameFeature],
    cuts: list[SceneCut],
    duration: float,
    budget: int,
    min_gap: float = OBSERVE_MIN_GAP,
) -> list[float]:
    """Pick <=budget frames that between them show everything the video shows.

    Uniform sampling wastes the budget on forty near-identical frames of one
    slide. Instead: seed with the frame just after each scene cut (a cut is the
    strongest evidence we have that the picture changed), then greedily take the
    frame LEAST similar to everything already chosen, by the same Bhattacharyya
    histogram distance the visual boundary signal uses.

    Ties break toward low motion, because a frame grabbed mid-pan is blurred and
    reads badly for both OCR and detection. Every input here was already measured
    in the single decode pass, so selection costs nothing extra.
    """
    if not features or budget <= 0:
        return []

    usable = [f for f in features if 0.0 <= f.ts <= max(duration, 0.0)]
    if not usable:
        return []

    def far_enough(ts: float, chosen: list[float]) -> bool:
        return all(abs(ts - c) >= min_gap for c in chosen)

    chosen: list[float] = []
    picked: list[FrameFeature] = []

    # Seed: just after each cut, where the picture is known to be new.
    for cut in cuts:
        target = cut.ts + 0.5
        nearest = min(usable, key=lambda f: abs(f.ts - target))
        if far_enough(nearest.ts, chosen) and len(chosen) < budget:
            chosen.append(nearest.ts)
            picked.append(nearest)

    # Always anchor the opening frame; a video with no cuts would otherwise start
    # its selection from wherever the novelty search happened to land.
    if len(chosen) < budget and far_enough(usable[0].ts, chosen):
        chosen.append(usable[0].ts)
        picked.append(usable[0])

    while len(chosen) < budget:
        best: tuple[tuple[float, float], FrameFeature] | None = None
        for f in usable:
            if not far_enough(f.ts, chosen):
                continue
            novelty = (min(hist_distance(f.hsv_hist, q.hsv_hist) for q in picked)
                       if picked else 1.0)
            # Time to the nearest frame we already hold. Novelty is rounded hard
            # so that near-identical frames TIE, and the temporal gap then breaks
            # the tie. Without this, a long static shot wins every round on
            # floating-point noise and swallows the budget: eight frames of one
            # slide, and whole minutes of the video never looked at.
            gap = min((abs(f.ts - c) for c in chosen), default=duration)
            key = (round(novelty, 2), gap)
            if best is None or key > best[0]:
                best = (key, f)
        if best is None:
            break  # every remaining frame is inside min_gap of one we already have
        chosen.append(best[1].ts)
        picked.append(best[1])

    return sorted(chosen)


def detect_scenes(path: Path) -> list[SceneCut]:
    """PySceneDetect content cuts. Never fatal - no cuts is a valid video."""
    try:
        from scenedetect import ContentDetector, detect

        scenes = detect(str(path), ContentDetector())
    except Exception:
        return []

    cuts: list[SceneCut] = []
    for start, _end in scenes[1:]:  # the first scene always starts at 0.0
        cuts.append(SceneCut(ts=round(float(start.seconds), 3)))
    return cuts


def describe_visual(features: list[FrameFeature]) -> str:
    """An honest, measured description. No model, so no invented semantics.

    Offline mode must never fabricate a scene description; it reports what was
    actually measured and lets the keyframe image do the persuading.
    """
    if not features:
        return "no visual data"

    edge = float(np.mean([f.edge_density for f in features]))
    motion = float(np.mean([f.motion for f in features]))
    bright = float(np.mean([f.brightness for f in features]))

    text_level = "high" if edge > 0.06 else "moderate" if edge > 0.025 else "low"
    motion_level = "static" if motion < 0.01 else "some motion" if motion < 0.04 else "high motion"
    light = "dark" if bright < 0.35 else "bright" if bright > 0.65 else "mid-tone"

    return (f"{text_level} on-screen text density, {motion_level}, {light} frame "
            f"(edges {edge:.3f}, motion {motion:.3f}, brightness {bright:.2f})")


def hist_distance(a: list[float], b: list[float]) -> float:
    """Bhattacharyya distance in 0..1. Symmetric and bounded, unlike chi-square."""
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if va.size == 0 or vb.size == 0:
        return 0.0
    bc = float(np.sum(np.sqrt(np.clip(va * vb, 0.0, None))))
    return float(math.sqrt(max(0.0, 1.0 - bc)))
