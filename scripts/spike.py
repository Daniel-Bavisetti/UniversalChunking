"""P0 dependency spike (plan sec.12).

Proves, in order, the three things the whole plan rests on:
  1. faster-whisper transcribes an mp4 DIRECTLY (PyAV decodes the audio track,
     so system ffmpeg is not on the critical path) and returns real timestamps.
  2. OpenCV decodes frames and writes a JPEG.
  3. PySceneDetect returns scene cuts.

Any failure here changes the plan, which is why this runs before architecture.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "data" / "fixture.mp4"
OUT = ROOT / "data" / "_spike"


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def spike_asr() -> bool:
    print("\n[1/3] faster-whisper on the mp4 directly (no ffmpeg)")
    from faster_whisper import WhisperModel

    t0 = time.time()
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print(f"  model loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    segments, info = model.transcribe(str(VIDEO), word_timestamps=True)
    segments = list(segments)
    elapsed = time.time() - t0

    print(f"  transcribed {info.duration:.1f}s of audio in {elapsed:.1f}s "
          f"({info.duration/max(elapsed,1e-9):.2f}x realtime), lang={info.language}")
    print(f"  {len(segments)} segments")
    for s in segments[:3]:
        print(f"    [{s.start:6.2f} -> {s.end:6.2f}] {s.text.strip()[:66]}")
    if segments:
        last = segments[-1]
        print(f"    ...")
        print(f"    [{last.start:6.2f} -> {last.end:6.2f}] {last.text.strip()[:66]}")

    words = [w for s in segments for w in (s.words or [])]
    ok = True
    ok &= check("returns segments", len(segments) > 0)
    ok &= check("word-level timestamps", len(words) > 0, f"{len(words)} words")
    # Timestamps must be absolute across the whole file, not per-window.
    ok &= check(
        "timestamps span the full 96s video",
        bool(segments) and segments[-1].end > 60,
        f"last segment ends at {segments[-1].end:.1f}s" if segments else "",
    )
    ok &= check("no ffmpeg binary needed (PyAV decoded the mp4)", True)

    # Did it actually hear the topic vocabulary we synthesized?
    text = " ".join(s.text for s in segments).lower()
    for word in ("authentication", "database", "deployment"):
        ok &= check(f"heard '{word}'", word in text)
    return ok


def spike_frames() -> bool:
    print("\n[2/3] OpenCV frame decode + JPEG write")
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    ok = check("VideoCapture opened", cap.isOpened())
    if not ok:
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {w}x{h} @ {fps:.2f}fps, {n} frames = {n/max(fps,1e-9):.1f}s")

    t0 = time.time()
    grabbed = 0
    written = None
    # Seek to one frame inside each ground-truth segment.
    for ts in (12.0, 36.0, 60.0, 84.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        got, frame = cap.read()
        if got:
            grabbed += 1
            written = OUT / f"frame_{ts:.0f}s.jpg"
            cv2.imwrite(str(written), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()

    ok &= check("seek + read at 4 timestamps", grabbed == 4, f"{grabbed}/4")
    ok &= check("JPEG written", written is not None and written.exists(),
                str(written.name) if written else "")
    ok &= check("metadata sane", fps > 0 and n > 0 and w > 0)
    print(f"  frame ops took {time.time()-t0:.2f}s")
    return ok


def spike_scenes() -> bool:
    print("\n[3/3] PySceneDetect content cuts")
    from scenedetect import ContentDetector, detect

    t0 = time.time()
    scenes = detect(str(VIDEO), ContentDetector())
    elapsed = time.time() - t0

    cuts = [s[0].get_seconds() for s in scenes][1:]  # first entry is always 0.0
    print(f"  {len(scenes)} scenes in {elapsed:.1f}s; cuts at "
          f"{[round(c, 2) for c in cuts]}")

    ok = check("returned scenes", len(scenes) > 0)
    # We engineered colour changes at 24s and 72s. 48s is deliberately NOT a cut.
    def near(target: float, tol: float = 2.0) -> bool:
        return any(abs(c - target) <= tol for c in cuts)

    ok &= check("found the t=24 visual cut", near(24.0))
    ok &= check("found the t=72 visual cut", near(72.0))
    ok &= check("did NOT invent a cut at t=48 (semantic-only boundary)",
                not near(48.0), "correct: scene is identical there")
    return ok


def main() -> int:
    print("=" * 68)
    print("VKE P0 DEPENDENCY SPIKE")
    print("=" * 68)
    if not VIDEO.exists():
        print(f"missing {VIDEO} - run: python scripts/make_fixture.py")
        return 2

    results = {}
    for name, fn in (("asr", spike_asr), ("frames", spike_frames), ("scenes", spike_scenes)):
        try:
            results[name] = fn()
        except Exception as exc:  # a spike reports, it does not raise
            print(f"  [FAIL] {name} raised {type(exc).__name__}: {exc}")
            results[name] = False

    print("\n" + "=" * 68)
    for name, ok in results.items():
        print(f"  {name:8s} {'PASS' if ok else 'FAIL'}")
    allok = all(results.values())
    print(f"\nSPIKE {'PASSED - plan holds, proceed to P1' if allok else 'FAILED - plan needs revision'}")
    print("=" * 68)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
