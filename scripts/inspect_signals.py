"""Plot s(t) and its components before anything is built on top of it.

Risk #5 in the plan: if the curve is bad, everything above it looks bad. So we
look at the curve first, against a fixture whose boundaries we know.

Usage:  python scripts/inspect_signals.py [video.mp4]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vke import media, signals  # noqa: E402
from vke.asr import transcribe  # noqa: E402
from vke.config import CONFIG_B, CONFIG_C, THRESHOLD_K  # noqa: E402
from vke.schemas import FrameFeature, SceneCut, Utterance  # noqa: E402

# Windows consoles default to cp1252, which cannot render block characters.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    BLOCKS_ASCII = True

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BLOCKS = " ▁▂▃▄▅▆▇█"


def spark(values: np.ndarray, width: int = 96) -> str:
    if values.size == 0:
        return ""
    idx = np.linspace(0, values.size - 1, width).astype(int)
    sampled = values[idx]
    hi = float(sampled.max()) or 1.0
    return "".join(BLOCKS[int(round(v / hi * (len(BLOCKS) - 1)))] for v in sampled)


def axis(duration: float, width: int = 96) -> str:
    line = [" "] * width
    for frac in (0.0, 0.25, 0.5, 0.75):
        pos = int(frac * width)
        label = f"{duration*frac:.0f}s"
        for i, ch in enumerate(label):
            if pos + i < width:
                line[pos + i] = ch
    return "".join(line)


def extract(video: Path, cache: Path) -> tuple[list, list, list, float]:
    """Extraction is slow; cache it so the curve can be re-tuned instantly."""
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        print(f"(using cached extraction: {cache.name})")
        return (
            [Utterance(**u) for u in blob["utterances"]],
            [FrameFeature(**f) for f in blob["features"]],
            [SceneCut(**c) for c in blob["cuts"]],
            blob["duration"],
        )

    print("extracting (first run is slow, then cached)...")
    t0 = time.time()
    meta = media.probe(video, video.stem)
    print(f"  probe      {time.time()-t0:5.1f}s  {meta.width}x{meta.height} "
          f"{meta.fps:.1f}fps {meta.duration:.1f}s audio={meta.has_audio}")

    t0 = time.time()
    utterances, provider = transcribe(video)
    print(f"  asr        {time.time()-t0:5.1f}s  {len(utterances)} utterances "
          f"via {provider}")

    t0 = time.time()
    features = media.extract_frames(video, DATA / "_frames")
    print(f"  frames     {time.time()-t0:5.1f}s  {len(features)} sampled")

    t0 = time.time()
    cuts = media.detect_scenes(video)
    print(f"  scenes     {time.time()-t0:5.1f}s  cuts at "
          f"{[round(c.ts,1) for c in cuts]}")

    cache.write_text(json.dumps({
        "utterances": [u.model_dump() for u in utterances],
        "features": [f.model_dump() for f in features],
        "cuts": [c.model_dump() for c in cuts],
        "duration": meta.duration,
    }), encoding="utf-8")
    return utterances, features, cuts, meta.duration


def peaks_above(grid: np.ndarray, score: np.ndarray, threshold: float,
                min_gap: float = 15.0) -> list[float]:
    """Local maxima above threshold, with non-maximum suppression."""
    candidates = [
        (float(score[i]), float(grid[i]))
        for i in range(1, score.size - 1)
        if score[i] >= threshold and score[i] >= score[i-1] and score[i] >= score[i+1]
    ]
    chosen: list[float] = []
    for _value, t in sorted(candidates, reverse=True):
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "fixture.mp4"
    if not video.exists():
        print(f"missing {video}")
        return 2

    gt_path = DATA / "fixture_ground_truth.json"
    truth = json.loads(gt_path.read_text()) if gt_path.exists() else None

    utterances, features, cuts, duration = extract(
        video, DATA / f"_extract_{video.stem}.json")

    curves = signals.compute_curves(utterances, features, cuts, duration)

    print(f"\n{'='*100}")
    print("SIGNAL CURVES (normalized)")
    print(f"{'='*100}")
    for name in ("semantic", "visual", "silence"):
        print(f"  {name:9s} |{spark(curves[name].normalized)}|")
    print(f"  {'':9s}  {axis(duration)}")

    if truth:
        marks = [" "] * 96
        for b in truth["boundaries"]:
            pos = int(b / duration * 96)
            if 0 <= pos < 96:
                marks[pos] = "^"
        print(f"  {'truth':9s}  {''.join(marks)}")
        print(f"             expected: " + "  ".join(
            f"{b:.0f}s({truth['kinds'][str(b)]})" for b in truth["boundaries"]))

    print(f"\n{'='*100}")
    print("FUSED SCORE s(t)")
    print(f"{'='*100}")

    results = {}
    for cfg in (CONFIG_B, CONFIG_C):
        res = signals.fuse(curves, cfg.weights, THRESHOLD_K)
        found = peaks_above(res.grid, res.score, res.threshold)
        results[cfg.key] = found
        w = "  ".join(f"{k}={v:.2f}" for k, v in cfg.weights.items())
        print(f"\n  {cfg.label}  ({w})   threshold={res.threshold:.3f}")
        print(f"  {'':9s} |{spark(res.score)}|")
        print(f"  {'':9s}  {axis(duration)}")
        print(f"  peaks: {[round(p,1) for p in found]}")

    if truth:
        print(f"\n{'='*100}")
        print("VERDICT")
        print(f"{'='*100}")
        tol = 3.0

        def matched(found: list[float], target: float) -> float | None:
            near = [f for f in found if abs(f - target) <= tol]
            return min(near, key=lambda f: abs(f - target)) if near else None

        for cfg_key, found in results.items():
            print(f"\n  {cfg_key}:")
            for b in truth["boundaries"]:
                hit = matched(found, b)
                kind = truth["kinds"][str(b)]
                if hit is not None:
                    print(f"    {b:5.1f}s {kind:15s} FOUND at {hit:5.1f}s "
                          f"(err {abs(hit-b):.1f}s)")
                else:
                    print(f"    {b:5.1f}s {kind:15s} missed")

        b_found = {b for b in truth["boundaries"] if matched(results["audio_only"], b)}
        c_found = {b for b in truth["boundaries"] if matched(results["vke_multimodal"], b)}
        visual_only = c_found - b_found
        print(f"\n  >>> boundaries VKE finds that audio-only misses: "
              f"{sorted(visual_only) if visual_only else 'NONE'}")
        if visual_only:
            print("  >>> THIS IS THE HEADLINE NUMBER - the visual signal earns its weight.")
        else:
            print("  >>> WARNING: no visual-only win. Tune weights before building on this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
