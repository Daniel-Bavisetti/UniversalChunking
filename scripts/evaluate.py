"""Evaluate the three chunking configurations against hand-labelled boundaries.

Deliberately excluded as unfalsifiable: LLM-judged "coherence" scored by the same
model family that wrote the summaries, and any metric where VKE defines its own
reference.

Usage:
    python scripts/evaluate.py                       # fixture + its ground truth
    python scripts/evaluate.py --video my_talk --truth labels.json
    python scripts/evaluate.py --jsonl out.jsonl     # also write per-boundary rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from vke.config import CONFIGS  # noqa: E402
from vke.store import VideoStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOLERANCES = (2.0, 5.0)


def match(predicted: list[float], truth: list[float], tol: float
          ) -> tuple[int, list[tuple[float, float | None]]]:
    """Greedy one-to-one matching within `tol` seconds."""
    unused = list(predicted)
    pairs: list[tuple[float, float | None]] = []
    hits = 0
    for t in truth:
        near = [p for p in unused if abs(p - t) <= tol]
        if near:
            best = min(near, key=lambda p: abs(p - t))
            unused.remove(best)
            pairs.append((t, best))
            hits += 1
        else:
            pairs.append((t, None))
    return hits, pairs


def prf(hits: int, n_pred: int, n_truth: int) -> tuple[float, float, float]:
    p = hits / n_pred if n_pred else 0.0
    r = hits / n_truth if n_truth else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="fixture")
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--jsonl", type=Path, default=None)
    args = ap.parse_args()

    truth_path = args.truth or (ROOT / "data" / "fixture_ground_truth.json")
    if not truth_path.exists():
        print(f"no ground truth at {truth_path}")
        return 2
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    targets: list[float] = truth["boundaries"]
    kinds: dict[str, str] = truth.get("kinds", {})

    store = VideoStore(args.video)
    units_by_config = store.load_units()
    if not units_by_config:
        print(f"no units for '{args.video}'; run scripts/process.py first")
        return 2

    print(f"video: {args.video}   reference boundaries: "
          f"{[round(t, 1) for t in targets]}\n")

    header = (f"{'config':<17}{'units':>6}{'mean len':>10}"
              + "".join(f"{'F1@'+str(int(t))+'s':>9}" for t in TOLERANCES)
              + f"{'err':>8}{'quality':>9}")
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for key, cfg in CONFIGS.items():
        units = units_by_config.get(key, [])
        predicted = [round(u.span.start, 3) for u in units[1:]]  # unit 0 starts at t=0
        lengths = [u.span.duration for u in units]
        quality = [u.quality for u in units if u.quality is not None]

        line = f"{cfg.label:<17}{len(units):>6}{np.mean(lengths) if lengths else 0:>9.1f}s"
        f1s = {}
        for tol in TOLERANCES:
            hits, pairs = match(predicted, targets, tol)
            _p, _r, f1 = prf(hits, len(predicted), len(targets))
            f1s[tol] = f1
            line += f"{f1:>9.2f}"
            if tol == TOLERANCES[-1]:
                for t, hit in pairs:
                    rows.append({
                        "video_id": args.video,
                        "config": key,
                        "reference_boundary": t,
                        "predicted_boundary": hit,
                        "boundary_error": round(abs(hit - t), 3) if hit else None,
                        "kind": kinds.get(str(t), "unknown"),
                        "tolerance": tol,
                    })

        errs = [min((abs(p - t) for p in predicted), default=999.0) for t in targets]
        mean_err = float(np.mean(errs)) if errs else 999.0
        mean_q = float(np.mean(quality)) if quality else 0.0
        line += f"{mean_err:>7.1f}s{mean_q:>9.3f}"
        print(line)
        summary[key] = {"f1": f1s, "err": mean_err, "predicted": predicted}

    # --- the headline: what does the visual weight buy? -------------------- #
    print("\nPer-boundary detection (tolerance 5s):")
    print(f"  {'reference':>10}  {'kind':<16}" +
          "".join(f"{c.label:>17}" for c in CONFIGS.values()))
    for t in targets:
        row = f"  {t:>9.1f}s  {kinds.get(str(t),'?'):<16}"
        for key in CONFIGS:
            hit = any(abs(p - t) <= 5.0 for p in summary[key]["predicted"])
            row += f"{'FOUND' if hit else 'missed':>17}"
        print(row)

    b = summary.get("audio_only", {}).get("predicted", [])
    c = summary.get("vke_multimodal", {}).get("predicted", [])
    wins = [t for t in targets
            if any(abs(p - t) <= 5.0 for p in c) and not any(abs(p - t) <= 5.0 for p in b)]
    print(f"\n  Boundaries VKE finds that audio-only misses: "
          f"{[round(t,1) for t in wins] if wins else 'none'}")
    print("  (Audio-only and VKE differ ONLY in the visual weight: 0.00 -> 0.40)")

    print(f"\n  Expensive model calls: 0   (no VLM or LLM in the offline path)")

    if args.jsonl:
        args.jsonl.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(f"\nwrote {len(rows)} rows to {args.jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
