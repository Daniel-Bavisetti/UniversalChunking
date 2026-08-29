"""Boundaries -> Knowledge Units.

All three configs run through this same code path. A, the VideoRAG baseline, cuts
on a fixed clock and ignores every signal; B and C differ only in their weight
vector. That is what makes the comparison an ablation rather than three separate
implementations.
"""

from __future__ import annotations

import numpy as np

from .config import (
    CHUNKER_VERSION,
    MAX_EVENT_SECONDS,
    MERGE_COHESION,
    MERGE_MAX_SECONDS,
    MIN_EVENT_SECONDS,
    PIPELINE_VERSION,
    SNAP_WINDOW,
    SPLIT_COHESION,
    SPLIT_MIN_SECONDS,
    THRESHOLD_K,
    ChunkConfig,
)
from .media import describe_visual
from .schemas import (
    BoundaryExplanation,
    FrameFeature,
    KnowledgeUnit,
    SceneCut,
    SignalContribution,
    Span,
    Utterance,
    VisualObservation,
)
from .signals import SignalCurve, fuse
from .text import tokenize


# --------------------------------------------------------------------------- #
# boundary detection
# --------------------------------------------------------------------------- #
def _local_maxima(score: np.ndarray, threshold: float) -> list[int]:
    return [
        i
        for i in range(1, score.size - 1)
        if score[i] >= threshold
        and score[i] >= score[i - 1]
        and score[i] >= score[i + 1]
    ]


def _suppress(
    candidates: list[tuple[float, float]], min_gap: float
) -> list[float]:
    """Greedy non-maximum suppression: keep the strongest, drop its neighbours."""
    chosen: list[float] = []
    for _score, t in sorted(candidates, key=lambda c: -c[0]):
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def _snap(t: float, edges: list[float], window: float) -> float | None:
    """Snap to the nearest utterance edge so we never cut mid-sentence."""
    if not edges:
        return None
    nearest = min(edges, key=lambda e: abs(e - t))
    return nearest if abs(nearest - t) <= window else None


def _explain(
    ts: float,
    grid: np.ndarray,
    score: np.ndarray,
    curves: dict[str, SignalCurve],
    weights: dict[str, float],
    threshold: float,
    snapped_from: float | None,
) -> BoundaryExplanation:
    idx = int(np.argmin(np.abs(grid - ts)))
    contributions = [
        SignalContribution(
            name=name,
            raw=round(float(curve.raw[idx]), 4),
            normalized=round(float(curve.normalized[idx]), 4),
            weight=weights.get(name, 0.0),
            contribution=round(
                float(curve.normalized[idx]) * weights.get(name, 0.0), 4
            ),
        )
        for name, curve in curves.items()
    ]
    contributions.sort(key=lambda c: -c.contribution)

    top = [c for c in contributions if c.contribution > 0.05]
    if top:
        names = ", ".join(f"{c.name} {c.contribution:.2f}" for c in top)
        summary = f"Boundary driven by {names} (threshold {threshold:.2f})."
    else:
        summary = f"Weak boundary (threshold {threshold:.2f})."

    return BoundaryExplanation(
        ts=round(ts, 3),
        score=round(float(score[idx]), 4),
        threshold=round(threshold, 4),
        signals=contributions,
        snapped_from=round(snapped_from, 3) if snapped_from is not None else None,
        summary=summary,
    )


def _enforce_max(
    boundaries: list[float],
    duration: float,
    grid: np.ndarray,
    score: np.ndarray,
    max_seconds: float,
) -> list[float]:
    """Split any interval longer than max_seconds at its strongest interior point."""
    out = list(boundaries)
    changed = True
    while changed:
        changed = False
        edges = [0.0, *sorted(out), duration]
        for a, b in zip(edges, edges[1:]):
            if b - a <= max_seconds:
                continue
            mask = (grid > a + max_seconds * 0.25) & (grid < b - max_seconds * 0.25)
            if not mask.any():
                mid = (a + b) / 2.0
            else:
                mid = float(grid[mask][int(np.argmax(score[mask]))])
            out.append(mid)
            changed = True
            break
    return sorted(out)


def detect_boundaries(
    curves: dict[str, SignalCurve],
    config: ChunkConfig,
    utterances: list[Utterance],
    duration: float,
) -> tuple[list[BoundaryExplanation], np.ndarray, np.ndarray, float]:
    """Return (boundaries, grid, score, threshold) for one config."""
    grid = next(iter(curves.values())).grid

    if config.is_fixed:
        # Baseline A: VideoRAG's fixed-window chunker. No signal is consulted, so
        # the explanation honestly records that.
        step = config.fixed_window or 30.0
        zeros = np.zeros(grid.size)
        boundaries = [
            BoundaryExplanation(
                ts=round(t, 3),
                score=0.0,
                threshold=0.0,
                signals=[],
                summary=f"Fixed {step:.0f}s window. No extraction influenced this cut.",
            )
            for t in np.arange(step, duration - 1.0, step)
        ]
        return boundaries, grid, zeros, 0.0

    result = fuse(curves, config.weights, THRESHOLD_K)
    peaks = _local_maxima(result.score, result.threshold)
    candidates = [(float(result.score[i]), float(grid[i])) for i in peaks]
    kept = _suppress(candidates, MIN_EVENT_SECONDS)

    edges = [u.span.start for u in utterances if u.span.start > 0.5]
    snapped: list[tuple[float, float | None]] = []
    for t in kept:
        target = _snap(t, edges, SNAP_WINDOW)
        snapped.append((target, t) if target is not None else (t, None))

    # Snapping can collide two boundaries onto one utterance edge.
    deduped: list[tuple[float, float | None]] = []
    for ts, origin in sorted(snapped):
        if deduped and abs(ts - deduped[-1][0]) < 1.0:
            continue
        deduped.append((ts, origin))

    final = _enforce_max(
        [ts for ts, _ in deduped], duration, grid, result.score, MAX_EVENT_SECONDS
    )
    origins = {ts: origin for ts, origin in deduped}

    explanations = [
        _explain(ts, grid, result.score, curves, config.weights,
                 result.threshold, origins.get(ts))
        for ts in final
        if 0.5 < ts < duration - 0.5
    ]
    return explanations, grid, result.score, result.threshold


# --------------------------------------------------------------------------- #
# unit assembly
# --------------------------------------------------------------------------- #
def _extractive_title(text: str, max_len: int = 62) -> str:
    """Highest-scoring sentence by content-word frequency, truncated.

    Extractive, not abstractive: with no LLM in the MVP we surface real words the
    speaker used rather than inventing a label.
    """
    if not text.strip():
        return "(no speech)"

    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".")
                 if len(s.strip()) > 12]
    if not sentences:
        sentences = [text.strip()]

    freq: dict[str, int] = {}
    for tok in tokenize(text):
        freq[tok] = freq.get(tok, 0) + 1

    def score(sentence: str) -> float:
        toks = tokenize(sentence)
        if not toks:
            return 0.0
        # Mean term frequency, mildly favouring earlier sentences.
        return sum(freq.get(t, 0) for t in toks) / len(toks)

    best = max(sentences, key=score)
    best = " ".join(best.split())
    return best if len(best) <= max_len else best[: max_len - 1].rstrip() + "…"


def _key_terms(text: str, limit: int = 6) -> list[str]:
    freq: dict[str, int] = {}
    for tok in tokenize(text):
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, c in ranked[:limit] if c > 1] or [w for w, _ in ranked[:limit]]


def attach_observations(
    unit: KnowledgeUnit,
    observations: list[VisualObservation],
) -> KnowledgeUnit:
    """Fold every observation inside the unit's span into the unit.

    `objects`, `ocr_text` and `actions` are pure PROJECTIONS of `observations` -
    they are rebuilt here from scratch and never written any other way. That is
    what makes the provenance guarantee structural rather than a convention: a
    label cannot appear in `objects` without an observation carrying its source,
    timestamp, model and confidence sitting behind it.

    Note what is absent: nothing derives an action from motion, edge density or
    scene cuts. `actions` stays empty unless a real semantic model produced one.
    """
    inside = [o for o in observations if unit.span.contains(o.ts)]

    # One frame of a slide and the next frame of the same slide both say "Deploy";
    # keep the most confident sighting of each distinct value.
    best: dict[tuple[str, str], VisualObservation] = {}
    for o in inside:
        key = (o.kind, o.value)
        current = best.get(key)
        if current is None or (o.confidence or 0.0) > (current.confidence or 0.0):
            best[key] = o

    kept = sorted(best.values(), key=lambda o: (-(o.confidence or 0.0), o.ts, o.value))
    unit.observations = kept

    def values(kind: str) -> list[str]:
        return [o.value for o in kept if o.kind == kind]

    unit.objects = values("object")
    unit.ocr_text = values("text")
    unit.actions = values("action")
    unit.visual_sources = sorted({o.source for o in kept})
    return unit


def build_units(
    video_id: str,
    config: ChunkConfig,
    boundaries: list[BoundaryExplanation],
    utterances: list[Utterance],
    features: list[FrameFeature],
    cuts: list[SceneCut],
    duration: float,
    providers: dict[str, str] | None = None,
    observations: list[VisualObservation] | None = None,
    enrichment: dict[str, str] | None = None,
) -> list[KnowledgeUnit]:
    edges = [0.0, *[b.ts for b in boundaries], duration]
    scene_starts = [0.0, *[c.ts for c in cuts]]

    units: list[KnowledgeUnit] = []
    for i, (start, end) in enumerate(zip(edges, edges[1:])):
        if end - start < 1.0:
            continue
        span = Span(start=round(start, 3), end=round(end, 3))

        overlapping = [
            u for u in utterances if u.span.start < end and u.span.end > start
        ]
        text = " ".join(u.text for u in overlapping).strip()
        speakers = sorted({u.speaker for u in overlapping if u.speaker})

        in_window = [f for f in features if start <= f.ts < end]
        scene_ids = [
            idx for idx, s in enumerate(scene_starts)
            if s < end and (idx + 1 >= len(scene_starts) or scene_starts[idx + 1] > start)
        ]

        # The opening boundary of the video is not a detected boundary.
        boundary = (
            boundaries[i - 1]
            if i > 0 and i - 1 < len(boundaries)
            else BoundaryExplanation(
                ts=round(start, 3), score=0.0, threshold=0.0, signals=[],
                summary="Start of video.",
            )
        )

        unit_id = f"{config.key}_{i:03d}"
        unit = KnowledgeUnit(
            id=unit_id,
            video_id=video_id,
            span=span,
            title=_extractive_title(text),
            transcript=text,
            visual_context=describe_visual(in_window),
            scene_ids=scene_ids,
            keyframe_url=f"/api/videos/{video_id}/keyframe/{unit_id}.jpg",
            boundary=boundary,
            config=config.key,
            entities=_key_terms(text),
            speakers=speakers,
            provenance={
                "pipeline_version": PIPELINE_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "config": config.key,
                "weights": dict(config.weights),
                "providers": dict(providers or {}),
                # Per-producer status, so an empty objects/ocr_text list is always
                # explainable: "0 detections" and "never ran" are different facts.
                "enrichment": dict(enrichment or {}),
            },
        )
        units.append(attach_observations(unit, observations or []))

    for prev, nxt in zip(units, units[1:]):
        prev.next_unit_id = nxt.id
        nxt.prev_unit_id = prev.id
    return units


# --------------------------------------------------------------------------- #
# refinement: merge over-segmentation, split under-segmentation
# --------------------------------------------------------------------------- #
def _cohesion(a: str, b: str) -> float:
    """Vocabulary overlap between two stretches of transcript."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def refine_boundaries(
    boundaries: list[BoundaryExplanation],
    utterances: list[Utterance],
    duration: float,
    grid: np.ndarray,
    score: np.ndarray,
    merge_below: float = MERGE_MAX_SECONDS,
    merge_cohesion: float = MERGE_COHESION,
    split_above: float = SPLIT_MIN_SECONDS,
) -> tuple[list[BoundaryExplanation], dict[str, int]]:
    """Second pass over the boundary set.

    Peak-picking is local, so it can over-segment a continuous explanation and
    under-segment a long stretch that drifts across topics. This pass fixes both,
    and only where the evidence supports it:

      MERGE - drop a boundary between two short neighbours that share most of
              their vocabulary (the split was noise, not a topic change).
      SPLIT - add one inside a long unit whose halves share almost no vocabulary
              (a real change the local peak was too weak to expose).
    """
    stats = {"merged": 0, "split": 0}
    if not utterances:
        return boundaries, stats

    def text_for(a: float, b: float) -> str:
        return " ".join(
            u.text for u in utterances if u.span.start < b and u.span.end > a
        )

    # --- merge ------------------------------------------------------------- #
    kept: list[BoundaryExplanation] = []
    for boundary in boundaries:
        edges = [0.0, *[k.ts for k in kept], boundary.ts]
        prev_start = edges[-2]
        after_end = min(boundary.ts + merge_below, duration)
        left, right = text_for(prev_start, boundary.ts), text_for(boundary.ts, after_end)

        short = (boundary.ts - prev_start) < merge_below
        similar = _cohesion(left, right) >= merge_cohesion
        if short and similar and kept:
            stats["merged"] += 1
            continue  # the boundary was noise; drop it
        kept.append(boundary)

    # --- split ------------------------------------------------------------- #
    out: list[BoundaryExplanation] = []
    edges = [0.0, *[k.ts for k in kept], duration]
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        if i - 1 >= 0 and i - 1 < len(kept):
            out.append(kept[i - 1])
        if (b - a) < split_above:
            continue
        mid = (a + b) / 2.0
        if _cohesion(text_for(a, mid), text_for(mid, b)) > SPLIT_COHESION:
            continue  # one coherent topic that simply runs long

        window = (grid > a + MIN_EVENT_SECONDS) & (grid < b - MIN_EVENT_SECONDS)
        if not window.any():
            continue
        best = float(grid[window][int(np.argmax(score[window]))])
        snapped = _snap(best, [u.span.start for u in utterances], SNAP_WINDOW) or best
        out.append(BoundaryExplanation(
            ts=round(snapped, 3),
            score=float(score[int(np.argmin(np.abs(grid - snapped)))]),
            threshold=0.0,
            signals=[],
            snapped_from=round(best, 3) if snapped != best else None,
            summary=("Split: this stretch ran long and its halves share almost no "
                     "vocabulary, so it held more than one topic."),
        ))
        stats["split"] += 1

    if len(kept) > len(edges) - 2:
        out.extend(kept[len(edges) - 2:])
    return sorted({round(b.ts, 3): b for b in out}.values(), key=lambda b: b.ts), stats
