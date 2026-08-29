"""Boundary signals and the fused score s(t).

Three independent signals, each measured over the whole video, each normalized
to 0..1, then combined with a configurable weight vector:

    s(t) = w_sem * semantic(t) + w_vis * visual(t) + w_sil * silence(t)

Signals that were considered and rejected (plan sec.9): action/entity change
(circular - they need model output that only runs near boundaries we have not
found yet), topic_shift (the same measurement as semantic at another window),
scene_change as a separate term (folded into visual, else it double-counts),
speaker_change (high cost and risk, correlated with silence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import texttiling
from .config import (
    SCENE_KERNEL_SIGMA,
    SEMANTIC_BLOCK_SECONDS,
    SILENCE_FULL_SCORE,
    SPEAKER_KERNEL_SIGMA,
)
from .media import hist_distance
from .schemas import FrameFeature, SceneCut, Utterance
from .text import tokenize

GRID_STEP = 0.5  # seconds between evaluations of s(t)
VISUAL_WINDOW = 4.0  # seconds each side for the windowed histogram comparison

@dataclass
class SignalCurve:
    """One signal evaluated on the shared time grid."""

    name: str
    grid: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray


def build_grid(duration: float, step: float = GRID_STEP) -> np.ndarray:
    n = max(2, int(duration / step) + 1)
    return np.round(np.arange(n) * step, 3)


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def robust_normalize(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1 without letting one spike flatten everything else.

    Percentile scaling is the right default, but it collapses on a SPARSE signal:
    if the depth score is non-zero at only a handful of grid points, the 5th and
    95th percentiles are both 0 and the whole signal would be zeroed - silently
    deleting exactly the sharp, confident boundaries we most want to keep. So we
    fall back to true min-max whenever the percentile range degenerates.
    """
    if values.size == 0:
        return values

    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi - lo >= 1e-9:
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-9:
        return np.zeros_like(values)  # genuinely constant: no information
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 1. semantic - TextTiling (Hearst 1997) lexical cohesion
# --------------------------------------------------------------------------- #
def token_stream(utterances: list[Utterance]) -> tuple[list[str], list[float]]:
    """Content tokens in order, each with the absolute time it was spoken.

    Word-level timestamps are used when the ASR provides them; otherwise a token
    is placed by linear interpolation across its utterance.
    """
    tokens: list[str] = []
    times: list[float] = []
    for u in utterances:
        if u.words:
            for w in u.words:
                for tok in tokenize(w.text):
                    tokens.append(tok)
                    times.append(w.start)
        else:
            toks = tokenize(u.text)
            if not toks:
                continue
            span = max(u.span.duration, 1e-6)
            for i, tok in enumerate(toks):
                tokens.append(tok)
                times.append(u.span.start + span * i / len(toks))
    return tokens, times


def semantic_curve(
    utterances: list[Utterance],
    grid: np.ndarray,
    block_seconds: float = SEMANTIC_BLOCK_SECONDS,
) -> SignalCurve:
    """TextTiling depth score over the transcript (see vke/texttiling.py).

    The block is sized from the measured token rate so that it always spans
    ~block_seconds of speech, whatever the speaking pace of the video.
    """
    tokens, times = token_stream(utterances)
    block = None
    if tokens and times and times[-1] > times[0]:
        rate = len(tokens) / max(times[-1] - times[0], 1e-6)  # content tokens/sec
        block = max(8, min(120, int(round(rate * block_seconds))))
    raw = texttiling.boundary_strength(tokens, times, grid, block=block)
    return SignalCurve("semantic", grid, raw, robust_normalize(raw))


# --------------------------------------------------------------------------- #
# 2. visual - windowed histogram change, absorbing scene cuts
# --------------------------------------------------------------------------- #
def visual_curve(
    features: list[FrameFeature],
    cuts: list[SceneCut],
    grid: np.ndarray,
    window: float = VISUAL_WINDOW,
) -> SignalCurve:
    """max(windowed histogram distance, nearest scene-cut kernel).

    Taking the max rather than adding a fourth scene term is what stops the
    visual modality from silently carrying ~2x its intended weight: a cut and the
    histogram jump around it are the same physical event.
    """
    raw = np.zeros(grid.size, dtype=np.float64)

    if features:
        ts = np.array([f.ts for f in features])
        hists = [f.hsv_hist for f in features]
        for i, t in enumerate(grid):
            before_idx = np.where((ts < t) & (ts >= t - window))[0]
            after_idx = np.where((ts >= t) & (ts <= t + window))[0]
            if before_idx.size and after_idx.size:
                mean_before = np.mean([hists[j] for j in before_idx], axis=0)
                mean_after = np.mean([hists[j] for j in after_idx], axis=0)
                raw[i] = hist_distance(list(mean_before), list(mean_after))

    normalized = robust_normalize(raw)

    if cuts:
        kernel = np.zeros(grid.size, dtype=np.float64)
        for cut in cuts:
            kernel = np.maximum(
                kernel,
                cut.confidence * np.exp(
                    -((grid - cut.ts) ** 2) / (2.0 * SCENE_KERNEL_SIGMA ** 2)
                ),
            )
        normalized = np.maximum(normalized, kernel)

    return SignalCurve("visual", grid, raw, normalized)


# --------------------------------------------------------------------------- #
# 3. silence - pauses between utterances
# --------------------------------------------------------------------------- #
def silence_curve(
    utterances: list[Utterance],
    grid: np.ndarray,
    full_score: float = SILENCE_FULL_SCORE,
) -> SignalCurve:
    """A pause of `full_score` seconds or longer scores 1.0 at its midpoint."""
    raw = np.zeros(grid.size, dtype=np.float64)
    for prev, nxt in zip(utterances, utterances[1:]):
        gap = nxt.span.start - prev.span.end
        if gap <= 0.05:
            continue
        mid = (prev.span.end + nxt.span.start) / 2.0
        amplitude = min(gap / full_score, 1.0)
        sigma = max(0.6, min(gap / 2.0, 3.0))
        raw = np.maximum(
            raw, amplitude * np.exp(-((grid - mid) ** 2) / (2.0 * sigma ** 2))
        )
    # Already 0..1 by construction; normalizing again would amplify noise on
    # videos whose pauses are all short.
    return SignalCurve("silence", grid, raw, np.clip(raw, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# 4. speaker - handover between voices
# --------------------------------------------------------------------------- #
def speaker_curve(
    changes: list[tuple[float, float]],
    grid: np.ndarray,
) -> SignalCurve:
    """A narrow peak at each detected speaker handover, scaled by confidence.

    Contributes exactly nothing on single-speaker audio, which is the honest
    outcome: most demo footage has one presenter, and the signal says so instead
    of inventing turns.
    """
    raw = np.zeros(grid.size, dtype=np.float64)
    for ts, confidence in changes:
        raw = np.maximum(
            raw,
            confidence * np.exp(-((grid - ts) ** 2) / (2.0 * SPEAKER_KERNEL_SIGMA ** 2)),
        )
    return SignalCurve("speaker", grid, raw, np.clip(raw, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #
@dataclass
class ScoreResult:
    grid: np.ndarray
    score: np.ndarray
    curves: dict[str, SignalCurve]
    weights: dict[str, float]
    threshold: float


def compute_curves(
    utterances: list[Utterance],
    features: list[FrameFeature],
    cuts: list[SceneCut],
    duration: float,
    speaker_change_points: list[tuple[float, float]] | None = None,
) -> dict[str, SignalCurve]:
    """Every signal, computed once. Configs then reuse these with new weights."""
    grid = build_grid(duration)
    return {
        "semantic": semantic_curve(utterances, grid),
        "visual": visual_curve(features, cuts, grid),
        "silence": silence_curve(utterances, grid),
        "speaker": speaker_curve(speaker_change_points or [], grid),
    }


def fuse(
    curves: dict[str, SignalCurve],
    weights: dict[str, float],
    threshold_k: float,
) -> ScoreResult:
    grid = next(iter(curves.values())).grid
    score = np.zeros(grid.size, dtype=np.float64)
    for name, curve in curves.items():
        score += weights.get(name, 0.0) * curve.normalized

    threshold = float(np.mean(score) + threshold_k * np.std(score))
    return ScoreResult(grid, score, curves, weights, threshold)
