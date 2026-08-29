"""TextTiling (Hearst, 1997) - lexical cohesion boundary detection.

Two details make this work, and omitting either produces a flat, useless signal:

1. Blocks are measured in TOKENS, not seconds. Speech density varies wildly, so
   fixed time windows produce empty or lopsided comparisons.

2. The boundary strength is the DEPTH SCORE, not the raw dissimilarity. In short
   speech, content words rarely repeat, so `1 - cosine` sits near 1.0 almost
   everywhere and discriminates nothing. The depth score measures how far a
   similarity valley falls below the peaks on either side of it, which cancels
   that baseline out.
"""

from __future__ import annotations

import math

import numpy as np


def block_similarity(
    tokens: list[str],
    times: list[float],
    block: int,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity between the `block` tokens either side of each gap.

    Returns (gap_times, similarities).
    """
    n = len(tokens)
    gap_times: list[float] = []
    sims: list[float] = []

    for i in range(block, n - block + 1, step):
        before = tokens[i - block:i]
        after = tokens[i:i + block]

        tf_b: dict[str, int] = {}
        tf_a: dict[str, int] = {}
        for t in before:
            tf_b[t] = tf_b.get(t, 0) + 1
        for t in after:
            tf_a[t] = tf_a.get(t, 0) + 1

        shared = set(tf_b) & set(tf_a)
        if shared:
            dot = sum(tf_b[k] * tf_a[k] for k in shared)
            nb = math.sqrt(sum(v * v for v in tf_b.values()))
            na = math.sqrt(sum(v * v for v in tf_a.values()))
            sim = dot / (nb * na) if nb and na else 0.0
        else:
            sim = 0.0

        # Place the comparison at the MIDPOINT between the two blocks, not at
        # the first token after the gap. Blocks are counted in tokens, so a long
        # pause makes consecutive token times jump; anchoring on times[i] would
        # skip straight over the silence where the boundary actually is.
        gap_times.append((times[i - 1] + times[i]) / 2.0)
        sims.append(sim)

    return np.asarray(gap_times, dtype=np.float64), np.asarray(sims, dtype=np.float64)


def depth_scores(sims: np.ndarray) -> np.ndarray:
    """Hearst's depth score: how deep is this valley relative to its shoulders.

    For each position, walk left and right to the nearest local maximum and sum
    the two rises. A deep, isolated valley scores high; a uniformly low region
    scores near zero.
    """
    n = sims.size
    if n == 0:
        return sims
    depths = np.zeros(n, dtype=np.float64)

    # Depth belongs at local minima only. Scoring every position turns a long
    # flat valley (common when vocabulary is fully disjoint) into a wide plateau
    # instead of a boundary, and the plateau then dominates normalization.
    minima: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sims[j + 1] == sims[i]:
            j += 1
        left_ok = i == 0 or sims[i - 1] > sims[i]
        right_ok = j == n - 1 or sims[j + 1] > sims[j]
        if left_ok and right_ok:
            minima.append((i + j) // 2)  # middle of a flat run
        i = j + 1

    for i in minima:
        # walk left while the sequence is (weakly) rising away from i
        left = sims[i]
        j = i
        while j > 0 and sims[j - 1] >= sims[j]:
            j -= 1
            left = sims[j]

        # walk right likewise
        right = sims[i]
        k = i
        while k < n - 1 and sims[k + 1] >= sims[k]:
            k += 1
            right = sims[k]

        depths[i] = (left - sims[i]) + (right - sims[i])

    return depths


def smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    if values.size < window or window < 2:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def boundary_strength(
    tokens: list[str],
    times: list[float],
    grid: np.ndarray,
    block: int | None = None,
) -> np.ndarray:
    """Depth score resampled onto the shared time grid."""
    n = len(tokens)
    if n < 8:
        return np.zeros(grid.size, dtype=np.float64)

    if block is None:
        block = max(8, min(120, n // 4))
    if n < 2 * block + 1:
        block = max(3, n // 3)
    if n < 2 * block + 1:
        return np.zeros(grid.size, dtype=np.float64)

    gap_times, sims = block_similarity(tokens, times, block)
    if gap_times.size == 0:
        return np.zeros(grid.size, dtype=np.float64)

    depths = depth_scores(smooth(sims))
    return np.interp(grid, gap_times, depths, left=0.0, right=0.0)
