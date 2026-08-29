"""Heuristic speaker segmentation.

Honest about what it is: this detects speaker *changes* from cheap acoustic
features, it does not identify people. Ids are opaque (`speaker_00`) and every
turn carries a confidence.

Deliberately not a neural diarizer - pyannote needs torch, which does not install
on this Python. The features here (pitch proxy, spectral centroid, energy) are
enough to separate voices that differ noticeably, and the confidence reports when
they do not.

Audio is decoded with PyAV, which faster-whisper already bundles, so this costs
no new dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schemas import Span, SpeakerTurn, Utterance

TARGET_RATE = 16000
MIN_TURN_SECONDS = 3.0


def load_audio(path: Path, rate: int = TARGET_RATE) -> tuple[np.ndarray, int]:
    """Decode the audio track to mono float32. Returns (samples, rate)."""
    try:
        import av
    except ImportError:
        return np.zeros(0, dtype=np.float32), rate

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return np.zeros(0, dtype=np.float32), rate
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray()
                    chunks.append(arr.reshape(-1).astype(np.float32))
            if not chunks:
                return np.zeros(0, dtype=np.float32), rate
            return np.concatenate(chunks), rate
    except Exception:
        return np.zeros(0, dtype=np.float32), rate


def _features(segment: np.ndarray, rate: int) -> np.ndarray | None:
    """Three cheap voice descriptors: centroid, bandwidth, energy."""
    if segment.size < rate // 8:  # under ~0.12s is not worth measuring
        return None

    window = np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(segment * window))
    freqs = np.fft.rfftfreq(segment.size, 1.0 / rate)

    total = spectrum.sum()
    if total < 1e-8:
        return None

    # Restrict to the speech band; room tone and hiss carry no speaker identity.
    band = (freqs >= 80) & (freqs <= 4000)
    spec, f = spectrum[band], freqs[band]
    if spec.sum() < 1e-8:
        return None

    centroid = float((spec * f).sum() / spec.sum())
    bandwidth = float(np.sqrt((spec * (f - centroid) ** 2).sum() / spec.sum()))
    energy = float(np.sqrt(np.mean(segment ** 2)))
    return np.array([centroid / 1000.0, bandwidth / 1000.0, energy * 10.0])


def diarize(
    video_path: Path,
    utterances: list[Utterance],
    max_speakers: int = 4,
) -> list[SpeakerTurn]:
    """Group utterances into speakers by acoustic similarity.

    Utterance-level rather than frame-level: ASR has already given us the speech
    boundaries, so we only need to decide which utterances sound alike.
    """
    if not utterances:
        return []

    samples, rate = load_audio(video_path)
    if samples.size == 0:
        return []

    vectors: list[np.ndarray] = []
    indexed: list[int] = []
    for i, utt in enumerate(utterances):
        a = int(utt.span.start * rate)
        b = int(min(utt.span.end, utt.span.start + 6.0) * rate)
        feat = _features(samples[a:min(b, samples.size)], rate)
        if feat is not None:
            vectors.append(feat)
            indexed.append(i)

    if len(vectors) < 2:
        return [SpeakerTurn(span=u.span, speaker="speaker_00", confidence=0.3)
                for u in utterances]

    matrix = np.vstack(vectors)
    mean, std = matrix.mean(axis=0), matrix.std(axis=0) + 1e-8
    normed = (matrix - mean) / std

    labels, separation = _cluster(normed, max_speakers)

    # Separation is how far apart the clusters actually are. When it is low the
    # voices are not distinguishable and we say so rather than inventing speakers.
    # Silhouette below ~0.45 means there is no real cluster structure - one voice
    # with natural variation. Reporting speakers we cannot actually hear would be
    # worse than reporting one.
    if separation < 0.45:
        labels = np.zeros(len(labels), dtype=int)
        confidence = 0.25
    else:
        confidence = float(np.clip(separation, 0.45, 0.9))

    assignment = {idx: int(lbl) for idx, lbl in zip(indexed, labels)}
    turns: list[SpeakerTurn] = []
    for i, utt in enumerate(utterances):
        speaker = f"speaker_{assignment.get(i, 0):02d}"
        turns.append(SpeakerTurn(span=utt.span, speaker=speaker, confidence=confidence))

    return _smooth(turns)


def _silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient in [-1, 1].

    A between/within ratio rises monotonically with k, so it always picks the
    largest k allowed and happily reports four speakers for one voice. Silhouette
    penalizes splitting a homogeneous group, which is exactly the mistake we need
    to avoid: claiming speakers that are not there.
    """
    unique = np.unique(labels)
    if unique.size < 2:
        return 0.0

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    scores: list[float] = []
    for i in range(len(points)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue  # a singleton cluster tells us nothing
        a = float(distances[i][same].mean())
        b = min(
            float(distances[i][labels == other].mean())
            for other in unique if other != labels[i]
        )
        denom = max(a, b)
        if denom > 0:
            scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else 0.0


def _cluster(points: np.ndarray, max_k: int) -> tuple[np.ndarray, float]:
    """k-means over k, choosing k by silhouette. No sklearn dependency."""
    best_labels = np.zeros(len(points), dtype=int)
    best_score = 0.0

    for k in range(2, min(max_k, len(points) // 2) + 1):
        labels, _centres = _kmeans(points, k)
        if labels is None:
            continue
        score = _silhouette(points, labels)
        if score > best_score:
            best_labels, best_score = labels, score

    return best_labels, best_score


def _kmeans(points: np.ndarray, k: int, iters: int = 25
            ) -> tuple[np.ndarray | None, np.ndarray]:
    rng = np.random.default_rng(0)  # deterministic: reproducible runs matter
    centres = points[rng.choice(len(points), size=k, replace=False)]
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iters):
        distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            member = points[labels == j]
            if len(member):
                centres[j] = member.mean(axis=0)
    if len(set(labels.tolist())) < k:
        return None, centres
    return labels, centres


def _smooth(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Drop single-utterance flickers; a real turn lasts more than one sentence."""
    if len(turns) < 3:
        return turns
    out = list(turns)
    for i in range(1, len(out) - 1):
        if out[i - 1].speaker == out[i + 1].speaker != out[i].speaker:
            if out[i].span.duration < MIN_TURN_SECONDS:
                out[i] = SpeakerTurn(span=out[i].span, speaker=out[i - 1].speaker,
                                     confidence=out[i].confidence * 0.8,
                                     method=out[i].method)
    return out


def apply_to_utterances(
    utterances: list[Utterance], turns: list[SpeakerTurn]
) -> list[Utterance]:
    by_start = {round(t.span.start, 3): t.speaker for t in turns}
    for utt in utterances:
        utt.speaker = by_start.get(round(utt.span.start, 3))
    return utterances


def speaker_changes(turns: list[SpeakerTurn]) -> list[tuple[float, float]]:
    """(timestamp, confidence) for each point where the speaker changes."""
    changes: list[tuple[float, float]] = []
    for prev, nxt in zip(turns, turns[1:]):
        if prev.speaker != nxt.speaker:
            changes.append((nxt.span.start, min(prev.confidence, nxt.confidence)))
    return changes
