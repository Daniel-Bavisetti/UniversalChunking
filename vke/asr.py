"""Speech to timestamped utterances.

Primary path is faster-whisper reading the video file directly (PyAV decodes the
audio track, so no system ffmpeg). If the model cannot load we fall back to an
.srt/.vtt sidecar, which carries real timestamps. Both produce identical
`Utterance` objects, so nothing downstream changes.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import ASR_COMPUTE, ASR_MODEL
from .schemas import Span, Utterance

_model_cache: dict[tuple[str, str], object] = {}


def _load_model(name: str, compute: str):
    key = (name, compute)
    if key not in _model_cache:
        from faster_whisper import WhisperModel

        _model_cache[key] = WhisperModel(name, device="cpu", compute_type=compute)
    return _model_cache[key]


def transcribe(
    path: Path,
    model_name: str = ASR_MODEL,
    compute: str = ASR_COMPUTE,
) -> tuple[list[Utterance], str]:
    """Return (utterances, provider_label). Never raises for missing speech."""
    sidecar = _find_sidecar(path)

    try:
        model = _load_model(model_name, compute)
        segments, _info = model.transcribe(
            str(path),
            word_timestamps=True,
            # Silence/static with no real speech otherwise gets decoded anyway:
            # the model falls back to higher sampling temperatures on low-confidence
            # audio and invents a plausible sentence, a different one each run. VAD
            # filtering strips non-speech before it ever reaches the decoder, and
            # disabling conditioning on previous text stops one hallucinated phrase
            # from being echoed into the next segment.
            vad_filter=True,
            condition_on_previous_text=False,
        )
        utterances = _from_whisper(segments)
        if utterances:
            return utterances, f"faster-whisper:{model_name}"
        # A video with no speech is valid, not an error.
        if sidecar is None:
            return [], f"faster-whisper:{model_name}"
    except Exception as exc:  # noqa: BLE001 - degrade loudly, never crash the run
        print(f"[asr] faster-whisper unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to sidecar")

    if sidecar is not None:
        return parse_sidecar(sidecar), f"sidecar:{sidecar.suffix.lstrip('.')}"
    return [], "none"


def _from_whisper(segments) -> list[Utterance]:
    from .schemas import Word

    out: list[Utterance] = []
    for i, seg in enumerate(segments):
        text = (seg.text or "").strip()
        if not text:
            continue
        words = [
            Word(text=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3))
            for w in (seg.words or [])
            if w.word and w.word.strip()
        ]
        # avg_logprob is a log probability; map it into a readable 0..1 band.
        conf = getattr(seg, "avg_logprob", None)
        confidence = 1.0 if conf is None else max(0.0, min(1.0, 1.0 + conf / 5.0))
        out.append(Utterance(
            id=f"u{i:04d}",
            span=Span(start=round(seg.start, 3), end=round(seg.end, 3)),
            text=text,
            confidence=round(confidence, 3),
            words=words,
        ))
    return out


# --------------------------------------------------------------------------- #
# sidecar fallback
# --------------------------------------------------------------------------- #
def _find_sidecar(video: Path) -> Path | None:
    for ext in (".srt", ".vtt"):
        candidate = video.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


_TS = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)


def parse_sidecar(path: Path) -> list[Utterance]:
    """Parse .srt/.vtt into utterances with absolute timestamps."""
    def to_seconds(h: str, m: str, s: str, ms: str) -> float:
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[Utterance] = []
    i = 0
    while i < len(lines):
        match = _TS.search(lines[i])
        if not match:
            i += 1
            continue
        g = match.groups()
        start, end = to_seconds(*g[:4]), to_seconds(*g[4:])
        i += 1
        buf: list[str] = []
        while i < len(lines) and lines[i].strip() and not _TS.search(lines[i]):
            buf.append(lines[i].strip())
            i += 1
        text = " ".join(buf).strip()
        if text:
            out.append(Utterance(
                id=f"u{len(out):04d}",
                span=Span(start=round(start, 3), end=round(end, 3)),
                text=text,
            ))
    return out


# --------------------------------------------------------------------------- #
# derived helpers used by the boundary scorer
# --------------------------------------------------------------------------- #
def silence_gaps(utterances: list[Utterance]) -> list[tuple[float, float]]:
    """(midpoint, gap_seconds) for every pause between consecutive utterances."""
    gaps: list[tuple[float, float]] = []
    for prev, nxt in zip(utterances, utterances[1:]):
        gap = nxt.span.start - prev.span.end
        if gap > 0:
            gaps.append(((prev.span.end + nxt.span.start) / 2.0, gap))
    return gaps


def utterance_edges(utterances: list[Utterance]) -> list[float]:
    """Candidate snap targets: every utterance start, plus the final end."""
    if not utterances:
        return []
    edges = [u.span.start for u in utterances]
    edges.append(utterances[-1].span.end)
    return sorted(set(edges))


def text_between(utterances: list[Utterance], start: float, end: float) -> str:
    return " ".join(
        u.text for u in utterances if u.span.start < end and u.span.end > start
    ).strip()
