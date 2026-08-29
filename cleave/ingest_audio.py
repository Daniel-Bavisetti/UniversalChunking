"""Audio ingestion: meetings, interviews, lectures, voice notes.

A recording becomes timestamped, speaker-attributed segments; the temporal
chunker downstream turns speaker turns into chunk boundaries, so nobody's
words are ever attributed to someone else.

Three engines, tried in an order that reflects what they cost:

  1. **External STT worker** — an HTTP service, when one is running. Preferred
     because it may be a bigger model on better hardware than this machine.
  2. **mlx-whisper** (vendored meetgraph engine selection) — Apple-Silicon GPU
     transcription. On an M-series Mac this is the difference between the GPU
     doing the work and a pinned CPU core doing it for minutes.
  3. **faster-whisper** (the video engine's ASR) — CPU, works everywhere.

Speaker labels come from meetgraph's Resemblyzer path: a learned voice
embedding per utterance, clustered online by nearest centroid. A learned
embedding separates similar voices far better than spectral features; when
Resemblyzer is not installed, the video engine's clustering diarizer is the
fallback, and if that also fails the transcript arrives unattributed rather
than the job failing.

Every choice is recorded — the engine label lands in the log and the fallback
reasons land in the ingest warnings, so a transcript with no speakers can
always explain itself.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

from .ingest_document import IngestResult
from .models import ContentElement, sha256_of

log = logging.getLogger(__name__)

STT_URL = os.environ.get("CLEAVE_STT_URL", "http://127.0.0.1:8000")
TIMEOUT_S = 600

#: auto — worker when reachable, local otherwise. local — never call the worker.
ASR_ENGINE = os.environ.get("CLEAVE_ASR_ENGINE", "auto").lower()
#: Whisper size preset (tiny/base/small/medium/large-v3) or an HF repo id.
ASR_MODEL = os.environ.get("CLEAVE_ASR_MODEL", "base")


def _from_worker(path: Path) -> list[dict]:
    """The external STT service, when one is running."""
    with path.open("rb") as f:
        resp = httpx.post(
            f"{STT_URL}/api/transcribe/sync",
            files={"file": (path.name, f)},
            data={"options": json.dumps({"diarize": True}), "format": "json"},
            timeout=TIMEOUT_S,
        )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"STT worker error: {data['error']}")
    result = data.get("result") or data
    return result.get("segments") or []


# ───────── local engines (vendored meetgraph + the video engine) ─────────

def _transcribe_mlx(audio, model_name: str) -> list[dict]:
    """Apple-Silicon GPU transcription, timestamps included.

    meetgraph's ``MLXWhisperTranscriber`` returns plain text because its app
    segments audio upstream; we need the segment timestamps, so this calls
    mlx-whisper directly and borrows only the preset→repo mapping.
    """
    import mlx_whisper  # noqa: PLC0415 — Apple-only, present via the extra

    from meetgraph.transcribe import MLXWhisperTranscriber  # noqa: PLC0415

    repo = (model_name if "/" in model_name
            else MLXWhisperTranscriber._PRESET.get(model_name,
                                                   MLXWhisperTranscriber._PRESET["base"]))
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo)
    return [
        {"text": (s.get("text") or "").strip(),
         "start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0))}
        for s in result.get("segments", [])
        if (s.get("text") or "").strip()
    ]


def _transcribe_cpu(path: Path) -> tuple[list[dict], str]:
    """faster-whisper via the video engine — CPU, no torch, decodes directly."""
    from vke.asr import transcribe  # noqa: PLC0415

    utterances, provider = transcribe(path, model_name=ASR_MODEL)
    return [
        {"text": u.text, "start": u.span.start, "end": u.span.end}
        for u in utterances
    ], provider


def _label_speakers(segments: list[dict], audio, rate: int,
                    path: Path, warnings: list[str]) -> str:
    """Attach a speaker label to each segment, best backend first.

    Returns the backend label used ("" when the transcript stays unattributed).
    Mutates ``segments`` in place; a segment that cannot be labelled keeps
    ``speaker=None`` rather than inheriting its neighbour's voice.
    """
    # 1 — Resemblyzer voice embeddings (vendored meetgraph)
    try:
        from meetgraph.diarize import SpeakerLabeler  # noqa: PLC0415

        labeler = SpeakerLabeler()
        if labeler.available and audio is not None and len(audio):
            for s in segments:
                lo, hi = int(s["start"] * rate), int(s["end"] * rate)
                s["speaker"] = labeler.label(audio[lo:hi], rate)
            return f"resemblyzer ({labeler.backend})"
        warnings.append("Resemblyzer voice embeddings unavailable — "
                        "falling back to spectral clustering")
    except Exception as exc:
        warnings.append(f"Resemblyzer diarization failed ({type(exc).__name__}) — "
                        "falling back to spectral clustering")

    # 2 — the video engine's clustering diarizer
    try:
        from vke.diarize import apply_to_utterances, diarize  # noqa: PLC0415
        from vke.schemas import Span, Utterance  # noqa: PLC0415

        utts = [Utterance(id=f"u{i:04d}", span=Span(start=s["start"], end=s["end"]),
                          text=s["text"])
                for i, s in enumerate(segments)]
        utts = apply_to_utterances(utts, diarize(path, utts))
        for s, u in zip(segments, utts):
            s["speaker"] = u.speaker
        return "vke spectral clustering"
    except Exception as exc:
        warnings.append(f"diarization unavailable ({type(exc).__name__}) — "
                        "transcript is unattributed; chunking falls back to pauses")
        return ""


def _from_local(path: Path, warnings: list[str]) -> tuple[list[dict], str]:
    """In-process transcription. → (segments, engine label)."""
    from meetgraph.transcribe import resolve_device  # noqa: PLC0415
    from vke.diarize import load_audio  # noqa: PLC0415

    audio, rate = load_audio(path)

    segments: list[dict] = []
    engine = ""
    if resolve_device("auto") == "mlx" and len(audio):
        try:
            segments = _transcribe_mlx(audio, ASR_MODEL)
            engine = f"mlx-whisper:{ASR_MODEL} (Apple GPU)"
        except Exception as exc:
            warnings.append(f"MLX transcription failed ({type(exc).__name__}) — "
                            "using faster-whisper on CPU")
    if not segments:
        segments, engine = _transcribe_cpu(path)

    if segments:
        backend = _label_speakers(segments, audio, rate, path, warnings)
        if backend:
            engine += f" · speakers via {backend}"
    return segments, engine


def ingest_audio(path: str | Path) -> IngestResult:
    """Transcribe a recording, preferring whichever engine is actually there."""
    path = Path(path)
    warnings: list[str] = []

    segments: list[dict] = []
    source = ""
    if ASR_ENGINE != "local":
        try:
            segments = _from_worker(path)
            source = "STT worker"
        except Exception as exc:
            log.info("STT worker unavailable (%s) — transcribing locally",
                     type(exc).__name__)
            if ASR_ENGINE == "worker":
                raise
    if not source:
        segments, source = _from_local(path, warnings)

    if not segments:
        warnings.append("transcript came back with no segments")
    log.info("transcribed %s via %s: %d segment(s)", path.name, source, len(segments))

    elements: list[ContentElement] = []
    for i, s in enumerate(segments):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        elements.append(ContentElement(
            id=f"el_{i:04d}",
            kind="speech_segment",
            text=text,
            t0=float(s.get("start", 0.0)),
            t1=float(s.get("end", 0.0)),
            speaker=s.get("speaker"),
            meta={k: s[k] for k in ("language", "avg_logprob") if s.get(k) is not None},
        ))

    from .cleaning import clean_elements  # noqa: PLC0415

    report = clean_elements(elements)
    elements = [e for e in elements if e.text]

    # Tier-1 meeting semantics: pattern-label questions, decisions and action
    # items while every utterance still has its own speaker and timestamps.
    from .meeting import annotate_elements  # noqa: PLC0415

    annotate_elements(elements)

    speakers = {e.speaker for e in elements if e.speaker}
    log.info("transcribed %s: %d segments, %d speaker(s) — %s",
             path.name, len(elements), len(speakers), report.summary())
    return IngestResult(
        elements=elements,
        title=path.stem,
        source_uri=str(path),
        sha256=sha256_of(str(path)),
        warnings=warnings,
        cleaning=report.to_dict(),
    )
