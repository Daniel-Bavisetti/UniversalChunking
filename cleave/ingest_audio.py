"""Audio ingestion (stretch): one HTTP call to the local STT worker.

Deliberately minimal — Cleave does not do audio intelligence of its own. The
STT worker (a separate venv: Docling pins transformers<5.9 on macOS, the MLX
audio stack needs >=5.14) returns timestamped, speaker-attributed segments;
they normalize into the same ContentElements as everything else and the
temporal chunker takes it from there.
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


def _from_local(path: Path) -> list[dict]:
    """In-process transcription with the video engine's own ASR.

    faster-whisper bundles PyAV, so it decodes an audio file directly with no
    system ffmpeg, and CTranslate2 means no torch. That makes a meeting
    recording work out of the box rather than depending on a second service
    someone remembered to start — which is the difference between a demo that
    runs and a demo that 500s.
    """
    from vke.asr import transcribe  # noqa: PLC0415
    from vke.diarize import apply_to_utterances, diarize  # noqa: PLC0415

    utterances, _provider = transcribe(path)
    try:
        turns = diarize(path, utterances)
        utterances = apply_to_utterances(utterances, turns)
    except Exception as exc:      # a heuristic must never fail the job
        log.info("diarization skipped (%s: %s)", type(exc).__name__, exc)
    return [
        {"text": u.text, "start": u.start, "end": u.end,
         "speaker": getattr(u, "speaker", None)}
        for u in utterances
    ]


def ingest_audio(path: str | Path) -> IngestResult:
    """Transcribe a recording, preferring whichever engine is actually there.

    A meeting, an interview or a lecture becomes timestamped, speaker-attributed
    segments; the temporal chunker then makes speaker turns the chunk
    boundaries, so nobody's words are ever attributed to another person.
    """
    path = Path(path)
    warnings: list[str] = []

    try:
        segments = _from_worker(path)
        source = "STT worker"
    except Exception as exc:
        log.info("STT worker unavailable (%s) — using the local ASR engine",
                 type(exc).__name__)
        warnings.append(
            f"external STT worker unavailable ({type(exc).__name__}); "
            "transcribed locally with faster-whisper instead")
        segments = _from_local(path)
        source = "local faster-whisper"

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
