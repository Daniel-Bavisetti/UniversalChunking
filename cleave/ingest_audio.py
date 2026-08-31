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
from pathlib import Path

import httpx

from .config import settings
from .http import request_with_retry
from .ingest_document import IngestResult
from .models import ContentElement, sha256_of

log = logging.getLogger(__name__)

#: Transcription of a long recording is genuinely slow, so this is generous.
TIMEOUT_S = 600


class STTUnavailable(RuntimeError):
    """The STT worker did not answer.

    It runs as a separate process (Docling pins ``transformers`` below the
    version the audio stack needs), so this is an environment problem rather
    than a problem with the file — and the message says so, because the old
    behaviour was a raw ``ConnectError`` that read like a corrupt upload.
    """


def ingest_audio(path: str | Path) -> IngestResult:
    path = Path(path)
    warnings: list[str] = []

    url = settings().stt_url
    # Read the bytes once rather than streaming the handle: a retry has to
    # send the body again, and a rewound file object is easy to get wrong.
    # Uploads are capped at 50MB, so holding one is acceptable.
    payload = path.read_bytes()
    try:
        resp = request_with_retry(
            "POST",
            f"{url}/api/transcribe/sync",
            files={"file": (path.name, payload)},
            data={"options": json.dumps({"diarize": True}), "format": "json"},
            timeout=TIMEOUT_S,
            attempts=2,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise STTUnavailable(
            f"STT worker at {url} did not respond ({exc}); start it, or drop the audio file from this job"
        ) from exc
    try:
        data = resp.json()
    except ValueError as exc:
        raise STTUnavailable(
            f"STT worker at {url} returned a non-JSON reply ({exc})"
        ) from exc
    if data.get("error"):
        raise RuntimeError(f"STT worker error: {data['error']}")

    result = data.get("result") or data
    segments = result.get("segments") or []
    if not segments:
        warnings.append("transcript came back with no segments")

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
