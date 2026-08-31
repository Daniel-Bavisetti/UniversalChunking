"""Video ingestion: connects video files to the external video/multimodal worker.

When a video file (.mp4, .mov, .mkv, .webm, .avi) is uploaded, this module
coordinates with the video processing worker or audio/STT worker to extract
timestamped speech segments, visual summaries, and scene transitions,
normalizing them into ContentElements.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from .config import settings
from .http import request_with_retry
from .ingest_document import IngestResult
from .models import ContentElement, sha256_of

log = logging.getLogger(__name__)

TIMEOUT_S = 900


class VideoWorkerUnavailable(RuntimeError):
    """The video/multimodal worker did not answer."""


def ingest_video(path: str | Path) -> IngestResult:
    """Ingest a video file by requesting multimodal extraction from the video/STT worker."""
    path = Path(path)
    warnings: list[str] = []
    cfg = settings()
    payload = path.read_bytes()

    # 1. Try dedicated video worker endpoint first
    video_url = cfg.video_url
    try:
        resp = request_with_retry(
            "POST",
            f"{video_url}/api/video/process",
            files={"file": (path.name, payload)},
            data={"options": json.dumps({"diarize": True, "extract_visuals": True, "ocr": True})},
            timeout=TIMEOUT_S,
            attempts=1,
        )
        if resp.status_code == 200:
            data = resp.json()
            elements = [_element_from_video_dict(d, i) for i, d in enumerate(data.get("elements", []))]
            if elements:
                log.info("ingest_video: %d elements from video worker for %s", len(elements), path.name)
                return IngestResult(
                    elements=elements,
                    title=data.get("title") or path.stem,
                    source_uri=str(path),
                    sha256=sha256_of(str(path)),
                    warnings=warnings,
                )
    except Exception as exc:
        log.debug("dedicated video worker at %s not available: %s", video_url, exc)

    # 2. Fall back to STT worker (extract audio transcript from video container)
    stt_url = cfg.stt_url
    try:
        resp = request_with_retry(
            "POST",
            f"{stt_url}/api/transcribe/sync",
            files={"file": (path.name, payload)},
            data={"options": json.dumps({"diarize": True}), "format": "json"},
            timeout=TIMEOUT_S,
            attempts=2,
        )
        resp.raise_for_status()
        stt_data = resp.json()
        elements = []
        for i, seg in enumerate(stt_data.get("segments", [])):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            elements.append(ContentElement(
                id=f"v_seg_{i:04d}",
                kind="speech_segment",
                text=text,
                t0=float(seg.get("start", 0.0)),
                t1=float(seg.get("end", 0.0)),
                speaker=seg.get("speaker"),
                meta={"source": "video_audio_track"},
            ))
        if elements:
            warnings.append("processed video audio track via STT worker (visual analysis inactive)")
            return IngestResult(
                elements=elements,
                title=path.stem,
                source_uri=str(path),
                sha256=sha256_of(str(path)),
                warnings=warnings,
            )
    except httpx.HTTPError as exc:
        raise VideoWorkerUnavailable(
            f"Video worker ({video_url}) and STT worker ({stt_url}) did not respond ({exc}). "
            "To process video files, ensure the video or STT worker is running, or upload a video contract JSON."
        ) from exc

    raise VideoWorkerUnavailable(
        f"Could not extract video or audio elements from {path.name}. "
        "Ensure the video file contains a valid audio/visual stream."
    )


def _element_from_video_dict(d: dict[str, Any], i: int) -> ContentElement:
    return ContentElement(
        id=str(d.get("id") or f"v_el_{i:04d}"),
        kind=str(d.get("kind") or "speech_segment"),
        text=str(d.get("text") or "").strip(),
        t0=d.get("t0"),
        t1=d.get("t1"),
        speaker=d.get("speaker"),
        meta=d.get("meta") or {},
    )
