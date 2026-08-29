"""FastAPI application: upload, job status, units, curves, media.

Byte-range streaming is the one piece that absolutely must work: without correct
206 responses the browser cannot seek, and jump-to-moment is the demo.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import CONFIGS, DEFAULT_CONFIG, UPLOAD_DIR, ensure_dirs
from . import graph as graph_mod
from . import providers
from .pipeline import process_video
from .schemas import Job
from .retrieve import UnitIndex, ask, search
from .store import VideoStore, list_videos
from .universal import export_jsonl, export_units

app = FastAPI(title="VKE - Video Knowledge Engine")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# In-process job registry. A dict plus asyncio is the right amount of machinery
# for a single-user demo; a queue would be infrastructure with no demo value.
JOBS: dict[str, Job] = {}


@app.on_event("startup")
async def _startup() -> None:
    ensure_dirs()


# --------------------------------------------------------------------------- #
# configs + library
# --------------------------------------------------------------------------- #
@app.get("/api/configs")
async def get_configs() -> list[dict]:
    return [
        {
            "key": c.key,
            "label": c.label,
            "description": c.description,
            "weights": c.weights,
            "fixed_window": c.fixed_window,
            "is_default": c.key == DEFAULT_CONFIG,
        }
        for c in CONFIGS.values()
    ]


@app.get("/api/videos")
async def get_videos() -> list[dict]:
    out = []
    for meta in list_videos():
        store = VideoStore(meta.video_id)
        units = store.load_units()
        out.append({
            **meta.model_dump(),
            "processed": store.is_processed,
            "unit_counts": {k: len(v) for k, v in units.items()},
        })
    return out


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str) -> dict:
    store = VideoStore(video_id)
    meta = store.load_meta()
    if meta is None:
        raise HTTPException(404, "video not found")
    units = store.load_units()
    return {
        **meta.model_dump(),
        "processed": store.is_processed,
        "unit_counts": {k: len(v) for k, v in units.items()},
    }


@app.get("/api/videos/{video_id}/units")
async def get_units(video_id: str, config: str | None = None) -> dict:
    units = VideoStore(video_id).load_units()
    if not units:
        raise HTTPException(404, "no units; has the video been processed?")
    if config:
        if config not in units:
            raise HTTPException(404, f"unknown config '{config}'")
        return {config: [u.model_dump() for u in units[config]]}
    return {k: [u.model_dump() for u in v] for k, v in units.items()}


@app.get("/api/videos/{video_id}/curves")
async def get_curves(video_id: str) -> dict:
    curves = VideoStore(video_id).load_curves()
    if curves is None:
        raise HTTPException(404, "no curves; has the video been processed?")
    return curves


def _graph_for(video_id: str) -> graph_mod.Graph | None:
    blob = VideoStore(video_id).load_graph()
    if not blob:
        return None
    g = graph_mod.Graph()
    for n in blob["nodes"]:
        data = {k: v for k, v in n.items() if k not in ("id", "type", "label")}
        g.add_node(graph_mod.Node(n["id"], n["type"], n["label"], data))
    for e in blob["edges"]:
        g.add_edge(e["source"], e["target"], e["type"], e.get("weight", 1.0))
    return g


@app.get("/api/videos/{video_id}/graph")
async def get_graph(video_id: str) -> dict:
    blob = VideoStore(video_id).load_graph()
    if blob is None:
        raise HTTPException(404, "no graph; has the video been processed?")
    return blob


@app.get("/api/videos/{video_id}/trace")
async def get_trace(video_id: str) -> dict:
    store = VideoStore(video_id)
    traces = store.load_traces()
    extraction = store.load_extraction()
    return {
        "video_id": video_id,
        "stages": [t.model_dump() for t in traces],
        "total_seconds": round(sum(t.seconds for t in traces), 2),
        "providers": extraction[3] if extraction else {},
        "usage": providers.USAGE.as_dict(),
    }


@app.post("/api/videos/{video_id}/search")
async def post_search(video_id: str, body: dict) -> dict:
    query = (body or {}).get("query", "").strip()
    config = (body or {}).get("config") or DEFAULT_CONFIG
    top_k = int((body or {}).get("top_k", 6))
    if not query:
        raise HTTPException(400, "query is required")

    units = VideoStore(video_id).load_units().get(config, [])
    if not units:
        raise HTTPException(404, f"no units for config '{config}'")
    hits = search(query, units, graph=_graph_for(video_id), top_k=top_k)
    return {"query": query, "config": config,
            "results": [h.model_dump() for h in hits]}


@app.post("/api/videos/{video_id}/ask")
async def post_ask(video_id: str, body: dict) -> dict:
    question = (body or {}).get("question", "").strip()
    config = (body or {}).get("config") or DEFAULT_CONFIG
    if not question:
        raise HTTPException(400, "question is required")

    units = VideoStore(video_id).load_units().get(config, [])
    if not units:
        raise HTTPException(404, f"no units for config '{config}'")
    return ask(question, units, graph=_graph_for(video_id),
               llm=providers.get_llm())


@app.get("/api/videos/{video_id}/export")
async def export(video_id: str, config: str = DEFAULT_CONFIG,
                 fmt: str = "json", schema: str = "vke") -> Response:
    """Export Knowledge Units.

    schema=vke        the native video schema
    schema=universal  the platform-neutral contract a future PDF or slide
                      pipeline would also emit (see vke/universal.py)
    """
    units = VideoStore(video_id).load_units().get(config)
    if not units:
        raise HTTPException(404, f"no units for config '{config}'")

    ref = f"video://{video_id}"
    if schema == "universal":
        if fmt == "jsonl":
            body, media = export_jsonl(units, ref), "application/x-ndjson"
        else:
            body, media = json.dumps(export_units(units, ref), indent=2), "application/json"
    else:
        payload = [u.model_dump() for u in units]
        if fmt == "jsonl":
            body = "\n".join(json.dumps(u, ensure_ascii=False) for u in payload)
            media = "application/x-ndjson"
        else:
            body, media = json.dumps(payload, indent=2), "application/json"

    ext = "jsonl" if fmt == "jsonl" else "json"
    name = f"{video_id}_{config}_{schema}.{ext}"
    return Response(body, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{name}"'})


# --------------------------------------------------------------------------- #
# media
# --------------------------------------------------------------------------- #
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 1024 * 512


def _guess_type(path: Path) -> str:
    return {
        ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
        ".mkv": "video/x-matroska", ".mov": "video/quicktime",
    }.get(path.suffix.lower(), "application/octet-stream")


@app.get("/api/videos/{video_id}/stream")
async def stream(video_id: str, request: Request) -> Response:
    """Serve the source video with HTTP range support.

    Browsers issue `Range: bytes=N-` when the user scrubs. Without a correct 206
    the <video> element cannot seek, which would break jump-to-moment - the whole
    point of the demo - so this is deliberately explicit rather than delegated.
    """
    path = VideoStore(video_id).video_path()
    if path is None or not path.exists():
        raise HTTPException(404, "video file not found")

    size = path.stat().st_size
    media_type = _guess_type(path)
    header = request.headers.get("range") or request.headers.get("Range")

    if not header:
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        )

    match = _RANGE.search(header)
    if not match:
        raise HTTPException(400, "malformed Range header")

    raw_start, raw_end = match.group(1), match.group(2)
    if raw_start == "":
        # Suffix form: "bytes=-500" means the LAST 500 bytes.
        length = int(raw_end or 0)
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1

    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        )

    async def body():
        remaining = end - start + 1
        with path.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                block = fh.read(min(_CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(
        body(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Cache-Control": "no-cache",
        },
    )


@app.get("/api/videos/{video_id}/keyframe/{unit_id}.jpg")
async def keyframe(video_id: str, unit_id: str) -> Response:
    path = VideoStore(video_id).keyframe_path(unit_id)
    if not path.exists():
        raise HTTPException(404, "no keyframe")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})


# --------------------------------------------------------------------------- #
# upload + processing
# --------------------------------------------------------------------------- #
def _run_job(job: Job, video_path: Path, video_id: str) -> None:
    def progress(stage: str, percent: int, message: str) -> None:
        job.stage, job.percent, job.message = stage, percent, message

    try:
        job.status = "running"
        _meta, _units, traces = process_video(video_path, video_id, progress=progress)
        job.traces = traces
        job.status = "done"
        job.percent = 100
        job.message = "Processed"
    except Exception as exc:  # surface the failure, never leave the job hanging
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "Failed"


@app.post("/api/videos")
async def upload(file: UploadFile) -> dict:
    ensure_dirs()
    name = Path(file.filename or "video.mp4").name
    video_id = f"{Path(name).stem[:40]}_{uuid.uuid4().hex[:6]}"

    store = VideoStore(video_id)
    dest = store.dir / name
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    job = Job(job_id=uuid.uuid4().hex[:12], video_id=video_id, status="queued")
    JOBS[job.job_id] = job
    # Processing is CPU-bound, so it goes to a worker thread rather than blocking
    # the event loop. The POST returns immediately either way.
    asyncio.get_running_loop().run_in_executor(None, _run_job, job, dest, video_id)
    return {"job_id": job.job_id, "video_id": video_id}


@app.post("/api/videos/{video_id}/reprocess")
async def reprocess(video_id: str) -> dict:
    store = VideoStore(video_id)
    video = store.video_path()
    if video is None:
        raise HTTPException(404, "source video not found")
    job = Job(job_id=uuid.uuid4().hex[:12], video_id=video_id, status="queued")
    JOBS[job.job_id] = job
    asyncio.get_running_loop().run_in_executor(None, _run_job, job, video, video_id)
    return {"job_id": job.job_id, "video_id": video_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.model_dump()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>VKE</h1><p>static/index.html missing</p>", 500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
