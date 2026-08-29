"""Cleave web app: upload → background job → routed, explainable knowledge units.

Deliberately small: a module-level job dict + FastAPI BackgroundTasks. Jobs run
in seconds and the artifacts land on disk as JSON; nothing here needs a queue,
a database, or cancellation.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "jobs"
MAX_UPLOAD = 50 * 1024 * 1024

_DOC_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm", ".md", ".txt"}
_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}
_CONTRACT_EXTS = {".json"}     # payloads from external modality workers (CONTRACT.md)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _rehydrate_jobs()
    yield


app = FastAPI(title="Cleave", docs_url="/docs", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "cleave" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "cleave" / "templates")


@dataclass(slots=True)
class Job:
    id: str
    filename: str
    status: str = "queued"            # queued | running | done | error
    progress: int = 0
    message: str = "queued"
    error: str | None = None
    created: float = field(default_factory=time.time)
    elapsed_s: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "error")

    @property
    def dir(self) -> Path:
        return DATA / self.id


JOBS: dict[str, Job] = {}


def _rehydrate_jobs() -> None:
    """Re-register finished jobs from disk on startup.

    The artifacts outlive the process, so a restart should not turn a completed
    job into a 404 — the results page reads from these files anyway. Only
    terminal jobs come back: anything that was mid-flight died with the old
    process and would be a lie to show as running.
    """
    if not DATA.exists():
        return
    for profile in DATA.glob("*/profile.json"):
        job_id = profile.parent.name
        if job_id in JOBS:
            continue
        try:
            meta = json.loads(profile.read_text())
        except (OSError, ValueError):
            continue
        inputs = list((profile.parent / "input").glob("*"))
        JOBS[job_id] = Job(
            id=job_id,
            filename=inputs[0].name if inputs else (meta.get("title") or job_id),
            status="done", progress=100, message="done",
            created=profile.stat().st_mtime,
            elapsed_s=meta.get("totals", {}).get("wall_clock_s", 0.0),
        )
    if JOBS:
        log.info("restored %d completed job(s) from disk", len(JOBS))


def _set(job: Job, progress: int, message: str) -> None:
    # progress never goes backwards; terminal states are set only by the worker
    job.progress = max(job.progress, min(99, progress))
    job.message = message


def run_job(job_id: str, input_path: Path) -> None:
    job = JOBS[job_id]
    job.status = "running"
    t0 = time.time()
    try:
        suffix = input_path.suffix.lower()
        _set(job, 5, "understanding the input…")

        if suffix == ".json":
            from .ingest_contract import load_contract  # noqa: PLC0415

            _set(job, 15, "importing contract payload…")
            imported, ready_units = load_contract(input_path)
            if ready_units:
                _write_artifacts(job, ready_units, None, None, t0,
                                 title=ready_units[0].context.document_title,
                                 source=str(input_path), warnings=[])
                job.elapsed_s = round(time.time() - t0, 1)
                job.progress, job.message, job.status = 100, "done", "done"
                return
            ingest = imported
        elif suffix in _AUDIO_EXTS:
            from .ingest_audio import ingest_audio  # noqa: PLC0415

            _set(job, 15, "transcribing (STT worker)…")
            ingest = ingest_audio(input_path)
        else:
            from .ingest_document import ingest_document  # noqa: PLC0415

            _set(job, 15, "parsing structure (Docling)…")
            ingest = ingest_document(input_path)

        _set(job, 60, f"{len(ingest.elements)} elements — building context graph…")
        from .chunkers import chunk  # noqa: PLC0415
        from .graph import ContextGraph  # noqa: PLC0415

        graph = ContextGraph(ingest.elements)
        _set(job, 70, "routing and chunking…")
        units, profile = chunk(ingest, graph)

        from .usage import Ledger  # noqa: PLC0415

        ledger = Ledger()
        enrichment: dict | None = None
        flagged_n = sum(1 for u in units if u.decision.escalation_flags)
        if flagged_n:
            from .enrich import enrich  # noqa: PLC0415

            _set(job, 78, f"enriching {flagged_n} context-poor chunks (selective)…")
            doc_text = "\n\n".join(e.text for e in ingest.elements if e.text)
            enrichment = enrich(
                units, doc_text, ledger=ledger,
                progress=lambda i, n: _set(job, 78 + int(10 * i / n),
                                           f"enriching in batches… {i}/{n} call(s)"))

        _set(job, 90, f"{len(units)} knowledge units — writing artifacts…")
        _write_artifacts(job, units, profile, graph, t0,
                         title=ingest.title, source=ingest.source_uri,
                         warnings=ingest.warnings, ledger=ledger,
                         enrichment=enrichment, cleaning=ingest.cleaning)

        job.elapsed_s = round(time.time() - t0, 1)
        job.progress, job.message, job.status = 100, "done", "done"
    except Exception as exc:  # surface honestly; never a silent dead job
        log.exception("job %s failed", job_id)
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "failed"
        job.status = "error"


def _write_artifacts(job: Job, units, profile, graph, t0: float, *,
                     title: str | None, source: str, warnings: list[str],
                     ledger=None, enrichment: dict | None = None,
                     cleaning: dict | None = None) -> None:
    """units.json + graph.json + profile.json for one job.

    Imported payloads arrive without a profile or graph, so both are optional
    and the totals are computed from the units either way.
    """
    from .usage import append_to_cumulative  # noqa: PLC0415

    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "units.json").write_text(
        json.dumps([u.to_dict() for u in units], indent=1))
    if graph is not None:
        (job.dir / "graph.json").write_text(json.dumps(graph.to_dict(), indent=1))

    usage = ledger.to_dict() if ledger is not None else None
    totals = {
        "units": len(units),
        "tier0_pct": round(
            100 * sum(1 for u in units if u.context.tier == 0) / max(1, len(units))),
        "flagged_for_enrichment": sum(1 for u in units if u.decision.escalation_flags),
        "enriched": sum(1 for u in units if u.context.situating_summary),
        # Model calls are counted from the ledger, not per unit: batching means
        # one call can serve several units, and the bill follows the calls.
        "llm_calls": (usage or {}).get("totals", {}).get("calls", 0),
        "cost_usd": round((usage or {}).get("totals", {}).get("cost_usd", 0.0), 6),
        "in_tokens": (usage or {}).get("totals", {}).get("in_tokens", 0),
        "out_tokens": (usage or {}).get("totals", {}).get("out_tokens", 0),
        "wall_clock_s": round(time.time() - t0, 1),
        "warnings": warnings,
    }
    prof = profile.to_dict() if profile is not None else {
        "route": "imported",
        "route_reason": "knowledge units produced by an external modality worker "
                        "and imported through the contract",
    }
    record = {"profile": prof, "totals": totals, "title": title, "source": source}
    if usage:
        record["usage"] = usage
    if enrichment:
        record["enrichment"] = enrichment
    if cleaning:
        record["cleaning"] = cleaning
    (job.dir / "profile.json").write_text(json.dumps(record, indent=1))

    if ledger is not None and ledger.total_calls:
        append_to_cumulative(ledger, job.id)


# ───────── pages ─────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    from .llm import describe_providers  # noqa: PLC0415
    from .usage import read_cumulative  # noqa: PLC0415

    scorecard = None
    sc_path = ROOT / "data" / "scorecard.json"
    if sc_path.exists():
        scorecard = json.loads(sc_path.read_text())
    jobs = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)[:10]
    return templates.TemplateResponse(request, "index.html", {
        "jobs": jobs, "scorecard": scorecard,
        "usage": read_cumulative(), "providers": describe_providers(),
    })


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return templates.TemplateResponse(request, "job.html", {"job": job})


@app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return templates.TemplateResponse(request, "_status.html", {"job": job})


@app.get("/jobs/{job_id}/results", response_class=HTMLResponse)
def job_results(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job or job.status != "done":
        raise HTTPException(404, "no results")
    units = json.loads((job.dir / "units.json").read_text())
    meta = json.loads((job.dir / "profile.json").read_text())
    graph_path = job.dir / "graph.json"
    graph = json.loads(graph_path.read_text()) if graph_path.exists() else None
    return templates.TemplateResponse(request, "_results.html", {
        "job": job, "units": units, "profile": meta["profile"],
        "totals": meta["totals"], "title": meta.get("title"), "graph": graph,
        "usage": meta.get("usage"), "enrichment": meta.get("enrichment"),
        "cleaning": meta.get("cleaning"),
    })


# ───────── api ─────────

@app.post("/api/jobs")
async def create_job(file: UploadFile, background: BackgroundTasks):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in _DOC_EXTS | _AUDIO_EXTS | _CONTRACT_EXTS:
        raise HTTPException(415, f"unsupported type {suffix!r}")
    job_id = uuid.uuid4().hex[:10]
    job = Job(id=job_id, filename=file.filename or f"upload{suffix}")
    dest_dir = DATA / job_id / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / job.filename
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk_bytes := await file.read(1 << 20):
                written += len(chunk_bytes)
                if written > MAX_UPLOAD:
                    raise HTTPException(413, "file exceeds 50MB")
                out.write(chunk_bytes)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    JOBS[job_id] = job
    background.add_task(run_job, job_id, dest)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/api/jobs/{job_id}/units")
def api_units(job_id: str):
    return _artifact(job_id, "units.json")


@app.get("/api/jobs/{job_id}/graph")
def api_graph(job_id: str):
    return _artifact(job_id, "graph.json")


@app.get("/api/jobs/{job_id}/profile")
def api_profile(job_id: str):
    return _artifact(job_id, "profile.json")


def _artifact(job_id: str, name: str) -> JSONResponse:
    path = DATA / job_id / name
    if not path.exists():
        raise HTTPException(404, f"{name} not available")
    return JSONResponse(json.loads(path.read_text()))


#: Embedding vectors per job, keyed by job id. Bounded because a long demo
#: session would otherwise hold every job's matrix in memory for good.
_QUERY_CACHE: OrderedDict[str, tuple] = OrderedDict()
_QUERY_CACHE_MAX = 8


@app.post("/jobs/{job_id}/query", response_class=HTMLResponse)
async def query_job(request: Request, job_id: str):
    form = await request.form()
    q = str(form.get("q", "")).strip()
    path = DATA / job_id / "units.json"
    if not q or not path.exists():
        raise HTTPException(400, "nothing to search")
    from .semantic import embed  # noqa: PLC0415

    if job_id not in _QUERY_CACHE:
        units = json.loads(path.read_text())
        vecs = embed([u["embed_text"] for u in units])
        if vecs is None:
            raise HTTPException(503, "embedding model unavailable — search needs MiniLM")
        _QUERY_CACHE[job_id] = (vecs, units)
        while len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
            _QUERY_CACHE.popitem(last=False)
    _QUERY_CACHE.move_to_end(job_id)
    vecs, units = _QUERY_CACHE[job_id]
    qv = embed([q])[0]
    scored = sorted(zip((float(v @ qv) for v in vecs), units),
                    key=lambda x: -x[0])[:5]
    return templates.TemplateResponse(request, "_query.html",
                                      {"q": q, "hits": scored})


_FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#0b1120"/>'
    '<path d="M16 5v22" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round"/>'
    '<path d="M10 11h-4M10 21h-4M22 11h4M22 21h4" stroke="#5eead4" stroke-width="2.5" '
    'stroke-linecap="round"/></svg>'
)


@app.get("/favicon.svg")
def favicon():
    return Response(_FAVICON, media_type="image/svg+xml")


@app.get("/api/usage")
def api_usage():
    """Install-wide token and cost ledger, broken down by model."""
    from .llm import describe_providers  # noqa: PLC0415
    from .usage import read_cumulative  # noqa: PLC0415

    return {"cumulative": read_cumulative(), "providers": describe_providers()}


@app.get("/health")
def health():
    from .llm import get_provider  # noqa: PLC0415

    p = get_provider()
    return {"ok": True, "jobs": len(JOBS), "llm": p.name, "model": p.model}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("cleave.app:app", host="127.0.0.1", port=8321, reload=False)
