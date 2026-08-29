"""Cleave web app: upload → background job → routed, explainable knowledge units.

Deliberately small: a module-level job dict + FastAPI BackgroundTasks. Jobs run
in seconds and the artifacts land on disk as JSON; nothing here needs a queue,
a database, or cancellation.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
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
    filename: str                      # display summary, e.g. "3 files (a.pdf, b.mp3, …)"
    filenames: list[str] = field(default_factory=list)
    status: str = "queued"            # queued | running | done | error
    progress: int = 0
    message: str = "queued"
    error: str | None = None
    created: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    use_llm: bool = True               # per-job override for LLM enrichment

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "error")

    @property
    def dir(self) -> Path:
        return DATA / self.id


JOBS: dict[str, Job] = {}


def _display_name(names: list[str]) -> str:
    """One-line summary for a job's input files, for job lists and titles."""
    if not names:
        return "upload"
    if len(names) == 1:
        return names[0]
    shown = ", ".join(names[:3])
    more = f", +{len(names) - 3} more" if len(names) > 3 else ""
    return f"{len(names)} files ({shown}{more})"


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
        inputs = sorted((profile.parent / "input").glob("*"))
        names = [p.name for p in inputs]
        JOBS[job_id] = Job(
            id=job_id,
            filename=_display_name(names) if names else (meta.get("title") or job_id),
            filenames=names,
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


def run_job(job_id: str, input_paths: list[Path]) -> None:
    """Process every input file through its own ingest → graph → chunk →
    enrich pipeline, then merge the results into one job.

    Each file keeps the routing decision that fits it — a spreadsheet and a
    PDF in the same batch still get tabular and structural routes respectively
    — they are simply reported together and searched together. Element and
    unit ids are namespaced per file (see ``_prefix_*``) so nothing collides
    when the results are combined.
    """
    job = JOBS[job_id]
    job.status = "running"
    t0 = time.time()
    n = len(input_paths)
    try:
        from .usage import Ledger  # noqa: PLC0415

        ledger = Ledger()
        all_units: list = []
        files_meta: list[dict] = []
        graph_nodes: list = []
        graph_edges: list = []
        enrichments: list[dict] = []

        for i, input_path in enumerate(input_paths):
            lo, hi = 5 + int(85 * i / n), 5 + int(85 * (i + 1) / n)

            def progress(frac: float, msg: str, lo=lo, hi=hi) -> None:
                _set(job, lo + int((hi - lo) * min(1.0, max(0.0, frac))), msg)

            units, file_meta, nodes, edges, enrichment = _process_file(
                job, input_path, prefix=f"f{i}_", ledger=ledger, progress=progress)
            all_units.extend(units)
            files_meta.append(file_meta)
            graph_nodes.extend(nodes)
            graph_edges.extend(edges)
            if enrichment:
                enrichments.append(enrichment)

        _set(job, 90, f"{len(all_units)} knowledge units — writing artifacts…")
        graph = {"nodes": graph_nodes, "edges": graph_edges} if (graph_nodes or graph_edges) else None
        _write_artifacts(job, all_units, files_meta, graph, t0,
                         ledger=ledger, enrichments=enrichments)

        job.elapsed_s = round(time.time() - t0, 1)
        job.progress, job.message, job.status = 100, "done", "done"
    except Exception as exc:  # surface honestly; never a silent dead job
        log.exception("job %s failed", job_id)
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "failed"
        job.status = "error"


def _prefix_element(e, prefix: str) -> None:
    """Namespace one element's id/parent_id/caption_ids so a second file's
    ``el_0000`` never collides with the first's."""
    e.id = prefix + e.id
    if e.parent_id:
        e.parent_id = prefix + e.parent_id
    if e.meta.get("caption_ids"):
        e.meta["caption_ids"] = [prefix + c for c in e.meta["caption_ids"]]


def _prefix_unit(u, prefix: str) -> None:
    """Namespace one unit's id and every relationship target it carries —
    all relationship targets in this pipeline are other units from the same
    file, so a uniform prefix keeps them internally consistent."""
    u.id = prefix + u.id
    for r in u.relationships:
        r.target_id = prefix + r.target_id


def _process_file(job: Job, input_path: Path, *, prefix: str, ledger, progress) -> tuple:
    """Run the single-file pipeline for one upload in a (possibly multi-file)
    job. Returns (units, file_meta, graph_nodes, graph_edges, enrichment)."""
    suffix = input_path.suffix.lower()
    filename = input_path.name
    progress(0.0, f"understanding {filename}…")

    if suffix == ".json":
        from .ingest_contract import load_contract  # noqa: PLC0415

        progress(0.1, f"importing contract payload… ({filename})")
        imported, ready_units = load_contract(input_path)
        if ready_units:
            for u in ready_units:
                _prefix_unit(u, prefix)
            file_meta = {
                "filename": filename,
                "title": ready_units[0].context.document_title,
                "source": str(input_path),
                "warnings": [],
                "profile": {
                    "route": "imported",
                    "route_reason": "knowledge units produced by an external modality "
                                    "worker and imported through the contract",
                },
                "cleaning": None,
            }
            progress(1.0, f"done ({filename})")
            return ready_units, file_meta, [], [], None
        ingest = imported
    elif suffix in _AUDIO_EXTS:
        from .ingest_audio import ingest_audio  # noqa: PLC0415

        progress(0.1, f"transcribing (STT worker)… ({filename})")
        ingest = ingest_audio(input_path)
    else:
        from .ingest_document import ingest_document  # noqa: PLC0415

        progress(0.1, f"parsing structure (Docling)… ({filename})")
        ingest = ingest_document(input_path)

    for e in ingest.elements:
        _prefix_element(e, prefix)

    progress(0.5, f"{len(ingest.elements)} elements — building context graph… ({filename})")
    from .chunkers import chunk  # noqa: PLC0415
    from .graph import ContextGraph  # noqa: PLC0415

    graph = ContextGraph(ingest.elements)
    progress(0.6, f"routing and chunking… ({filename})")
    units, profile = chunk(ingest, graph)
    for u in units:
        _prefix_unit(u, prefix)

    enrichment: dict | None = None
    flagged_n = sum(1 for u in units if u.decision.escalation_flags)
    if flagged_n:
        from .enrich import enrich  # noqa: PLC0415

        progress(0.7, (f"enriching {flagged_n} context-poor chunks… ({filename})"
                       if job.use_llm else f"skipping LLM enrichment ({filename})…"))
        doc_text = "\n\n".join(e.text for e in ingest.elements if e.text)
        enrichment = enrich(
            units, doc_text, ledger=ledger, use_llm=job.use_llm,
            progress=lambda i, n2: progress(
                0.7 + 0.25 * i / max(1, n2), f"enriching… {i}/{n2} ({filename})"))

    file_meta = {
        "filename": filename, "title": ingest.title, "source": ingest.source_uri,
        "warnings": ingest.warnings, "profile": profile.to_dict(), "cleaning": ingest.cleaning,
    }
    progress(1.0, f"done ({filename})")
    gd = graph.to_dict()
    return units, file_meta, gd["nodes"], gd["edges"], enrichment


def _merge_cleaning(cleanings: list[dict | None]) -> dict | None:
    total_fixes = elements_changed = chars_removed = 0
    by_rule: Counter = Counter()
    for c in cleanings:
        if not c:
            continue
        total_fixes += c.get("total_fixes", 0)
        elements_changed += c.get("elements_changed", 0)
        chars_removed += c.get("chars_removed", 0)
        by_rule.update(c.get("by_rule", {}))
    if not total_fixes:
        return None
    return {"total_fixes": total_fixes, "elements_changed": elements_changed,
            "chars_removed": chars_removed, "by_rule": dict(by_rule.most_common())}


def _merge_enrichment(parts: list[dict]) -> dict | None:
    if not parts:
        return None
    merged = dict(parts[-1])
    for k in ("flagged", "enriched", "api_calls", "calls_saved_by_batching"):
        merged[k] = sum(p.get(k, 0) for p in parts)
    return merged


def _write_artifacts(job: Job, units, files_meta: list[dict], graph: dict | None, t0: float, *,
                     ledger=None, enrichments: list[dict] | None = None) -> None:
    """units.json + graph.json + profile.json for one job.

    A single-file job writes the same flat schema this always has (``profile``,
    ``cleaning``, ``title``, ``source`` at the top level). A multi-file job adds
    a ``files`` array with one entry per input, alongside combined totals so
    existing single-file consumers of profile.json see nothing new.
    """
    from .usage import append_to_cumulative  # noqa: PLC0415

    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "units.json").write_text(
        json.dumps([u.to_dict() for u in units], indent=1))
    if graph is not None:
        (job.dir / "graph.json").write_text(json.dumps(graph, indent=1))

    usage = ledger.to_dict() if ledger is not None else None
    warnings = [w if len(files_meta) == 1 else f"{f['filename']}: {w}"
               for f in files_meta for w in f["warnings"]]
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
        "use_llm": job.use_llm,
    }

    record: dict = {"totals": totals}
    if len(files_meta) == 1:
        fm = files_meta[0]
        record["profile"] = fm["profile"]
        record["title"] = fm["title"]
        record["source"] = fm["source"]
        if fm["cleaning"]:
            record["cleaning"] = fm["cleaning"]
    else:
        record["files"] = files_meta
        record["title"] = job.filename
        combined_cleaning = _merge_cleaning([f["cleaning"] for f in files_meta])
        if combined_cleaning:
            record["cleaning"] = combined_cleaning
    if usage:
        record["usage"] = usage
    merged_enrichment = _merge_enrichment(enrichments or [])
    if merged_enrichment:
        record["enrichment"] = merged_enrichment
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
        "job": job, "units": units, "profile": meta.get("profile"),
        "totals": meta["totals"], "title": meta.get("title"), "graph": graph,
        "usage": meta.get("usage"), "enrichment": meta.get("enrichment"),
        "cleaning": meta.get("cleaning"), "files": meta.get("files"),
    })


# ───────── api ─────────

@app.post("/api/jobs")
async def create_job(background: BackgroundTasks, files: list[UploadFile] = File(...),
                     use_llm: str = Form("true")):
    job_id = uuid.uuid4().hex[:10]
    dest_dir = DATA / job_id / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    dest_paths: list[Path] = []
    try:
        for i, f in enumerate(files):
            fname = f.filename or f"upload{i}"
            suffix = Path(fname).suffix.lower()
            if suffix not in _DOC_EXTS | _AUDIO_EXTS | _CONTRACT_EXTS:
                raise HTTPException(415, f"unsupported type {suffix!r} ({fname})")
            dest = dest_dir / fname
            if dest.exists():  # two files with the same name in one batch
                dest = dest_dir / f"{dest.stem}_{i}{dest.suffix}"
            written = 0
            with dest.open("wb") as out:
                while chunk_bytes := await f.read(1 << 20):
                    written += len(chunk_bytes)
                    if written > MAX_UPLOAD:
                        raise HTTPException(413, f"{fname} exceeds 50MB")
                    out.write(chunk_bytes)
            names.append(dest.name)
            dest_paths.append(dest)
        if not dest_paths:
            raise HTTPException(400, "no files uploaded")
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    job = Job(id=job_id, filename=_display_name(names), filenames=names,
             use_llm=use_llm.lower() not in ("false", "0", "off", "no"))
    JOBS[job_id] = job
    background.add_task(run_job, job_id, dest_paths)
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
