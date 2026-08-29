"""Cleave web app: upload → background job → routed, explainable knowledge units.

Deliberately small: a module-level job dict + FastAPI BackgroundTasks. Jobs run
in seconds and the artifacts land on disk as JSON; nothing here needs a queue,
a database, or cancellation.
"""

from __future__ import annotations

import json
import logging
import re
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

# Extension tables only — neither module pulls a heavy dependency at import
# time; the engines behind them load lazily when a matching file arrives.
from .ingest_image import IMAGE_EXTS as _IMAGE_EXTS
from .ingest_video import VIDEO_EXTS as _VIDEO_EXTS

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "jobs"
MAX_UPLOAD = 50 * 1024 * 1024

_DOC_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm", ".md", ".txt"}
_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}
_CONTRACT_EXTS = {".json"}     # payloads from external modality workers (CONTRACT.md)

_ACCEPTED = _DOC_EXTS | _AUDIO_EXTS | _CONTRACT_EXTS | _IMAGE_EXTS | _VIDEO_EXTS

#: One-click demo inputs, one per data type the pipeline understands. Paths are
#: repo-relative; big fixtures are reused rather than duplicated. Each entry
#: names the route it exercises so the homepage can say WHY it is interesting.
SAMPLES: dict[str, dict] = {
    "report":   {"path": "tests/fixtures/executive_summary.pdf",
                 "label": "PDF report", "kind": "document", "tone": "#4d82ff",
                 "desc": "sections, tables and figures — structural route"},
    "essay":    {"path": "tests/fixtures/flat_essay.md",
                 "label": "Markdown essay", "kind": "document", "tone": "#4d82ff",
                 "desc": "flat prose, no headings — semantic topic drift"},
    "webpage":  {"path": "data/samples/product_page.html",
                 "label": "HTML page", "kind": "web", "tone": "#38bdf8",
                 "desc": "headings and a spec table — structural route"},
    "csv":      {"path": "tests/fixtures/sales_q3.csv",
                 "label": "CSV dataset", "kind": "spreadsheet", "tone": "#2dd4bf",
                 "desc": "480 rows — schema card + header-repeating row groups"},
    "workbook": {"path": "tests/fixtures/people_ops.xlsx",
                 "label": "Excel workbook", "kind": "spreadsheet", "tone": "#2dd4bf",
                 "desc": "three sheets, each profiled separately"},
    "json":     {"path": "data/samples/orders.json",
                 "label": "JSON records", "kind": "data", "tone": "#c084fc",
                 "desc": "a plain API export — chunked as data, not rejected"},
    "chart":    {"path": "data/samples/quarterly_chart.png",
                 "label": "Chart image", "kind": "image", "tone": "#f59e0b",
                 "desc": "OCR + objects + what the picture asserts"},
    "meeting":  {"path": "tests/fixtures/audio_sample.m4a",
                 "label": "Meeting audio", "kind": "audio", "tone": "#34d399",
                 "desc": "transcribed and speaker-labelled, chunked by turns"},
    "video":    {"path": "data/fixture.mp4",
                 "label": "Screencast video", "kind": "video", "tone": "#f472b6",
                 "desc": "four topics found by scene + speech + topic signals"},
    "contract": {"path": "tests/fixtures/video_contract.json",
                 "label": "Worker contract", "kind": "data", "tone": "#c084fc",
                 "desc": "an external worker's payload — CONTRACT.md in action"},
}


def _ensure_logging() -> None:
    """Cleave's own INFO lines reach the console however the app was started.

    Under `python -m uvicorn` the root logger has no handler, so the startup
    status line — the thing you most want to see before a demo — would be
    swallowed. Only this package's logger is touched; uvicorn keeps its own.
    """
    pkg = logging.getLogger("cleave")
    if not pkg.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        pkg.addHandler(handler)
    pkg.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _ensure_logging()
    _rehydrate_jobs()
    # Check providers at boot, not on first use: a dead key should be visible
    # before a job is submitted, not discovered halfway through a demo.
    from .health import enrichment_banner, system_status  # noqa: PLC0415

    banner = enrichment_banner(system_status(refresh=True))
    log.info("%s (%s)", banner["text"], banner["detail"])
    yield


app = FastAPI(title="Cleave", docs_url="/docs", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "cleave" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "cleave" / "templates")


def _highlight_scene(text: str) -> "Markup":
    """Bold the ``Scene: ...`` line inside a video chunk's content.

    A multimodal chunk's content is transcript, then Scene/Actions/Text-on-screen/
    Visible lines stacked as plain paragraphs (see ingest_video._as_cleave_unit) -
    without this the visual description reads as just another line of prose.
    """
    from markupsafe import Markup, escape  # noqa: PLC0415

    lines = []
    for line in (text or "").split("\n"):
        if line.startswith("Scene: "):
            lines.append(f'<strong class="text-[color:var(--text)]">Scene:</strong>'
                          f'{escape(line[len("Scene:"):])}')
        else:
            lines.append(str(escape(line)))
    return Markup("\n".join(lines))


templates.env.filters["highlight_scene"] = _highlight_scene


@dataclass(slots=True)
class FileState:
    """Live progress of one input within a job, for the per-file list in the
    status view. A nine-file job showing a single opaque line is how a stalled
    input hides; one row per file is how it doesn't."""

    name: str
    status: str = "queued"            # queued | running | done | error
    percent: int = 0
    message: str = "queued"


#: Stage pills for the status view, per input kind. The threshold is the job
#: percent at which the stage begins for a single-input job (frac f maps to
#: 5 + 85*f; video engine stages map through 0.1 + 0.5 * vke_pct/100).
_PIPELINES: dict[str, list[tuple[str, int]]] = {
    "document": [("understand", 5), ("parse structure", 13), ("graph", 47),
                 ("route & chunk", 56), ("enrich", 64), ("write", 90)],
    "tabular":  [("understand", 5), ("parse table", 13), ("profile & graph", 47),
                 ("tabular chunk", 56), ("enrich", 64), ("write", 90)],
    "audio":    [("understand", 5), ("transcribe", 13), ("graph", 47),
                 ("temporal chunk", 56), ("enrich", 64), ("write", 90)],
    "video":    [("probe", 5), ("transcribe", 28), ("scenes & frames", 36),
                 ("objects & OCR", 44), ("boundary chunk", 48),
                 ("keyframe vision", 53), ("write", 90)],
    "image":    [("understand", 5), ("OCR & objects", 13), ("graph", 47),
                 ("route & chunk", 56), ("enrich", 64), ("write", 90)],
    "web":      [("fetch", 5), ("extract article", 13), ("graph", 47),
                 ("route & chunk", 56), ("enrich", 64), ("write", 90)],
    "contract": [("validate", 5), ("import units", 13), ("write", 90)],
}

#: Batches mix kinds and rescale each file's progress window, so the per-kind
#: thresholds above would lie — the coarse path is the honest one there.
_GENERIC_PIPELINE: list[tuple[str, int]] = [
    ("understand", 5), ("extract", 15), ("graph", 60),
    ("route & chunk", 70), ("enrich", 78), ("write", 90)]


def _input_kind(name: str) -> str:
    """Which ingest pipeline a filename (or URL) will take."""
    if name.startswith(("http://", "https://")):
        return "web"
    suffix = Path(name).suffix.lower()
    if suffix in _AUDIO_EXTS:
        return "audio"
    if suffix in _VIDEO_EXTS:
        return "video"
    if suffix in _IMAGE_EXTS:
        return "image"
    if suffix in _CONTRACT_EXTS:
        return "contract"
    if suffix in {".csv", ".xlsx"}:
        return "tabular"
    return "document"


@dataclass(slots=True)
class Job:
    id: str
    filename: str                      # display summary, e.g. "3 files (a.pdf, b.mp3, …)"
    filenames: list[str] = field(default_factory=list)
    files: list[FileState] = field(default_factory=list)
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
    def pipeline(self) -> list[tuple[str, int]]:
        if len(self.filenames) == 1:
            return _PIPELINES[_input_kind(self.filenames[0])]
        return _GENERIC_PIPELINE

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
            files=[FileState(name=n, status="done", percent=100, message="done")
                   for n in names],
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


def run_job(job_id: str, input_paths: list[Path | str]) -> None:
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
        failures: list[str] = []

        for i, input_path in enumerate(input_paths):
            lo, hi = 5 + int(85 * i / n), 5 + int(85 * (i + 1) / n)
            fs = job.files[i] if i < len(job.files) else None
            if fs:
                fs.status, fs.message = "running", "starting…"

            def progress(frac: float, msg: str, lo=lo, hi=hi, fs=fs) -> None:
                _set(job, lo + int((hi - lo) * min(1.0, max(0.0, frac))), msg)
                if fs:
                    fs.percent = max(fs.percent, int(100 * min(1.0, max(0.0, frac))))
                    fs.message = msg

            # One bad file must not take the other eight with it: the failure is
            # recorded, shown on its own row, and the job carries on.
            try:
                units, file_meta, nodes, edges, enrichment = _process_file(
                    job, input_path, prefix=f"f{i}_", ledger=ledger, progress=progress)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                log.exception("job %s: input %s failed", job_id, input_path)
                failures.append(f"{_input_name(input_path)}: {reason}")
                if fs:
                    fs.status, fs.percent = "error", 100
                    fs.message = reason[:200]
                files_meta.append(_failure_meta(input_path, reason))
                continue

            if fs:
                fs.status, fs.percent, fs.message = "done", 100, "done"
            all_units.extend(units)
            files_meta.append(file_meta)
            graph_nodes.extend(nodes)
            graph_edges.extend(edges)
            if enrichment:
                enrichments.append(enrichment)

        if not all_units and failures:
            # Nothing survived — that is a failed job, not an empty result.
            raise RuntimeError("; ".join(failures))

        _set(job, 90, f"{len(all_units)} knowledge units — writing artifacts…")
        graph = {"nodes": graph_nodes, "edges": graph_edges} if (graph_nodes or graph_edges) else None
        _write_artifacts(job, all_units, files_meta, graph, t0,
                         ledger=ledger, enrichments=enrichments)

        job.elapsed_s = round(time.time() - t0, 1)
        job.progress, job.message, job.status = 100, "done", "done"
        if failures:
            job.message = f"done — {len(failures)} input(s) failed"
    except Exception as exc:  # surface honestly; never a silent dead job
        log.exception("job %s failed", job_id)
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "failed"
        job.status = "error"


def _input_name(input_path) -> str:
    return input_path if isinstance(input_path, str) else Path(input_path).name


def _failure_hint(reason: str) -> str:
    """One sentence of what-to-do for the person reading a failure, matched on
    the raw exception text. The raw error stays shown; this sits next to it."""
    r = reason.lower()
    if "contract version" in r:
        return ("This file declares a \"contract\" key, so it was read as a modality-"
                "worker payload. Set \"contract\": 1 (see CONTRACT.md) — or remove the "
                "key and the file will be chunked as ordinary JSON data.")
    if "unsupported document type" in r or "unsupported image type" in r \
            or "unsupported video type" in r:
        return "This extension isn't handled yet — the accepted formats are listed on the upload form."
    if "expecting value" in r or "jsondecodeerror" in r:
        return "The file isn't valid JSON — check for a trailing comma or a truncated download."
    if "stt worker" in r:
        return ("Transcription needs an engine: install the 'video' and 'meetings' extras "
                "for local ASR, or start the STT worker on the port in CLEAVE_STT_URL.")
    if "no content could be extracted" in r:
        return ("The page returned no usable text — it may need JavaScript rendering "
                "(install the 'web-rendered' extra) or it may block automated fetches.")
    if "not an http(s) url" in r:
        return "Only http:// and https:// links can be fetched."
    if "exceeds" in r and "mb" in r:
        return "The upload limit is 50 MB per file — trim the file or raise MAX_UPLOAD."
    return ("The full traceback is in the server log. The other files in this job "
            "were processed normally.")


def _failure_meta(input_path, reason: str) -> dict:
    """A files_meta entry for an input that failed, so the results page shows
    the failure alongside the files that worked instead of erasing it."""
    name = _input_name(input_path)
    return {
        "filename": name, "title": None, "source": str(input_path),
        "error": reason,
        "hint": _failure_hint(reason),
        "warnings": [f"processing failed — {reason}"],
        "profile": {"route": "failed", "route_reason": reason},
        "cleaning": None, "figures": None,
    }


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


def _process_file(job: Job, input_path: Path | str, *, prefix: str, ledger,
                  progress) -> tuple:
    """Run the single-file pipeline for one input in a (possibly multi-input)
    job. Returns (units, file_meta, graph_nodes, graph_edges, enrichment).

    An input is a path on disk or an http(s) URL; a URL is fetched to Markdown
    first and is otherwise indistinguishable from a document downstream.
    """
    from .ingest_web import is_url  # noqa: PLC0415

    if isinstance(input_path, str) and is_url(input_path):
        return _process_url(job, input_path, prefix=prefix, ledger=ledger,
                            progress=progress)

    input_path = Path(input_path)
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
    elif suffix in _VIDEO_EXTS:
        from .ingest_video import ingest_video  # noqa: PLC0415

        progress(0.1, f"video engine: transcript, scenes, vision… ({filename})")
        imported, ready_units = ingest_video(
            input_path, progress=lambda f, m: progress(0.1 + 0.5 * f, m))
        if ready_units:
            # VKE's own multimodal boundaries were kept; they arrive finished.
            for u in ready_units:
                _prefix_unit(u, prefix)
            from .meeting import collect_unit_semantics, refine_ambiguous  # noqa: PLC0415

            collect_unit_semantics(ready_units)
            refine_ambiguous(ready_units, use_llm=job.use_llm, ledger=ledger)
            file_meta = {
                "filename": filename,
                "title": ready_units[0].context.document_title,
                "source": str(input_path),
                "warnings": [],
                "profile": {
                    "route": "multimodal",
                    "route_reason": "boundaries drawn by the video engine, which scores "
                                    "speech, visual novelty and topic drift together — "
                                    "signals that do not survive flattening to elements",
                },
                "cleaning": None,
            }
            progress(1.0, f"done ({filename})")
            return ready_units, file_meta, [], [], None
        ingest = imported
    elif suffix in _IMAGE_EXTS:
        from .ingest_image import ingest_image  # noqa: PLC0415

        progress(0.1, f"reading the picture (OCR, objects, vision)… ({filename})")
        ingest = ingest_image(input_path, use_llm=job.use_llm, ledger=ledger)
    else:
        from .ingest_document import ingest_document  # noqa: PLC0415

        progress(0.1, f"parsing structure (Docling)… ({filename})")
        ingest = ingest_document(input_path, use_llm=job.use_llm, ledger=ledger)

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

    if any(u.temporal is not None for u in units):
        from .meeting import collect_unit_semantics, refine_ambiguous  # noqa: PLC0415

        collect_unit_semantics(units)
        refine_ambiguous(units, use_llm=job.use_llm, ledger=ledger)

    enrichment = _maybe_enrich(job, units, ingest, ledger, progress, filename)

    file_meta = {
        "filename": filename, "title": ingest.title, "source": ingest.source_uri,
        "warnings": ingest.warnings, "profile": profile.to_dict(), "cleaning": ingest.cleaning,
        "figures": ingest.figures,
    }
    progress(1.0, f"done ({filename})")
    gd = graph.to_dict()
    return units, file_meta, gd["nodes"], gd["edges"], enrichment


def _process_url(job: Job, url: str, *, prefix: str, ledger, progress) -> tuple:
    """Fetch a web page and run it through the ordinary document pipeline."""
    from .chunkers import chunk  # noqa: PLC0415
    from .graph import ContextGraph  # noqa: PLC0415
    from .ingest_web import ingest_web  # noqa: PLC0415

    progress(0.0, f"fetching {url}…")
    ingest = ingest_web(url, use_llm=job.use_llm, ledger=ledger)

    for e in ingest.elements:
        _prefix_element(e, prefix)
    progress(0.5, f"{len(ingest.elements)} elements — building context graph… ({url})")
    graph = ContextGraph(ingest.elements)
    progress(0.6, f"routing and chunking… ({url})")
    units, profile = chunk(ingest, graph)
    for u in units:
        _prefix_unit(u, prefix)

    enrichment = _maybe_enrich(job, units, ingest, ledger, progress, url)
    file_meta = {
        "filename": url, "title": ingest.title, "source": url,
        "warnings": ingest.warnings, "profile": profile.to_dict(),
        "cleaning": ingest.cleaning, "figures": ingest.figures,
    }
    progress(1.0, f"done ({url})")
    gd = graph.to_dict()
    return units, file_meta, gd["nodes"], gd["edges"], enrichment


def _maybe_enrich(job: Job, units, ingest, ledger, progress, label: str) -> dict | None:
    """Selective enrichment for whichever inputs produced flagged units."""
    flagged_n = sum(1 for u in units if u.decision.escalation_flags)
    if not flagged_n:
        return None
    from .enrich import enrich  # noqa: PLC0415

    progress(0.7, (f"enriching {flagged_n} context-poor chunks… ({label})"
                   if job.use_llm else f"skipping LLM enrichment ({label})…"))
    doc_text = "\n\n".join(e.text for e in ingest.elements if e.text)
    return enrich(
        units, doc_text, ledger=ledger, use_llm=job.use_llm,
        progress=lambda i, n2: progress(
            0.7 + 0.25 * i / max(1, n2), f"enriching… {i}/{n2} ({label})"))


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


def _merge_figures(reports: list[dict | None]) -> dict | None:
    """Combine per-file figure reports into one job-level total."""
    total = {"figures": 0, "understood": 0, "skipped": 0, "llm_calls": 0,
             "cost_usd": 0.0, "reasons": {}}
    seen = False
    for r in reports:
        if not r or not r.get("figures"):
            continue
        seen = True
        for k in ("figures", "understood", "skipped", "llm_calls"):
            total[k] += r.get(k, 0)
        total["cost_usd"] += r.get("cost_usd", 0.0)
        for producer, why in (r.get("reasons") or {}).items():
            total["reasons"].setdefault(producer, why)
    total["cost_usd"] = round(total["cost_usd"], 6)
    return total if seen else None


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
    from .meeting import minutes as _minutes  # noqa: PLC0415

    mins = _minutes(units)
    if any(mins.values()):
        record["minutes"] = mins
    combined_figures = _merge_figures([f.get("figures") for f in files_meta])
    if combined_figures:
        record["figures"] = combined_figures
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
    from .health import enrichment_banner, system_status  # noqa: PLC0415

    jobs = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)[:10]
    checks = system_status()
    return templates.TemplateResponse(request, "index.html", {
        "jobs": jobs, "scorecard": scorecard,
        "usage": read_cumulative(), "providers": describe_providers(),
        "checks": checks, "banner": enrichment_banner(checks),
        "samples": SAMPLES,
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
        "figures": meta.get("figures"), "minutes": meta.get("minutes"),
    })


# ───────── api ─────────

@app.post("/api/jobs")
async def create_job(background: BackgroundTasks,
                     files: list[UploadFile] = File(default=[]),
                     urls: str = Form(""),
                     use_llm: str = Form("true")):
    """Start a job from uploaded files, pasted URLs, or both in one batch."""
    from .ingest_web import is_url  # noqa: PLC0415

    job_id = uuid.uuid4().hex[:10]
    dest_dir = DATA / job_id / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    dest_paths: list[Path | str] = []

    link_list = [u.strip() for u in re.split(r"[\s,]+", urls or "") if u.strip()]
    for link in link_list:
        if not is_url(link):
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(400, f"not an http(s) URL: {link!r}")
        names.append(link)
        dest_paths.append(link)

    try:
        for i, f in enumerate(files):
            if not f.filename:
                continue          # browsers post an empty part for an unused input
            fname = f.filename
            suffix = Path(fname).suffix.lower()
            if suffix not in _ACCEPTED:
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
            raise HTTPException(400, "no files or URLs supplied")
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    job = Job(id=job_id, filename=_display_name(names), filenames=names,
             files=[FileState(name=n) for n in names],
             use_llm=use_llm.lower() not in ("false", "0", "off", "no"))
    JOBS[job_id] = job
    background.add_task(run_job, job_id, dest_paths)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/api/samples")
async def run_sample(background: BackgroundTasks, name: str = Form(...),
                     use_llm: str = Form("true")):
    """Start a job from bundled sample inputs — `name` is a SAMPLES key, or
    'all' for one combined job across every data type."""
    keys = list(SAMPLES) if name == "all" else [name]
    unknown = [k for k in keys if k not in SAMPLES]
    if unknown:
        raise HTTPException(404, f"unknown sample {unknown[0]!r}")

    job_id = uuid.uuid4().hex[:10]
    dest_dir = DATA / job_id / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    dest_paths: list[Path | str] = []
    for k in keys:
        src = ROOT / SAMPLES[k]["path"]
        if not src.exists():
            shutil.rmtree(dest_dir.parent, ignore_errors=True)
            raise HTTPException(500, f"sample file missing on disk: {SAMPLES[k]['path']}")
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        names.append(dest.name)
        dest_paths.append(dest)

    job = Job(id=job_id, filename=_display_name(names), filenames=names,
              files=[FileState(name=n) for n in names],
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
def health(refresh: bool = False):
    """Subsystem status. ``?refresh=true`` re-probes instead of using the cache."""
    from .health import enrichment_banner, system_status  # noqa: PLC0415
    from .llm import get_provider  # noqa: PLC0415

    p = get_provider()
    checks = system_status(refresh=refresh)
    return {
        "ok": all(c["ok"] for c in checks if c["key"] != "vision"),
        "jobs": len(JOBS), "llm": p.name, "model": p.model,
        "enrichment": enrichment_banner(checks), "checks": checks,
    }


@app.get("/status", response_class=HTMLResponse)
def status_panel(request: Request, refresh: bool = False):
    """The status panel on its own, so the UI can re-probe without a reload."""
    from .health import enrichment_banner, system_status  # noqa: PLC0415

    checks = system_status(refresh=refresh)
    return templates.TemplateResponse(request, "_system_status.html", {
        "checks": checks, "banner": enrichment_banner(checks),
    })


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("cleave.app:app", host="127.0.0.1", port=8321, reload=False)
