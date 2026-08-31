"""The JSON surface: job creation, raw artifacts, usage, health."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from . import uploads
from .jobs import DATA, JOBS, Job, display_name, job_artifact

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/jobs")
async def create_job(background: BackgroundTasks, files: list[UploadFile] = File(...),
                     use_llm: str = Form("true")):
    from ..pipeline import run_job  # noqa: PLC0415 — avoids an import cycle

    job_id = uuid.uuid4().hex[:10]
    dest_dir = DATA / job_id / "input"
    try:
        names, dest_paths = await uploads.save_uploads(files, dest_dir)
    except Exception:
        uploads.discard(dest_dir)
        raise
    job = Job(id=job_id, filename=display_name(names), filenames=names,
              use_llm=use_llm.lower() not in ("false", "0", "off", "no"))
    JOBS[job_id] = job
    background.add_task(run_job, job_id, dest_paths)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/api/jobs/{job_id}/units")
def api_units(job_id: str):
    return _artifact(job_id, "units.json")


@router.get("/api/jobs/{job_id}/graph")
def api_graph(job_id: str):
    return _artifact(job_id, "graph.json")


@router.get("/api/jobs/{job_id}/profile")
def api_profile(job_id: str):
    return _artifact(job_id, "profile.json")


def _artifact(job_id: str, name: str) -> JSONResponse:
    """One job artifact, looked up through the registry.

    These three routes used to build ``DATA / job_id / name`` from the raw URL
    while every HTML route checked ``JOBS`` first — so a job id of ``../..``
    read whatever it liked. ``job_artifact`` does the membership check.
    """
    path = job_artifact(job_id, name)
    if not path.exists():
        raise HTTPException(404, f"{name} not available")
    try:
        return JSONResponse(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        log.warning("job %s: %s unreadable (%s)", job_id, name, exc)
        raise HTTPException(404, f"{name} is unreadable") from exc


@router.get("/api/usage")
def api_usage():
    """Install-wide token and cost ledger, broken down by model."""
    from ..llm import describe_providers  # noqa: PLC0415
    from ..usage import read_cumulative  # noqa: PLC0415

    return {"cumulative": read_cumulative(), "providers": describe_providers()}


@router.get("/health")
def health():
    from ..llm import get_provider  # noqa: PLC0415

    p = get_provider()
    return {"ok": True, "jobs": len(JOBS), "llm": p.name, "model": p.model}
