"""The processing pipeline: video in, Knowledge Units out.

Extraction runs ONCE. All three configs then reuse the same signal curves with
different weights, which is both the efficient thing to do and the reason the
comparison is a true ablation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import detect as detect_mod
from . import diarize as diarize_mod
from . import media, providers, signals
from .asr import transcribe
from . import graph as graph_mod
from .chunker import (
    attach_observations,
    build_units,
    detect_boundaries,
    refine_boundaries,
)
from .config import CONFIGS, DEFAULT_CONFIG, ENABLE_DIARIZATION, ENABLE_REFINE
from .enrich import enrich, polish_titles, validate
from .schemas import KnowledgeUnit, StageTrace, VideoMeta, VisualObservation
from .store import VideoStore

ProgressFn = Callable[[str, int, str], None]

STAGES = [
    ("probe", 5, "Reading video metadata"),
    ("asr", 35, "Transcribing speech"),
    ("frames", 55, "Sampling frames"),
    ("scenes", 60, "Detecting scenes"),
    ("diarize", 68, "Separating speakers"),
    ("observe", 74, "Detecting objects and on-screen text"),
    ("signals", 78, "Computing boundary signals"),
    ("chunking", 84, "Generating knowledge units"),
    ("keyframes", 89, "Extracting keyframes"),
    ("vision", 95, "Analysing keyframes"),
    ("done", 100, "Processed"),
]


def _noop(stage: str, percent: int, message: str) -> None:
    pass


def process_video(
    video_path: Path,
    video_id: str,
    progress: ProgressFn | None = None,
    force: bool = False,
) -> tuple[VideoMeta, dict[str, list[KnowledgeUnit]], list[StageTrace]]:
    report = progress or _noop
    store = VideoStore(video_id)
    traces: list[StageTrace] = []

    def stage(name: str) -> tuple[float, int, str]:
        pct, msg = next((p, m) for s, p, m in STAGES if s == name)
        report(name, pct, msg)
        return time.time(), pct, msg

    # --- probe ------------------------------------------------------------- #
    t0, _, _ = stage("probe")
    meta = media.probe(video_path, video_id)
    store.save_meta(meta)
    traces.append(StageTrace(stage="probe", seconds=round(time.time() - t0, 2),
                             detail={"duration": meta.duration, "fps": meta.fps}))

    # --- extraction (cached: this is the only expensive part) -------------- #
    cached = None if force else store.load_extraction()
    if cached is not None:
        utterances, features, cuts, provider_info, turns = cached
        report("asr", 68, "Reusing cached extraction")
        traces.append(StageTrace(stage="extraction", seconds=0.0,
                                 detail={"cached": True}))
    else:
        t0, _, _ = stage("asr")
        utterances, asr_provider = transcribe(video_path)
        traces.append(StageTrace(stage="asr", seconds=round(time.time() - t0, 2),
                                 detail={"utterances": len(utterances),
                                         "provider": asr_provider}))

        t0, _, _ = stage("frames")
        features = media.extract_frames(video_path, store.dir / "_frames")
        traces.append(StageTrace(stage="frames", seconds=round(time.time() - t0, 2),
                                 detail={"sampled": len(features)}))

        t0, _, _ = stage("scenes")
        cuts = media.detect_scenes(video_path)
        traces.append(StageTrace(stage="scenes", seconds=round(time.time() - t0, 2),
                                 detail={"cuts": len(cuts)}))

        t0, _, _ = stage("diarize")
        turns = []
        if ENABLE_DIARIZATION:
            try:
                turns = diarize_mod.diarize(video_path, utterances)
                utterances = diarize_mod.apply_to_utterances(utterances, turns)
            except Exception as exc:  # never let a heuristic stop the run
                print(f"[pipeline] diarization skipped ({type(exc).__name__}: {exc})")
        speakers_found = len({t.speaker for t in turns})
        traces.append(StageTrace(stage="diarize", seconds=round(time.time() - t0, 2),
                                 detail={"turns": len(turns),
                                         "speakers": speakers_found}))

        chosen = providers.describe_providers()
        provider_info = {
            "asr": asr_provider,
            "scenes": "pyscenedetect-content",
            "diarization": "energy_centroid" if turns else "none",
            "llm": chosen["llm"],
            "vision": chosen["vision"],
        }
        store.save_extraction(utterances, features, cuts, provider_info, turns)

    # --- enrichment: observed ONCE for the whole video --------------------- #
    # Frames are chosen video-wide rather than per unit, so a single detection
    # pass serves all three configs. That is a third of the work, and it keeps
    # the headline comparison an honest ablation: every config sees identical
    # visual evidence, and only the boundaries differ.
    t0, _, _ = stage("observe")
    cached_obs = None if force else store.load_observations()
    if cached_obs is not None:
        observations, enrichment = cached_obs
    else:
        observations, enrichment = detect_mod.observe(
            video_path, features, cuts, meta.duration)
        store.save_observations(observations, enrichment)
    traces.append(StageTrace(
        stage="observe", seconds=round(time.time() - t0, 2),
        detail={"observations": len(observations),
                "objects": sum(1 for o in observations if o.kind == "object"),
                "ocr_lines": sum(1 for o in observations if o.kind == "text"),
                **enrichment},
    ))

    # --- signals (computed once, shared by every config) ------------------- #
    t0, _, _ = stage("signals")
    changes = diarize_mod.speaker_changes(turns) if turns else []
    curves = signals.compute_curves(
        utterances, features, cuts, meta.duration, speaker_change_points=changes)
    traces.append(StageTrace(stage="signals", seconds=round(time.time() - t0, 2),
                             detail={"grid_points": int(curves["semantic"].grid.size)}))

    # --- chunking: three configs, one code path ---------------------------- #
    t0, _, _ = stage("chunking")
    units_by_config: dict[str, list[KnowledgeUnit]] = {}
    score_curves: dict[str, list[float]] = {}
    refine_stats: dict[str, dict[str, int]] = {}
    for key, config in CONFIGS.items():
        boundaries, grid, score, _threshold = detect_boundaries(
            curves, config, utterances, meta.duration
        )
        # The fixed baseline is left exactly as VideoRAG produces it; refining it
        # would flatter the comparison.
        if ENABLE_REFINE and not config.is_fixed:
            boundaries, refine_stats[key] = refine_boundaries(
                boundaries, utterances, meta.duration, grid, score)
        units = build_units(
            video_id, config, boundaries, utterances, features, cuts,
            meta.duration, provider_info, observations, enrichment,
        )
        units = enrich(units)
        for unit in units:
            unit.flags = validate(unit)
        units_by_config[key] = units
        score_curves[key] = [round(float(v), 4) for v in score]
    store.save_units(units_by_config)
    traces.append(StageTrace(
        stage="chunking", seconds=round(time.time() - t0, 2),
        detail={**{key: len(units) for key, units in units_by_config.items()},
                "refine": refine_stats},
    ))

    # --- graph over the displayed config ----------------------------------- #
    g = graph_mod.build(meta, units_by_config.get(DEFAULT_CONFIG, []), cuts)
    store.save_graph(g.to_dict())
    traces.append(StageTrace(stage="graph", seconds=0.0, detail=g.to_dict()["stats"]))

    grid = curves["semantic"].grid
    store.save_curves({
        "grid": [round(float(t), 3) for t in grid],
        "signals": {
            name: [round(float(v), 4) for v in curve.normalized]
            for name, curve in curves.items()
        },
        "scores": score_curves,
        "cuts": [c.ts for c in cuts],
        "utterances": [
            {"start": u.span.start, "end": u.span.end, "text": u.text,
             "speaker": u.speaker}
            for u in utterances
        ],
    })

    # --- keyframes --------------------------------------------------------- #
    t0, _, _ = stage("keyframes")
    # Collect every wanted moment first, then decode once. This loop used to open
    # and release a VideoCapture per unit per config - about ninety times.
    # A moment just inside the unit is more representative than its edge.
    keyframe_ts: dict[str, float] = {
        unit.id: min(unit.span.start + 1.5, max(unit.span.start, unit.span.end - 0.5))
        for units in units_by_config.values() for unit in units
    }
    wanted = {uid: ts for uid, ts in keyframe_ts.items()
              if not store.keyframe_path(uid).exists()}
    grabbed = media.grab_frames(video_path, sorted(set(wanted.values())))
    written = 0
    for unit_id, ts in wanted.items():
        frame = grabbed.get(ts)
        if frame is not None and media.write_keyframe(frame, store.keyframe_path(unit_id)):
            written += 1
    traces.append(StageTrace(stage="keyframes", seconds=round(time.time() - t0, 2),
                             detail={"written": written}))

    # --- Pass 2: expensive analysis, only on keyframes we already extracted --- #
    t0, _, _ = stage("vision")
    vision = providers.get_vision()
    llm = providers.get_llm()
    analysed = 0
    if vision.name != "offline" or llm.name != "offline":
        # Only the displayed config is enriched: the other two exist for the
        # comparison, and paying for three sets of model calls buys nothing.
        target = units_by_config.get(DEFAULT_CONFIG, [])
        for unit in target:
            frame = store.keyframe_path(unit.id)
            if vision.name != "offline" and frame.exists():
                result = vision.describe(frame, unit.transcript)
                if result.source == "vlm":
                    ts = keyframe_ts.get(unit.id, unit.span.start)
                    # MERGE, never overwrite. The detector and OCR already wrote
                    # evidence into this unit; assigning result.objects straight
                    # onto unit.objects would silently delete it.
                    attach_observations(unit, [
                        *unit.observations,
                        *_vlm_observations(result, ts, vision.name),
                    ])
                    # visual_source describes visual_context alone, so it only
                    # changes when the description itself did.
                    if result.description:
                        unit.visual_context = result.description
                        unit.visual_source = "vlm"
                    analysed += 1
        if llm.name != "offline":
            polish_titles(target, llm)
        store.save_units(units_by_config)
    traces.append(StageTrace(
        stage="vision", seconds=round(time.time() - t0, 2),
        detail={"keyframes_analysed": analysed,
                "vision_provider": vision.name, "llm_provider": llm.name,
                **providers.USAGE.as_dict()},
    ))

    store.save_traces(traces)
    report("done", 100, "Processed")
    return meta, units_by_config, traces


def _vlm_observations(result, ts: float, model: str) -> list[VisualObservation]:
    """A VLM's answer, converted to evidence.

    confidence is None throughout: a chat-completions response carries no
    per-item score, and stamping 1.0 on it would make a guess look exactly like
    a detector's measured 0.95. This is the only producer allowed to emit an
    action, because it is the only semantic model in the pipeline.
    """
    kinds = (("object", result.objects), ("text", result.ocr_text),
             ("action", result.actions))
    return [
        VisualObservation(kind=kind, value=str(value).strip(), source="vlm",
                          ts=round(ts, 3), model=model, confidence=None)
        for kind, values in kinds for value in values if str(value).strip()
    ]


def summarize(units_by_config: dict[str, list[KnowledgeUnit]]) -> str:
    parts = []
    for key, units in units_by_config.items():
        durations = [u.span.duration for u in units]
        mean = float(np.mean(durations)) if durations else 0.0
        parts.append(f"{key}: {len(units)} units (mean {mean:.1f}s)")
    return " | ".join(parts)
