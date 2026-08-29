"""The honesty contract, as assertions.

VKE's whole credibility rests on never dressing a measurement up as
understanding. That is a property of the code, so it belongs in the test suite
rather than in a comment somebody can drift away from:

  * `actions` stays empty unless a real semantic model produced one
  * no observation can claim a heuristic as its source
  * every observation carries source, timestamp, model - and a confidence only
    when its producer actually supplies one
  * objects/ocr_text/actions are pure projections of `observations`, so nothing
    can populate them without leaving provenance behind
  * none of it can move a boundary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vke import detect, signals  # noqa: E402
from vke.chunker import attach_observations, build_units, detect_boundaries  # noqa: E402
from vke.config import CONFIG_C  # noqa: E402
from vke.schemas import (  # noqa: E402
    BoundaryExplanation,
    FrameFeature,
    KnowledgeUnit,
    SceneCut,
    Span,
    Utterance,
    VisualObservation,
)

SEMANTIC_SOURCES = {"object_detector", "ocr", "vlm"}


def _observations() -> list[VisualObservation]:
    return [
        VisualObservation(kind="object", value="laptop", source="object_detector",
                          ts=5.0, model="yolov10n-onnx", confidence=0.88,
                          box=[0.1, 0.1, 0.4, 0.5]),
        VisualObservation(kind="object", value="laptop", source="object_detector",
                          ts=12.0, model="yolov10n-onnx", confidence=0.61),
        VisualObservation(kind="text", value="Deploy Production", source="ocr",
                          ts=8.0, model="rapidocr-3.9.2", confidence=0.97),
        VisualObservation(kind="action", value="typing", source="vlm",
                          ts=9.0, model="gpt-4o-mini", confidence=None),
        VisualObservation(kind="object", value="car", source="object_detector",
                          ts=95.0, model="yolov10n-onnx", confidence=0.9),
    ]


def _unit(start: float = 0.0, end: float = 30.0) -> KnowledgeUnit:
    return KnowledgeUnit(
        id="u0", video_id="v", span=Span(start=start, end=end), title="t",
        transcript="some words", boundary=BoundaryExplanation(
            ts=start, score=0.0, threshold=0.0),
    )


# --------------------------------------------------------------------------- #
# 1. actions are never invented
# --------------------------------------------------------------------------- #
def test_the_observe_stage_never_produces_an_action():
    """Action recognition is deferred, so this stage must not emit one - and in
    particular must never derive one from motion or edge density."""
    obs = [o for o in _observations() if o.source != "vlm"]
    unit = attach_observations(_unit(), obs)
    assert unit.actions == []


def test_actions_come_only_from_a_semantic_model():
    unit = attach_observations(_unit(), _observations())
    assert unit.actions == ["typing"]
    assert all(o.source == "vlm" for o in unit.observations if o.kind == "action")


def test_detect_module_never_writes_an_action():
    """A structural guard: nothing in detect.py may construct kind="action"."""
    source = (ROOT / "vke" / "detect.py").read_text(encoding="utf-8")
    assert 'kind="action"' not in source


def test_measured_signals_never_become_observation_values():
    """motion / edge_density may GATE where a model runs; they may never be a
    value a model is credited with."""
    source = (ROOT / "vke" / "detect.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if "value=" in line:
            assert "motion" not in line and "edge" not in line, line


# --------------------------------------------------------------------------- #
# 2. provenance is complete and never fabricated
# --------------------------------------------------------------------------- #
def test_no_observation_claims_a_heuristic_source():
    unit = attach_observations(_unit(), _observations())
    assert {o.source for o in unit.observations} <= SEMANTIC_SOURCES


def test_every_observation_carries_a_model_and_a_timestamp_in_span():
    unit = attach_observations(_unit(), _observations())
    assert unit.observations
    for o in unit.observations:
        assert o.model, "an observation without a model is unattributable"
        assert unit.span.contains(o.ts)


def test_confidence_is_real_or_absent_but_never_invented():
    unit = attach_observations(_unit(), _observations())
    for o in unit.observations:
        if o.source in ("object_detector", "ocr"):
            assert o.confidence is not None, "a scoring producer must record its score"
        if o.source == "vlm":
            assert o.confidence is None, "a VLM ships no score; 1.0 would be a lie"


def test_confidence_defaults_to_none_not_one():
    """The schema default itself must not manufacture certainty."""
    o = VisualObservation(kind="object", value="x", source="vlm", ts=0.0, model="m")
    assert o.confidence is None


# --------------------------------------------------------------------------- #
# 3. the flattened lists cannot be written behind the evidence's back
# --------------------------------------------------------------------------- #
def test_flat_lists_are_exactly_the_projection_of_observations():
    unit = attach_observations(_unit(), _observations())
    for field, kind in (("objects", "object"), ("ocr_text", "text"),
                        ("actions", "action")):
        assert set(getattr(unit, field)) == {
            o.value for o in unit.observations if o.kind == kind
        }


def test_reattaching_replaces_rather_than_accumulates():
    """A second pass (the VLM) must not leave orphaned values with no evidence."""
    unit = attach_observations(_unit(), _observations())
    attach_observations(unit, [])
    assert unit.objects == [] and unit.ocr_text == [] and unit.actions == []
    assert unit.observations == [] and unit.visual_sources == []


def test_observations_outside_the_span_are_not_claimed():
    unit = attach_observations(_unit(0.0, 30.0), _observations())
    assert "car" not in unit.objects  # it was seen at 95s


def test_duplicate_sightings_keep_the_most_confident():
    unit = attach_observations(_unit(), _observations())
    laptop = [o for o in unit.observations if o.value == "laptop"]
    assert len(laptop) == 1 and laptop[0].confidence == 0.88


def test_visual_sources_lists_every_contributing_producer():
    unit = attach_observations(_unit(), _observations())
    assert unit.visual_sources == ["object_detector", "ocr", "vlm"]


def test_the_vlm_merges_with_detector_evidence_instead_of_replacing_it():
    """pipeline.py used to assign result.objects straight onto unit.objects,
    which would silently delete everything the detector and OCR had found."""
    from vke.pipeline import _vlm_observations
    from vke.schemas import VisionResult

    unit = attach_observations(_unit(), [
        o for o in _observations() if o.source != "vlm" and o.ts < 30.0])
    assert "laptop" in unit.objects and "Deploy Production" in unit.ocr_text

    result = VisionResult(description="an IDE", ocr_text=["Save"],
                          objects=["monitor"], actions=["typing"], source="vlm")
    attach_observations(unit, [*unit.observations,
                               *_vlm_observations(result, 9.0, "gpt-4o-mini")])

    assert "laptop" in unit.objects and "monitor" in unit.objects
    assert "Deploy Production" in unit.ocr_text and "Save" in unit.ocr_text
    assert unit.actions == ["typing"]
    assert unit.visual_sources == ["object_detector", "ocr", "vlm"]
    # and the detector's own scores survived the merge unchanged
    laptop = next(o for o in unit.observations if o.value == "laptop")
    assert laptop.source == "object_detector" and laptop.confidence == 0.88


# --------------------------------------------------------------------------- #
# 4. enrichment cannot move a boundary
# --------------------------------------------------------------------------- #
def _scoring_inputs():
    utts = [
        Utterance(id=f"u{i}", span=Span(start=i * 10.0, end=i * 10.0 + 8.0),
                  text="alpha beta gamma delta" if i < 3 else "kappa lambda mu nu")
        for i in range(6)
    ]
    feats = [FrameFeature(ts=t / 2.0, hsv_hist=[0.5, 0.5], edge_density=0.03,
                          motion=0.01, brightness=0.5) for t in range(120)]
    return utts, feats, [SceneCut(ts=30.0)]


def test_boundaries_are_identical_with_and_without_observations():
    utts, feats, cuts = _scoring_inputs()
    curves = signals.compute_curves(utts, feats, cuts, 60.0)
    bounds, _, _, _ = detect_boundaries(curves, CONFIG_C, utts, 60.0)

    without = build_units("v", CONFIG_C, bounds, utts, feats, cuts, 60.0, {}, [])
    with_obs = build_units("v", CONFIG_C, bounds, utts, feats, cuts, 60.0, {},
                           _observations())

    assert [u.span.model_dump() for u in without] == \
           [u.span.model_dump() for u in with_obs]
    assert [u.boundary.ts for u in without] == [u.boundary.ts for u in with_obs]


def test_the_scorer_cannot_even_see_enrichment():
    """Structural: signals.py is upstream of every model and must stay that way.

    A boundary that depended on model output would be circular - model output
    only exists near candidates the score already found. Asserted against
    imports and signatures rather than prose, so that merely *writing* about
    detection in a comment does not fail the build.
    """
    import inspect

    source = (ROOT / "vke" / "signals.py").read_text(encoding="utf-8")
    assert "from .detect" not in source
    assert "import detect" not in source

    for fn in (signals.compute_curves, signals.fuse, detect_boundaries):
        params = inspect.signature(fn).parameters
        assert "observations" not in params, fn.__name__
        assert not any("ocr" in p or "object" in p for p in params), fn.__name__


# --------------------------------------------------------------------------- #
# 5. absence is reported, never hidden
# --------------------------------------------------------------------------- #
def test_disabled_enrichment_says_so_rather_than_returning_a_silent_empty(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(detect, "ENABLE_OBJECT_DETECTION", False)
    monkeypatch.setattr(detect, "ENABLE_OCR", False)
    obs, status = detect.observe(tmp_path / "nope.mp4", [], [], 10.0)
    assert obs == []
    assert status["objects"].startswith("not_requested")
    assert status["ocr"].startswith("not_requested")
    assert status["actions"].startswith("not_requested")


def test_an_undecodable_video_reports_unavailable_and_does_not_raise(tmp_path: Path):
    obs, status = detect.observe(
        tmp_path / "missing.mp4",
        [FrameFeature(ts=0.0, hsv_hist=[1.0], edge_density=0.1, motion=0.0,
                      brightness=0.5)],
        [], 10.0)
    assert obs == []
    assert status["objects"].startswith("unavailable")


def test_a_broken_detector_is_reported_not_raised(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("model file is corrupt")

    monkeypatch.setattr(detect, "_download", boom)
    problem = detect.ObjectDetector().load()
    assert problem.startswith("unavailable")
    assert "corrupt" in problem


def test_a_detector_that_failed_to_load_returns_no_detections():
    """Never raise from the hot path: a detector that never loaded yields [],
    so the pipeline keeps going."""
    assert detect.ObjectDetector().detect(np.zeros((32, 32, 3), np.uint8), 1.0) == []


def test_failure_reasons_stay_short_enough_to_stamp_into_provenance():
    reason = detect._reason(RuntimeError("x" * 5000))
    assert len(reason) <= 141
