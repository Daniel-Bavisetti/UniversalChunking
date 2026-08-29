"""P1 tests.

Deterministic offline providers are what make these possible: no network, no
model downloads beyond the cached ASR model, and identical output every run.

The most important test in the file is `test_configs_produce_different_boundaries`
- if that ever passes trivially, the project's central claim is dead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vke import signals, texttiling  # noqa: E402
from vke.asr import parse_sidecar, silence_gaps, utterance_edges  # noqa: E402
from vke.chunker import _snap, _suppress, build_units, detect_boundaries  # noqa: E402
from vke.config import CONFIG_A, CONFIG_B, CONFIG_C, MIN_EVENT_SECONDS  # noqa: E402
from vke.media import hist_distance  # noqa: E402
from vke.schemas import (  # noqa: E402
    BoundaryExplanation,
    FrameFeature,
    KnowledgeUnit,
    SceneCut,
    SignalContribution,
    Span,
    Utterance,
)

DATA = ROOT / "data"
EXTRACT = DATA / "_extract_fixture.json"
GROUND_TRUTH = DATA / "fixture_ground_truth.json"


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
def test_span_duration_and_containment():
    s = Span(start=10.0, end=25.5)
    assert s.duration == 15.5
    assert s.contains(10.0) and s.contains(25.0)
    assert not s.contains(25.5)  # end is exclusive
    assert not s.contains(9.9)


def test_span_overlap():
    a = Span(start=0, end=10)
    assert a.overlaps(Span(start=9, end=20))
    assert not a.overlaps(Span(start=10, end=20))  # touching is not overlapping


def test_span_duration_is_serialized():
    # The UI reads `duration` straight off the JSON, so it must survive dumping.
    assert "duration" in Span(start=1.0, end=3.5).model_dump()


def test_signal_name_is_not_a_literal():
    """A future PDF chunker must be able to emit its own signal names (plan sec.20)."""
    sc = SignalContribution(name="heading", raw=1.0, normalized=1.0,
                            weight=0.5, contribution=0.5)
    assert sc.name == "heading"


# --------------------------------------------------------------------------- #
# absolute timestamps - the baseline's core defect
# --------------------------------------------------------------------------- #
def test_sidecar_timestamps_are_absolute(tmp_path: Path):
    srt = tmp_path / "clip.srt"
    srt.write_text(
        "1\n00:00:04,500 --> 00:00:07,250\nHello there\n\n"
        "2\n00:01:02,000 --> 00:01:05,500\nSecond line\n\n"
        "3\n01:00:01,000 --> 01:00:03,000\nAn hour in\n",
        encoding="utf-8",
    )
    u = parse_sidecar(srt)
    assert len(u) == 3
    assert u[0].span.start == pytest.approx(4.5)
    assert u[1].span.start == pytest.approx(62.0)   # minutes carried, not reset
    assert u[2].span.start == pytest.approx(3601.0)  # hours carried
    assert [x.span.start for x in u] == sorted(x.span.start for x in u)


def test_utterance_edges_and_gaps():
    utts = [
        Utterance(id="a", span=Span(start=0, end=5), text="one"),
        Utterance(id="b", span=Span(start=8, end=12), text="two"),
    ]
    assert utterance_edges(utts) == [0.0, 8.0, 12.0]
    gaps = silence_gaps(utts)
    assert len(gaps) == 1
    midpoint, length = gaps[0]
    assert midpoint == pytest.approx(6.5)
    assert length == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# signals in isolation
# --------------------------------------------------------------------------- #
def test_robust_normalize_bounds_and_flatness():
    out = signals.robust_normalize(np.array([0.0, 1.0, 2.0, 3.0, 100.0]))
    assert out.min() >= 0.0 and out.max() <= 1.0
    flat = signals.robust_normalize(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(flat, 0.0)  # a constant signal carries no information


def test_robust_normalize_preserves_sparse_spikes():
    """A signal that is zero almost everywhere must survive normalization.

    Percentile scaling collapses here (5th and 95th are both 0). If that zeroed
    the array we would silently delete the sharpest, most confident boundaries.
    """
    sparse = np.zeros(200)
    sparse[100] = 1.9
    out = signals.robust_normalize(sparse)
    assert out[100] == pytest.approx(1.0)
    assert out.sum() == pytest.approx(1.0)  # everything else stays zero


def test_hist_distance_is_zero_for_identical_and_one_for_disjoint():
    a = [1.0, 0.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0, 1.0]
    assert hist_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert hist_distance(a, b) == pytest.approx(1.0, abs=1e-6)


def test_silence_curve_peaks_at_the_pause():
    utts = [
        Utterance(id="a", span=Span(start=0, end=10), text="before"),
        Utterance(id="b", span=Span(start=14, end=24), text="after"),
    ]
    grid = signals.build_grid(24.0)
    curve = signals.silence_curve(utts, grid)
    peak_t = float(grid[int(np.argmax(curve.normalized))])
    assert peak_t == pytest.approx(12.0, abs=1.0)  # midpoint of the 4s gap
    assert curve.normalized.max() == pytest.approx(1.0, abs=0.01)  # 4s >= full score


def test_visual_curve_peaks_at_a_scene_cut():
    grid = signals.build_grid(60.0)
    curve = signals.visual_curve([], [SceneCut(ts=30.0)], grid)
    assert float(grid[int(np.argmax(curve.normalized))]) == pytest.approx(30.0, abs=1.0)
    assert curve.normalized[0] < 0.1  # influence is local, not global


def test_visual_curve_absorbs_cuts_rather_than_adding_them():
    """A cut must not stack on top of the histogram signal (double counting)."""
    grid = signals.build_grid(20.0)
    feats = [
        FrameFeature(ts=float(t), hsv_hist=[1.0, 0.0] if t < 10 else [0.0, 1.0],
                     edge_density=0.0, motion=0.0, brightness=0.5)
        for t in range(0, 20)
    ]
    with_cut = signals.visual_curve(feats, [SceneCut(ts=10.0)], grid)
    assert with_cut.normalized.max() <= 1.0


def test_semantic_curve_fires_on_vocabulary_change_not_on_continuation():
    def utt(i: int, start: float, words: str) -> Utterance:
        return Utterance(id=f"u{i}", span=Span(start=start, end=start + 4.0), text=words)

    # 0-40s one vocabulary, 40-80s a completely different one.
    auth = "authentication password credentials session token login user account"
    dbs = "database migration schema table column index query rollback planner"
    utts = [utt(i, i * 4.0, auth) for i in range(10)]
    utts += [utt(10 + i, 40.0 + i * 4.0, dbs) for i in range(10)]

    grid = signals.build_grid(80.0)
    curve = signals.semantic_curve(utts, grid, block_seconds=20.0)
    peak_t = float(grid[int(np.argmax(curve.normalized))])
    assert abs(peak_t - 40.0) < 8.0, f"expected a peak near 40s, got {peak_t}"


def test_texttiling_depth_is_zero_on_uniform_similarity():
    """A flat similarity series has no valleys, so it must yield no boundaries."""
    assert np.allclose(texttiling.depth_scores(np.full(20, 0.5)), 0.0)


def test_texttiling_depth_does_not_plateau():
    """A long flat valley must produce ONE boundary, not a wide band of them."""
    sims = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
    depths = texttiling.depth_scores(sims)
    assert np.count_nonzero(depths) == 1


# --------------------------------------------------------------------------- #
# boundary selection mechanics
# --------------------------------------------------------------------------- #
def test_suppress_keeps_strongest_and_enforces_spacing():
    kept = _suppress([(0.9, 10.0), (0.8, 12.0), (0.7, 40.0)], min_gap=15.0)
    assert kept == [10.0, 40.0]  # 12.0 loses to the stronger neighbour


def test_snap_only_within_window():
    edges = [10.0, 30.0]
    assert _snap(10.8, edges, 2.0) == 10.0
    assert _snap(14.0, edges, 2.0) is None  # too far: leave it where it is


def test_fixed_config_reports_no_signal_influence():
    grid = signals.build_grid(100.0)
    curves = {
        "semantic": signals.SignalCurve("semantic", grid, np.zeros(grid.size), np.zeros(grid.size)),
        "visual": signals.SignalCurve("visual", grid, np.zeros(grid.size), np.zeros(grid.size)),
        "silence": signals.SignalCurve("silence", grid, np.zeros(grid.size), np.zeros(grid.size)),
    }
    bounds, _g, _s, _t = detect_boundaries(curves, CONFIG_A, [], 100.0)
    assert [b.ts for b in bounds] == [30.0, 60.0, 90.0]
    # Honesty: the baseline must not claim any signal drove its cuts.
    assert all(b.signals == [] for b in bounds)
    assert all("Fixed" in b.summary for b in bounds)


def test_units_are_contiguous_and_linked():
    bounds = [BoundaryExplanation(ts=30.0, score=1.0, threshold=0.5),
              BoundaryExplanation(ts=60.0, score=1.0, threshold=0.5)]
    utts = [Utterance(id="u0", span=Span(start=0, end=90), text="hello world testing")]
    units = build_units("vid", CONFIG_C, bounds, utts, [], [], 90.0)

    assert len(units) == 3
    assert units[0].span.start == 0.0 and units[-1].span.end == 90.0
    for a, b in zip(units, units[1:]):
        assert a.span.end == b.span.start          # no gaps, no overlaps
        assert a.next_unit_id == b.id
        assert b.prev_unit_id == a.id
    assert units[0].prev_unit_id is None
    assert units[-1].next_unit_id is None


def test_every_unit_carries_a_boundary_explanation():
    bounds = [BoundaryExplanation(ts=30.0, score=1.0, threshold=0.5)]
    units = build_units("vid", CONFIG_C, bounds, [], [], [], 60.0)
    assert all(u.boundary is not None for u in units)
    assert units[0].boundary.summary == "Start of video."


# --------------------------------------------------------------------------- #
# integration + regression against the fixture
# --------------------------------------------------------------------------- #
requires_fixture = pytest.mark.skipif(
    not EXTRACT.exists() or not GROUND_TRUTH.exists(),
    reason="fixture not built; run scripts/make_fixture.py then scripts/process.py",
)


def _load_fixture():
    blob = json.loads(EXTRACT.read_text(encoding="utf-8"))
    return (
        [Utterance(**u) for u in blob["utterances"]],
        [FrameFeature(**f) for f in blob["features"]],
        [SceneCut(**c) for c in blob["cuts"]],
        blob["duration"],
    )


def _boundaries_for(config):
    utts, feats, cuts, duration = _load_fixture()
    curves = signals.compute_curves(utts, feats, cuts, duration)
    bounds, _g, _s, _t = detect_boundaries(curves, config, utts, duration)
    return [b.ts for b in bounds]


@requires_fixture
def test_configs_produce_different_boundaries():
    """The central claim: changing only the weights changes where chunks begin.

    If A, B and C ever agree, the comparison demo is meaningless - so this
    failing is a red alert, not a flaky test.
    """
    a = _boundaries_for(CONFIG_A)
    b = _boundaries_for(CONFIG_B)
    c = _boundaries_for(CONFIG_C)
    assert a != b, "fixed and audio-only agree - the signals are doing nothing"
    assert b != c, "audio-only and VKE agree - the visual weight is doing nothing"


@requires_fixture
def test_visual_weight_finds_boundaries_audio_alone_misses():
    """The headline number: B->C differs only by the visual weight."""
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    tol = 3.0

    def found(bounds: list[float], target: float) -> bool:
        return any(abs(x - target) <= tol for x in bounds)

    b = _boundaries_for(CONFIG_B)
    c = _boundaries_for(CONFIG_C)

    visual_only = [t for t in truth["boundaries"]
                   if truth["kinds"][str(t)] in ("visual_only", "both")]
    wins = [t for t in visual_only if found(c, t) and not found(b, t)]
    assert wins, (
        "VKE found no boundary that audio-only missed; the visual signal is "
        f"not earning its weight. B={b} C={c} truth={truth['boundaries']}"
    )


@requires_fixture
def test_vke_beats_baselines_on_boundary_error():
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    targets = truth["boundaries"]

    def mean_error(bounds: list[float]) -> float:
        if not bounds:
            return 999.0
        return float(np.mean([min(abs(x - t) for x in bounds) for t in targets]))

    err_a = mean_error(_boundaries_for(CONFIG_A))
    err_c = mean_error(_boundaries_for(CONFIG_C))
    assert err_c < err_a, f"VKE ({err_c:.2f}s) should beat fixed windows ({err_a:.2f}s)"
    assert err_c < 3.0, f"VKE mean boundary error {err_c:.2f}s is too high"


@requires_fixture
def test_boundaries_land_on_utterance_edges():
    """Chunks must not start mid-sentence."""
    utts, _f, _c, _d = _load_fixture()
    edges = [u.span.start for u in utts]
    for ts in _boundaries_for(CONFIG_C):
        assert min(abs(ts - e) for e in edges) < 0.05, \
            f"boundary {ts} does not sit on an utterance edge"


@requires_fixture
def test_minimum_event_spacing_is_respected():
    bounds = _boundaries_for(CONFIG_C)
    for a, b in zip(bounds, bounds[1:]):
        assert b - a >= MIN_EVENT_SECONDS - 0.5


@requires_fixture
def test_units_round_trip_through_json():
    utts, feats, cuts, duration = _load_fixture()
    curves = signals.compute_curves(utts, feats, cuts, duration)
    bounds, _g, _s, _t = detect_boundaries(curves, CONFIG_C, utts, duration)
    units = build_units("fixture", CONFIG_C, bounds, utts, feats, cuts, duration)

    restored = [KnowledgeUnit(**json.loads(u.model_dump_json())) for u in units]
    assert [u.id for u in restored] == [u.id for u in units]
    assert all(u.provenance["weights"] == CONFIG_C.weights for u in restored)


# --------------------------------------------------------------------------- #
# P2 enrichment
# --------------------------------------------------------------------------- #
def _demo_units() -> list[KnowledgeUnit]:
    from vke.chunker import build_units
    bounds = [BoundaryExplanation(ts=30.0, score=0.8, threshold=0.4,
                                  signals=[SignalContribution(
                                      name="visual", raw=1.0, normalized=1.0,
                                      weight=0.4, contribution=0.4)]),
              BoundaryExplanation(ts=60.0, score=0.8, threshold=0.4)]
    utts = [
        Utterance(id="u0", span=Span(start=0, end=30),
                  text="We configure authentication. The user signs in with a password."),
        Utterance(id="u1", span=Span(start=30, end=60),
                  text="The authentication token expires. The user must sign in again."),
        Utterance(id="u2", span=Span(start=60, end=90),
                  text="Now the database migration adds a table and an index."),
    ]
    return build_units("vid", CONFIG_C, bounds, utts, [], [], 90.0)


def test_enrich_populates_context_without_copying_neighbours():
    from vke.enrich import enrich
    units = enrich(_demo_units())

    assert units[0].summary and units[1].summary
    assert units[1].prev_summary == units[0].summary
    assert units[1].next_summary == units[2].summary
    assert units[0].prev_summary is None
    assert units[-1].next_summary is None


def test_context_summarises_rather_than_copying_long_neighbours():
    """Context must not paste the neighbour's full text.

    Copying adjacent transcripts inflates every chunk and re-introduces exactly
    the redundancy event chunking is meant to remove. (For a transcript shorter
    than the summary budget, summary == transcript is correct, so this needs a
    genuinely long neighbour to test anything.)
    """
    from vke.chunker import build_units
    from vke.enrich import enrich

    long_text = " ".join(
        f"The authentication service validates the user credential number {i}."
        for i in range(20)
    )
    utts = [
        Utterance(id="u0", span=Span(start=0, end=30), text=long_text),
        Utterance(id="u1", span=Span(start=30, end=60),
                  text="Now the database migration adds a table and an index."),
    ]
    bounds = [BoundaryExplanation(ts=30.0, score=0.8, threshold=0.4)]
    units = enrich(build_units("v", CONFIG_C, bounds, utts, [], [], 60.0))

    assert len(units[0].transcript) > 400
    assert len(units[1].prev_summary) < len(units[0].transcript) / 2


def test_carried_entities_are_only_those_introduced_earlier():
    from vke.enrich import enrich
    units = enrich(_demo_units())
    assert units[0].carried_entities == []  # nothing precedes the first unit
    # "user" appears in units 0 and 1, so unit 1 carries it.
    assert "user" in units[1].entities
    assert "user" in units[1].carried_entities


def test_quality_penalises_evidence_free_boundaries():
    """A fixed-clock cut must not claim boundary confidence it does not have."""
    from vke.enrich import enrich
    from vke.chunker import build_units
    utts = [Utterance(id="u0", span=Span(start=0, end=90),
                      text="authentication token user password session login account")]

    fixed_bounds = [BoundaryExplanation(ts=30.0, score=0.0, threshold=0.0, signals=[])]
    fixed = enrich(build_units("v", CONFIG_A, fixed_bounds, utts, [], [], 90.0))
    signal = enrich(_demo_units())

    assert fixed[1].quality_parts["boundary_confidence"] == 0.0
    assert signal[1].quality_parts["boundary_confidence"] > 0.5


def test_validator_flags_short_and_silent_units():
    from vke.enrich import enrich, validate
    bounds = [BoundaryExplanation(ts=3.0, score=0.9, threshold=0.4)]
    units = enrich(build_units("v", CONFIG_C, bounds, [], [], [], 40.0))
    assert "too_short" in validate(units[0])
    assert "no_speech" in validate(units[0])


# --------------------------------------------------------------------------- #
# P4 universal adapter (plan sec.20)
# --------------------------------------------------------------------------- #
def test_universal_mapping_shape():
    from vke.enrich import enrich
    from vke.universal import to_universal

    unit = enrich(_demo_units())[1]
    u = to_universal(unit, "video://vid")

    assert u.id == unit.id
    assert u.source["source_type"] == "video"
    assert u.content["primary"] == unit.transcript
    assert u.context["preceding"] == unit.prev_summary
    assert u.relationships["previous"] == unit.prev_unit_id
    assert u.confidence == unit.quality


def test_universal_evidence_is_a_list_of_typed_locators():
    """The one design decision that makes another modality possible later."""
    from vke.enrich import enrich
    from vke.universal import to_universal

    u = to_universal(enrich(_demo_units())[1])
    assert isinstance(u.evidence, list) and len(u.evidence) == 1
    loc = u.evidence[0]
    assert loc.kind == "time_span"
    assert set(loc.ref) == {"start", "end"}
    # A PDF pipeline would emit kind="page_region" with a different ref, through
    # this same field and the same downstream code.


def test_universal_export_is_json_serialisable():
    import json as _json
    from vke.enrich import enrich
    from vke.universal import export_jsonl, export_units

    units = enrich(_demo_units())
    blob = export_units(units, "video://vid")
    assert len(blob) == len(units)
    _json.dumps(blob)  # must not raise

    lines = export_jsonl(units).splitlines()
    assert len(lines) == len(units)
    assert all(_json.loads(line)["source"]["source_type"] == "video" for line in lines)


def test_boundary_explanation_survives_into_universal_metadata():
    """Boundary explanation is the most portable idea in the project - keep it."""
    from vke.enrich import enrich
    from vke.universal import to_universal

    u = to_universal(enrich(_demo_units())[1])
    boundary = u.metadata["boundary"]
    assert boundary["signals"], "signal contributions must survive the mapping"
    assert boundary["signals"][0]["name"] == "visual"
