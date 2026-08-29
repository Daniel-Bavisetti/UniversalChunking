"""Tests for the scope added after the MVP: providers, diarization, graph,
retrieval, Q&A and the merge/split refinement pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vke import signals  # noqa: E402
from vke.chunker import build_units, refine_boundaries  # noqa: E402
from vke.config import CONFIG_B, CONFIG_C  # noqa: E402
from vke.enrich import enrich  # noqa: E402
from vke.schemas import (  # noqa: E402
    BoundaryExplanation,
    SceneCut,
    SignalContribution,
    Span,
    SpeakerTurn,
    Utterance,
    VideoMeta,
)


def _demo_units():
    bounds = [
        BoundaryExplanation(
            ts=30.0, score=0.8, threshold=0.4,
            signals=[SignalContribution(name="visual", raw=1.0, normalized=1.0,
                                        weight=0.4, contribution=0.4)]),
        BoundaryExplanation(ts=60.0, score=0.8, threshold=0.4),
    ]
    utts = [
        Utterance(id="u0", span=Span(start=0, end=30),
                  text="We configure authentication. The user signs in with a password."),
        Utterance(id="u1", span=Span(start=30, end=60),
                  text="The authentication token expires. The user must sign in again."),
        Utterance(id="u2", span=Span(start=60, end=90),
                  text="Now the database migration adds a table and an index."),
    ]
    return enrich(build_units("vid", CONFIG_C, bounds, utts, [], [], 90.0))


# --------------------------------------------------------------------------- #
# the ablation must stay controlled
# --------------------------------------------------------------------------- #
def test_audio_only_and_vke_differ_in_exactly_one_weight():
    """The comparison claim depends on this and nothing else.

    If a second weight ever varies between B and C, "the visual signal found
    these boundaries" stops being attributable, and the demo argument goes with it.
    """
    differing = [
        k for k in set(CONFIG_B.weights) | set(CONFIG_C.weights)
        if CONFIG_B.weights.get(k, 0.0) != CONFIG_C.weights.get(k, 0.0)
    ]
    assert differing == ["visual"], f"expected only 'visual' to differ, got {differing}"
    assert CONFIG_B.weights["visual"] == 0.0
    assert CONFIG_C.weights["visual"] > 0.0


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
def test_offline_providers_are_available_and_honest():
    from vke import providers

    assert providers.OfflineLLM().complete("anything") == ""
    result = providers.OfflineVision().describe(Path("nope.jpg"), "transcript")
    # The offline path must never invent a description.
    assert result.description == ""
    assert result.source == "heuristic"


def test_provider_selection_falls_back_without_a_key():
    from vke import providers

    settings = providers.ProviderSettings(llm="openai", vision="openai", api_key="")
    assert providers.get_llm(settings).name == "offline"
    assert providers.get_vision(settings).name == "offline"


def test_vision_parsing_tolerates_a_fenced_json_block():
    from vke.providers import _parse_vision

    fence = "```"
    raw = (
        "Here you go:\n" + fence + "json\n"
        '{"description": "a login form", "ocr_text": ["Username", "Password"], '
        '"objects": ["laptop"], "actions": ["typing"]}\n' + fence
    )
    parsed = _parse_vision(raw)
    assert parsed.description == "a login form"
    assert parsed.ocr_text == ["Username", "Password"]
    assert parsed.objects == ["laptop"]
    assert parsed.source == "vlm"


def test_vision_parsing_survives_a_model_that_ignores_the_format():
    from vke.providers import _parse_vision

    parsed = _parse_vision("A presenter stands beside a slide.")
    assert "presenter" in parsed.description
    assert parsed.source == "vlm"


# --------------------------------------------------------------------------- #
# diarization
# --------------------------------------------------------------------------- #
def test_silhouette_rejects_splitting_one_homogeneous_group():
    """Guards a real bug: a between/within ratio rises monotonically with k, so
    it always picked the largest k and reported four speakers for one voice."""
    from vke.diarize import _silhouette

    blob = np.random.default_rng(0).normal(0, 1, size=(30, 3))
    forced = np.array([i % 3 for i in range(30)])
    assert _silhouette(blob, forced) < 0.45

    separated = np.vstack([
        np.random.default_rng(1).normal(-6, 0.3, size=(15, 3)),
        np.random.default_rng(2).normal(+6, 0.3, size=(15, 3)),
    ])
    assert _silhouette(separated, np.array([0] * 15 + [1] * 15)) > 0.6


def test_diarization_degrades_quietly_without_audio(tmp_path: Path):
    from vke.diarize import diarize

    fake = tmp_path / "silent.mp4"
    fake.write_bytes(b"not a video")
    assert diarize(fake, [Utterance(id="u0", span=Span(start=0, end=5), text="hi")]) == []


def test_speaker_changes_only_fire_on_a_handover():
    from vke.diarize import speaker_changes

    turns = [
        SpeakerTurn(span=Span(start=0, end=5), speaker="speaker_00"),
        SpeakerTurn(span=Span(start=5, end=10), speaker="speaker_00"),
        SpeakerTurn(span=Span(start=10, end=15), speaker="speaker_01"),
    ]
    changes = speaker_changes(turns)
    assert len(changes) == 1
    assert changes[0][0] == 10.0


def test_speaker_signal_is_silent_without_handovers():
    grid = signals.build_grid(60.0)
    assert np.allclose(signals.speaker_curve([], grid).normalized, 0.0)
    peaked = signals.speaker_curve([(30.0, 0.8)], grid)
    assert float(grid[int(np.argmax(peaked.normalized))]) == pytest.approx(30.0, abs=1.0)


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def _graph_fixture():
    from vke import graph as graph_mod

    units = _demo_units()
    meta = VideoMeta(video_id="vid", filename="v.mp4", duration=90.0,
                     fps=25.0, width=640, height=480)
    return graph_mod, graph_mod.build(meta, units, [SceneCut(ts=30.0)]), units


def test_graph_has_the_expected_node_and_edge_types():
    graph_mod, g, units = _graph_fixture()
    stats = g.to_dict()["stats"]
    assert stats["nodes_by_type"]["Event"] == len(units)
    assert stats["nodes_by_type"]["Video"] == 1
    assert graph_mod.PRECEDES in stats["edges_by_type"]
    assert graph_mod.MENTIONS in stats["edges_by_type"]


def test_graph_temporal_chain_follows_unit_order():
    graph_mod, g, units = _graph_fixture()
    precedes = [(e.source, e.target) for e in g.edges if e.type == graph_mod.PRECEDES]
    assert precedes == [
        (f"event:{a.id}", f"event:{b.id}") for a, b in zip(units, units[1:])
    ]


def test_graph_expansion_reaches_events_through_shared_entities():
    """The graph must do real work in retrieval, not merely be drawn."""
    graph_mod, g, _units = _graph_fixture()
    entity = next(n for n in g.nodes.values() if n.type == graph_mod.ENTITY)
    reached = g.expand([entity.id], hops=1, node_types=[graph_mod.EVENT])
    assert reached, "an entity must reach the events that mention it"


def test_graph_serialises_and_reloads():
    _graph_mod, g, _units = _graph_fixture()
    blob = json.loads(json.dumps(g.to_dict()))
    assert blob["stats"]["node_count"] == len(g.nodes)
    assert len(blob["edges"]) == len(g.edges)


# --------------------------------------------------------------------------- #
# retrieval and Q&A
# --------------------------------------------------------------------------- #
def test_search_ranks_the_topically_correct_unit_first():
    from vke.retrieve import search

    units = _demo_units()
    hits = search("database migration table index", units, top_k=3)
    assert hits, "expected at least one hit"
    top = next(u for u in units if u.id == hits[0].unit_id)
    assert "database" in top.transcript


def test_every_search_hit_is_timestamp_grounded():
    from vke.retrieve import search

    for hit in search("authentication password", _demo_units(), top_k=5):
        assert hit.span.end > hit.span.start   # a real, clickable interval
        assert hit.reason                       # with a stated reason
        assert hit.unit_id


def test_search_returns_nothing_for_an_unrelated_query():
    from vke.retrieve import search

    assert search("quantum chromodynamics", _demo_units()) == []


def test_graph_expansion_widens_the_result_set():
    from vke import graph as graph_mod
    from vke.retrieve import search

    units = _demo_units()
    meta = VideoMeta(video_id="vid", filename="v.mp4", duration=90.0,
                     fps=25.0, width=640, height=480)
    g = graph_mod.build(meta, units, [])
    without = search("migration", units, top_k=6)
    with_graph = search("migration", units, graph=g, top_k=6)
    assert len(with_graph) >= len(without)
    if len(with_graph) > len(without):
        assert any("graph" in h.reason for h in with_graph)


def test_time_hints_are_parsed_from_natural_language():
    from vke.retrieve import parse_time_hint

    assert parse_time_hint("what happened at 2:15?")[0] == pytest.approx(135.0)
    assert parse_time_hint("show me 3 minutes in")[0] == pytest.approx(180.0)
    assert parse_time_hint("what came before the error")[1] == "before"
    assert parse_time_hint("and then what followed")[1] == "after"
    assert parse_time_hint("authentication")[0] is None


def test_ask_offline_returns_evidence_and_never_fabricates():
    from vke.retrieve import ask

    result = ask("what is the migration about?", _demo_units())
    assert result["answer_source"] == "extractive_offline"
    assert result["evidence"], "an answer must be backed by evidence"
    for item in result["evidence"]:
        assert item["span"]["end"] > item["span"]["start"]


def test_ask_says_so_when_nothing_matches():
    from vke.retrieve import ask

    result = ask("quantum chromodynamics", _demo_units())
    assert result["answer_source"] == "no_evidence"
    assert result["evidence"] == []


# --------------------------------------------------------------------------- #
# merge / split refinement
# --------------------------------------------------------------------------- #
def test_refine_merges_a_spurious_split_in_continuous_speech():
    text = "authentication token user password session login account credential"
    utts = [
        Utterance(id=f"u{i}", span=Span(start=i * 5.0, end=i * 5.0 + 5.0), text=text)
        for i in range(12)
    ]
    grid = signals.build_grid(60.0)
    bounds = [BoundaryExplanation(ts=10.0, score=0.5, threshold=0.4),
              BoundaryExplanation(ts=25.0, score=0.5, threshold=0.4)]
    refined, stats = refine_boundaries(
        bounds, utts, 60.0, grid, np.zeros(grid.size))
    assert stats["merged"] >= 1
    assert len(refined) < len(bounds)


def test_refine_leaves_a_genuine_topic_change_alone():
    utts = [
        Utterance(id="u0", span=Span(start=0, end=20),
                  text="authentication token password session login credential user"),
        Utterance(id="u1", span=Span(start=20, end=40),
                  text="database migration schema table column index query rollback"),
    ]
    grid = signals.build_grid(40.0)
    bounds = [BoundaryExplanation(ts=20.0, score=0.9, threshold=0.4)]
    refined, stats = refine_boundaries(
        bounds, utts, 40.0, grid, np.zeros(grid.size))
    assert stats["merged"] == 0
    assert [b.ts for b in refined] == [20.0]


# --------------------------------------------------------------------------- #
# the adapter boundary
# --------------------------------------------------------------------------- #
def test_nothing_downstream_of_chunker_imports_a_video_module():
    """chunker.py is the last video-specific module (docs/ARCHITECTURE.md sec.3).

    This rule is the entire multi-modality integration contract: a future PDF
    pipeline can reuse storage, search, the graph, export and the UI only because
    none of them know what a frame or an audio track is. A stray import here
    would quietly cost that, so it is enforced rather than documented.
    """
    video_specific = {"media", "asr", "signals", "texttiling", "diarize"}
    downstream = ["enrich", "graph", "retrieve", "store", "universal"]

    offenders = []
    for name in downstream:
        source = (ROOT / "vke" / f"{name}.py").read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("from .", "import .")):
                continue
            module = stripped.split()[1].lstrip(".").split(".")[0]
            if module in video_specific:
                offenders.append(f"{name}.py: {stripped}")

    assert not offenders, (
        "these modules sit downstream of chunker.py and must not import video "
        f"internals: {offenders}"
    )
