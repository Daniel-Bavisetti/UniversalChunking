"""Meeting semantics: tiered, anchored, honest about ambiguity."""

from __future__ import annotations

from cleave.chunkers import chunk
from cleave.graph import ContextGraph
from cleave.ingest_document import IngestResult
from cleave.meeting import (
    annotate_elements,
    classify_utterance,
    collect_unit_semantics,
    minutes,
    refine_ambiguous,
)
from cleave.models import ContentElement, RelationType


def _seg(i, text, t0, t1, speaker):
    return ContentElement(id=f"el_{i:04d}", kind="speech_segment", text=text,
                          t0=t0, t1=t1, speaker=speaker)


def _pipeline(elements):
    ing = IngestResult(elements=elements, title="standup", source_uri="m.m4a",
                       sha256="0" * 64)
    graph = ContextGraph(ing.elements)
    units, _profile = chunk(ing, graph)
    collect_unit_semantics(units)
    return units


# ───────── deterministic classification ─────────

def test_question_mark_is_a_question():
    sem = classify_utterance("What is the budget for Q3?")
    assert sem["type"] == "question" and sem["confidence"] >= 0.9
    assert sem["method"] == "deterministic"


def test_interrogative_without_question_mark_still_detected():
    # ASR drops punctuation constantly; the opener carries the signal.
    sem = classify_utterance("when do we ship the migration")
    assert sem["type"] == "question"


def test_strong_decision_language():
    sem = classify_utterance("We decided to go with PostgreSQL for the store.")
    assert sem["type"] == "decision" and sem["confidence"] >= 0.85


def test_action_item_with_owner_and_deadline_is_confident():
    sem = classify_utterance("Sarah will update the runbook by Friday.")
    assert sem["type"] == "action_item"
    assert sem["owner"] == "Sarah"
    assert "friday" in sem["deadline"].lower()
    assert sem["confidence"] >= 0.9


def test_bare_future_tense_is_ambiguous_not_asserted():
    # "this will improve latency" is narration, not an assignment.
    sem = classify_utterance("The new index will improve latency.")
    assert sem is None or sem["confidence"] < 0.75


def test_plain_statement_is_not_labelled():
    assert classify_utterance("The deploy finished around noon.") is None


# ───────── anchoring: speaker + timestamps survive ─────────

def test_annotation_preserves_speaker_and_timestamps():
    els = [_seg(0, "Who owns the rollback plan?", 10.0, 13.5, "Priya")]
    assert annotate_elements(els) == 1
    sem = els[0].meta["semantics"]
    assert sem["speaker"] == "Priya"
    assert sem["timestamp_start"] == 10.0 and sem["timestamp_end"] == 13.5


def test_semantics_ride_through_temporal_chunking():
    els = [
        _seg(0, "Okay, quick sync on the launch.", 0.0, 4.0, "Priya"),
        _seg(1, "We decided to delay the launch to March.", 4.0, 9.0, "Priya"),
        _seg(2, "Tom will notify the customers by Friday.", 9.0, 14.0, "Priya"),
    ]
    annotate_elements(els)
    units = _pipeline(els)
    sems = [s for u in units for s in u.metadata.get("semantics", [])]
    types = {s["type"] for s in sems}
    assert "decision" in types and "action_item" in types
    action = next(s for s in sems if s["type"] == "action_item")
    assert action["owner"] == "Tom" and action["timestamp_start"] == 9.0


# ───────── question → answer linking ─────────

def test_question_links_to_next_speakers_answer():
    els = [
        _seg(0, "How long does the migration take?", 0.0, 3.0, "Priya"),
        _seg(1, "About two hours including the index rebuild.", 3.2, 8.0, "Tom"),
    ]
    annotate_elements(els)
    units = _pipeline(els)
    q = next(u for u in units if u.temporal.speaker == "Priya")
    a = next(u for u in units if u.temporal.speaker == "Tom")
    assert any(r.type == RelationType.ANSWERED_BY and r.target_id == a.id
               for r in q.relationships)
    assert any(r.type == RelationType.ANSWERS and r.target_id == q.id
               for r in a.relationships)


def test_backchannel_does_not_count_as_an_answer():
    els = [
        _seg(0, "Can we cut the P99 latency further?", 0.0, 3.0, "Priya"),
        _seg(1, "Yeah.", 3.2, 3.6, "Tom"),
    ]
    annotate_elements(els)
    units = _pipeline(els)
    q = next(u for u in units if u.temporal.speaker == "Priya")
    assert not any(r.type == RelationType.ANSWERED_BY for r in q.relationships)


def test_same_speaker_never_answers_their_own_question():
    els = [
        _seg(0, "Should we enable the cache?", 0.0, 2.5, "Priya"),
        _seg(1, "I mean the write-through cache specifically.", 2.6, 6.0, "Priya"),
    ]
    annotate_elements(els)
    units = _pipeline(els)
    for u in units:
        assert not any(r.type == RelationType.ANSWERED_BY for r in u.relationships)


# ───────── tier 3: the LLM only touches the ambiguous minority ─────────

def _fake_provider(rows):
    import json as _json

    class Fake:
        name, model = "fake", "fake-model"

        def is_configured(self):
            return True

        def complete_json(self, prompt, *, system=None, schema=None, image=None):
            return _json.dumps({"results": rows}), {
                "model": self.model, "in_tokens": 500, "out_tokens": 60}

    return Fake()


def test_refinement_sends_only_ambiguous_candidates(monkeypatch):
    els = [
        _seg(0, "We decided to adopt the new schema.", 0.0, 3.0, "Priya"),   # confident
        _seg(1, "We should probably make sure to test that.", 3.0, 7.0, "Tom"),  # ambiguous
    ]
    annotate_elements(els)
    units = _pipeline(els)

    sent_prompts = []
    fake = _fake_provider([])
    original = fake.complete_json

    def spy(prompt, **kw):
        sent_prompts.append(prompt)
        return original(prompt, **kw)

    fake.complete_json = spy
    monkeypatch.setattr("cleave.llm.get_provider", lambda: fake)
    totals = refine_ambiguous(units)
    assert totals["candidates"] == 1                     # only the ambiguous one
    assert "make sure" in sent_prompts[0]
    assert "adopt the new schema" not in sent_prompts[0]  # confident item stays home


def test_refinement_can_demote_to_statement(monkeypatch):
    els = [_seg(0, "This will make onboarding easier for everyone.", 0.0, 4.0, "Tom")]
    annotate_elements(els)
    units = _pipeline(els)
    sems = [s for u in units for s in u.metadata.get("semantics", [])]
    if not sems:      # pattern may not even flag it — that is also correct
        return
    ref = f"{units[0].id}#0"
    monkeypatch.setattr("cleave.llm.get_provider",
                        lambda: _fake_provider([{"id": ref, "type": "statement"}]))
    totals = refine_ambiguous(units)
    assert totals["refined"] == 1
    assert sems[0]["type"] == "statement" and sems[0]["method"] == "llm"
    assert not sems[0]["ambiguous"]


def test_no_provider_leaves_candidates_marked_ambiguous():
    els = [_seg(0, "We should probably make sure to test that.", 0.0, 4.0, "Tom")]
    annotate_elements(els)
    units = _pipeline(els)
    totals = refine_ambiguous(units, use_llm=False)
    assert totals["api_calls"] == 0
    for u in units:
        for s in u.metadata.get("semantics", []):
            assert s["method"] == "deterministic"        # nothing pretended to be verified


# ───────── minutes ─────────

def test_minutes_collects_only_confident_items_and_marks_open_questions():
    els = [
        _seg(0, "What's blocking the release?", 0.0, 3.0, "Priya"),
        _seg(1, "The flaky auth test. We decided to quarantine it.", 3.2, 8.0, "Tom"),
        _seg(2, "Tom will fix the auth test by Friday.", 8.2, 12.0, "Priya"),
        _seg(3, "Is the changelog ready?", 12.2, 14.0, "Priya"),
    ]
    annotate_elements(els)
    units = _pipeline(els)
    m = minutes(units)
    assert len(m["decisions"]) == 1
    assert m["action_items"][0]["owner"] == "Tom"
    answered = {q["text"][:20]: q["answered"] for q in m["questions"]}
    assert answered["What's blocking the "] is True
    # every item is anchored
    for group in m.values():
        for item in group:
            assert item["unit_id"] and item["timestamp_start"] is not None
