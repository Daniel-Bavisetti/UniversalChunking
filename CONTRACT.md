# Cleave Modality Contract

Any external modality worker (the video repo, future image/audio workers) integrates with
Cleave by producing data conforming to `cleave/models.py`. That file is frozen; this page
is the integration guide.

## What a worker must produce

Either of two levels of integration:

### Level 1 — elements in, Cleave chunks (preferred)

Emit a JSON array of **ContentElement** objects. Cleave runs its own graph → router →
chunking → enrichment over them.

```json
{
  "id": "el_0042",
  "kind": "speech_segment",            // see ELEMENT_KINDS in models.py
  "text": "So the budget for Q2 is on this slide...",
  "t0": 84.2, "t1": 97.8,              // seconds, required for temporal routing
  "speaker": "SPEAKER_01",             // optional
  "page": null, "bbox": null,
  "parent_id": null, "level": null,
  "meta": {"visual_summary": "presenter points at a bar chart"}   // video: keyframe caption
}
```

Rules:
- `kind` for video: `speech_segment` for spoken spans, `visual_event` for scene-level
  visual descriptions. Timestamps (`t0`/`t1`) are what trigger Cleave's temporal chunker.
- `kind: "table"` elements should carry `meta.grid` (list of row-lists), `meta.header_row`,
  and optionally `meta.sheet`. A table-dominated element set triggers the tabular route,
  which profiles columns and emits a schema card plus header-repeating row groups.
- `meta.visual_summary` on a speech_segment carries what was visible while it was said —
  it lands in the unit's `metadata` and is displayed alongside the transcript.
- Order the array by time (or reading order). Cleave adds NEXT/PREVIOUS itself.

### Level 2 — finished units in

Emit **KnowledgeUnit** objects (see `KnowledgeUnit.to_dict()` for the exact JSON shape).
Required fields: `id, content, modality, context{...}, provenance{source_uri}, decision
{strategy, reason}`; use `temporal{start_s, end_s, speaker}` for timed content. Cleave
renders them in the same UI natively.

## Semantics workers must respect

- **Never sever what belongs together**: don't emit a unit whose meaning lives in another
  unit you also emitted (a caption without its figure description, a mid-sentence cut).
- **decision.reason is user-facing**: one honest sentence about why the boundary exists.
- **Measured costs only**: if you called a model, count it in `decision.llm_calls` /
  `cost_usd`; zero means zero.
- IDs are worker-scoped; Cleave prefixes them on import, so collisions are fine.

## Wire format

A single JSON file: `{"contract": 1, "source_uri": "...", "elements": [...]}` or
`{"contract": 1, "source_uri": "...", "units": [...]}`.

`contract` is required and must be `1`; anything else is rejected rather than guessed at.
Element and unit ids are prefixed on import, so two workers can use the same local ids
safely.

## Try it

`tests/fixtures/video_contract.json` is a working three-segment example. Upload it like any
other file — it routes `temporal`, produces speaker-attributed units, and carries each
segment's `visual_summary` through to the unit. That round trip is the integration test:
if your worker's output behaves the same way, it is done.

```bash
uv run python -c "
from cleave.ingest_contract import load_contract
ing, units = load_contract('tests/fixtures/video_contract.json')
print(len(ing.elements), 'elements imported')"
```
