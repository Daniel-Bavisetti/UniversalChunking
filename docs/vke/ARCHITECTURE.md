# Architecture

How VKE is put together, and why each piece exists. For the data flow see
[PIPELINE.md](PIPELINE.md); for day-to-day use see [WORKFLOW.md](WORKFLOW.md).

---

## 1. The one idea

Everything here serves a single claim:

> **Extraction determines chunking.** What we measure in the speech and the
> picture decides where a chunk begins — and every chunk can show you the
> arithmetic that put its boundary there.

A module earns its place by contributing to that claim, or by making it
demonstrable. Nothing else is in the tree.

---

## 2. Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  static/index.html          one screen: player · timeline · unit ·   │
│                             search · graph · diagnostics             │
├──────────────────────────────────────────────────────────────────────┤
│  api.py                     FastAPI · byte-range streaming · jobs    │
├──────────────────────────────────────────────────────────────────────┤
│  pipeline.py                orchestration · staging · tracing        │
├───────────────┬──────────────────────────────────┬───────────────────┤
│  EXTRACTION   │  DECISION                        │  PRODUCT          │
│  media.py     │  texttiling.py                   │  chunker.py       │
│  asr.py       │  signals.py                      │  enrich.py        │
│  diarize.py   │     ↓                            │  graph.py         │
│  providers.py │  s(t) = Σ wᵢ · signalᵢ(t)        │  retrieve.py      │
│  detect.py    │                                  │  universal.py     │
├───────────────┴──────────────────────────────────┴───────────────────┤
│  schemas.py (the contract)   ·   store.py (JSON per video)           │
│  config.py  (weights, versions, the three configurations)            │
└──────────────────────────────────────────────────────────────────────┘
```

**Dependency direction is strictly downward.** `schemas.py` imports nothing of
ours; `config.py` imports only stdlib. Nothing in the decision layer knows the
API exists.

---

## 3. The adapter boundary

One rule carries the entire multi-modality story:

> **`chunker.py` is the last video-specific module.** Nothing downstream of it
> may import `media`, `asr`, `signals`, `texttiling` or `diarize`.

Left of that line, code may assume video freely — frames, audio tracks, scene
cuts. Right of it, code sees only `KnowledgeUnit`.

```
media · asr · diarize · detect · signals · texttiling → chunker → KnowledgeUnit
├──────────── VIDEO-SPECIFIC ────────────────┤            ├── PORTABLE ──→
                                                          enrich · graph
                                                          retrieve · store
                                                          api · universal · UI
```

A future PDF pipeline supplies its own extraction (`pdf_extract.py`) and its own
boundary signals (`heading`, `font_change`, `whitespace_gap`), emits
`KnowledgeUnit`s, and inherits storage, search, the graph, export and the whole
comparison UI without a line of change. That is why `SignalContribution.name` is
`str` and not a `Literal` — a one-character decision that keeps
`BoundaryExplanation` modality-neutral.

Verify the rule holds:

```bash
grep -nE "^from \.(media|asr|signals|texttiling|diarize|detect)" \
     vke/{enrich,graph,retrieve,store,universal,api}.py   # must print nothing
```

---

## 4. Module by module

| Module | Lines | Responsibility | Notes |
|---|---:|---|---|
| `schemas.py` | ~230 | The data contract | Add fields freely; never rename or remove |
| `config.py` | ~110 | Weights, thresholds, versions, the 3 configs | The whole experiment lives here |
| `media.py` | ~185 | Probe, single-pass frame features, keyframes, scene cuts | One decode pass produces every visual measurement |
| `asr.py` | ~155 | Speech → utterances with **absolute** timestamps | faster-whisper, `.srt` sidecar fallback |
| `diarize.py` | ~215 | Heuristic speaker segmentation | PyAV + numpy; no torch, no sklearn |
| `texttiling.py` | ~145 | Hearst depth scoring | Split out because it is intricate enough to test alone |
| `signals.py` | ~285 | The four signals, normalization, fusion | Curves computed **once**, reused by every config |
| `detect.py` | ~290 | Object detection + OCR on selected frames | ONNX on CPU; never raises, never feeds a signal |
| `chunker.py` | ~375 | Peaks → boundaries → units, merge/split refinement | The adapter boundary |
| `enrich.py` | ~200 | Summary, context, carried entities, quality, validator | Optional LLM title polish |
| `graph.py` | ~215 | Event/entity graph + expansion | Adjacency dicts; no graph library |
| `retrieve.py` | ~250 | TF-IDF search, graph expansion, temporal filter, Q&A | Every hit is timestamp-grounded |
| `providers.py` | ~265 | LLM + vision behind two Protocols | One OpenAI-compatible adapter for every vendor |
| `universal.py` | ~135 | `KnowledgeUnit` → `UniversalKnowledgeUnit` | Pure mapping, no I/O |
| `store.py` | ~180 | Per-video JSON artifacts | Atomic writes; extraction is cached |
| `pipeline.py` | ~200 | Stage orchestration, progress, traces | The only module that knows the order |
| `api.py` | ~300 | HTTP surface | Byte-range streaming lives here |

---

## 5. Decisions worth knowing

### One decode pass
`media.extract_frames` walks the video once at ~2 fps and emits HSV histogram,
edge density, motion and brightness per sample. Every visual measurement comes
from that pass; nothing re-decodes.

### Curves computed once, reused by three configs
`signals.compute_curves` runs a single time. The three configurations then apply
different weight vectors to the *same* curves. That is why the comparison costs
almost nothing, and why it is a genuine ablation rather than three pipelines.

### Two-pass compute
Pass 1 is cheap and covers the whole video: ASR, frame features, scene cuts,
diarization. It **finds** boundaries. Pass 2 is expensive and runs only on the
keyframes already extracted: VLM description, on-screen text, objects, actions.
It **enriches** them.

Pass 2 never feeds boundary detection, and that is not an oversight — it is
forced. Model output only exists near candidate boundaries, and candidates come
from the score, so a score that depended on model output would be circular. This
is why `action_change` and `entity_change` are not signals.

### Semantic enrichment is observed once, per timestamp
`detect.py` picks <=40 representative frames for the WHOLE video and runs the
detector and OCR over them once. Each config's units then absorb whatever
observations fall inside their span. Enriching per config would triple the cost
and, worse, make the headline comparison unfair - the baselines would carry no
visual evidence at all. Identical evidence, different boundaries, is the only
version of that comparison worth showing.

The models are ONNX on **CPU** (~120ms/frame for yolov10n, ~1s/frame for PP-OCR).
`onnxruntime` and `huggingface-hub` are already hard dependencies of
`faster-whisper`, so object detection adds no new package. The GPU is never
touched: no CUDA, no cuDNN, no VRAM ceiling.

### A measurement is never presented as understanding
The distinction is structural, not editorial:

* `VisualObservation.source` has no `heuristic` member. An observation exists
  only when real inference produced it; edge density, motion and brightness stay
  in `FrameFeature` and `visual_context`.
* `objects` / `ocr_text` / `actions` are pure projections of `observations`, so
  nothing can populate them without leaving provenance behind.
* `confidence` is `float | None`, never a defaulted `1.0`. A detector and an OCR
  line ship real scores; a VLM ships none and records `None`.
* `actions` is empty unless a real semantic model produced one. It is never
  derived from motion - "high motion" measures pixel change, and is not a claim
  that anyone is doing anything.
* "found nothing" and "never ran" get different strings in
  `provenance["enrichment"]`, and the UI prints whichever applies rather than
  showing an empty list a viewer could read either way.

`tests/test_enrichment_honesty.py` asserts all of it.

### No embedding model on the critical path
The semantic signal is TextTiling over term-frequency vectors. Deterministic,
dependency-free, and a citable algorithm. Embeddings appear only in retrieval,
where TF-IDF is sufficient at this scale.

### Storage is a directory of JSON
At a few hundred units per video, a database buys nothing and a directory you can
open in an editor is worth a great deal at 2am. Writes are atomic
(`tmp` + `replace`) so a crash never leaves half an artifact.

### Offline is the default, and it is honest
No API key is required for anything on the critical path. Where a model would
normally supply semantics, the offline path reports what it *measured* and
records `source: "heuristic"`, so a measurement can never be mistaken for a
description. `ask()` returns evidence rather than a fabricated narrative.

---

## 6. Failure behaviour

Every stage degrades rather than aborting:

| Failure | Behaviour |
|---|---|
| No audio track | `has_audio=False`; utterances empty; visual and scene signals still drive boundaries |
| ASR model unavailable | Falls back to `.srt`/`.vtt` sidecar, logged loudly |
| No scene cuts | `visual` falls back to the continuous histogram signal alone |
| Diarization fails | Skipped; `speaker` curve is all zeros and contributes nothing |
| Vision/LLM provider errors | Counted in `USAGE.failures`; extractive output is kept |
| Object detector unavailable | `provenance.enrichment.objects` records why; `objects` empty; boundaries unchanged |
| OCR not installed | Records `unavailable: rapidocr not installed`; boundaries unchanged |
| Model download fails | Same as unavailable - the reason is captured, capped to one line |
| No action model and no VLM | `actions` stays empty and says `not_requested`; nothing is inferred from motion |
| Silhouette below threshold | Reports **one** speaker rather than inventing turns |
| Corrupt video | `probe` raises before any expensive work starts |

The pattern: a missing signal contributes zero, it never contributes noise.

---

## 7. Versioning

`PIPELINE_VERSION`, `CHUNKER_VERSION` and `SCHEMA_VERSION` are stamped into every
unit's `provenance`, alongside the exact weight vector and provider names used.
Two units from different runs are always distinguishable.

---

## 8. Relationship to VideoRAG

`VideoRAG/` is untouched. It is reused as the **baseline we measure against**:
`chunking_by_video_segments` is 53 lines of fixed-30s windows packed to a token
budget, reimplemented here as Config A.

Its modules could not be imported on this Python anyway — `_videoutil/*` and
`vdb_nanovectordb.py` import `torch` / `imagebind` / `faster_whisper` at module
load. Two of its defects motivated the design: segment times recovered by
`eval()`-ing a filename substring, and ASR timestamps never converted from
segment-relative to absolute. Every timestamp in VKE is absolute, which is what
makes jump-to-moment exact.
