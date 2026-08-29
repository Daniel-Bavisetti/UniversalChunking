# VKE — Video Knowledge Engine

**Multimodal temporal event chunking.** Traditional video RAG cuts a video on a fixed
clock and turns it into text. VKE measures what is happening in the speech *and* the
picture, and lets those measurements decide where a chunk begins.

Built for the "AI-Powered Universal Chunking and Information Extraction" challenge,
scoped deliberately to video — the hardest modality — on top of
[HKUDS/VideoRAG](https://github.com/HKUDS/VideoRAG).

---

## The claim, and how to check it

Three chunking configurations run through **one code path**. They differ only by a weight
vector, so the comparison is a true ablation rather than three implementations:

| Config | Weights | What it represents |
|---|---|---|
| **Fixed 30s** | — | The VideoRAG baseline, reimplemented. No signal is consulted. |
| **Audio-only** | `semantic .40 · silence .20 · speaker .15 · visual .00` | Everything VKE does except look at the picture. |
| **VKE Multimodal** | `semantic .40 · silence .20 · speaker .15 · visual .40` | The same, with vision switched on. |

B and C are **identical except for the visual weight**, so any boundary that appears in C
and not B is attributable to vision and nothing else. That is a controlled ablation, and
`test_audio_only_and_vke_differ_in_exactly_one_weight` fails if it ever stops being one.

On the bundled fixture (ground truth: a visual-only boundary, a semantic-only boundary,
and one where both change):

```
config            units  mean len    F1@2s    F1@5s     err  quality
Fixed 30s             4     26.2s     0.00     0.33    7.5s    0.480
Audio-only            4     26.2s     0.33     0.33    3.9s    0.601
VKE Multimodal        4     26.2s     0.67     1.00    0.9s    0.648

   reference  kind                Fixed 30s     Audio-only   VKE Multimodal
       25.5s  visual_only             FOUND         missed            FOUND
       52.8s  semantic_only          missed          FOUND            FOUND
       79.3s  both                   missed         missed            FOUND

  Boundaries VKE finds that audio-only misses: 25.5s, 79.3s
  Expensive model calls: 0
```

Reproduce with `python scripts/evaluate.py`. The quality column is scored
independently of the boundary metrics, and ranks the configs the same way -
`Fixed 30s` scores **0.000** on boundary confidence because no measurement stands
behind its cuts, and it says so.

Every chunk carries the arithmetic that produced its boundary:

```
WHY THIS BOUNDARY            score 0.462   threshold 0.270
  visual     1.00 × 0.40 = 0.400
  silence    0.31 × 0.20 = 0.062
  semantic   0.00 × 0.40 = 0.000
  snapped 25.0s → 25.4s to land on an utterance edge
```

---

## Quick start

Runs offline. **No API keys, no GPU, no system ffmpeg, no torch.**

```bash
pip install -r requirements.txt

python scripts/make_fixture.py          # build the test video + ground truth
python scripts/process.py data/fixture.mp4
python -m uvicorn vke.api:app --port 8000
# open http://localhost:8000
```

To process your own video:

```bash
python scripts/process.py path/to/talk.mp4 --id my_talk
```

Or drag a file onto the page.

### Other commands

```bash
python scripts/spike.py                       # prove the risky dependencies work
python scripts/inspect_signals.py             # plot s(t) vs ground truth
python scripts/evaluate.py --jsonl out.jsonl  # metrics + per-boundary rows
python -m pytest tests/ -q                    # 57 tests
```

---

## How it works

```
mp4
 ├─▶ faster-whisper ──────────▶ Utterance[]  (absolute timestamps)
 ├─▶ OpenCV single pass @2fps ▶ FrameFeature[] + keyframe JPEGs
 ├─▶ PySceneDetect ───────────▶ SceneCut[]
 └─▶ diarize (PyAV+numpy) ────▶ SpeakerTurn[]
                                     │
                          UNIFIED TIMELINE (one absolute axis)
                                     │
  s(t) = .40·semantic + .40·visual + .20·silence + .15·speaker
                                     │
     adaptive threshold μ+kσ → NMS → snap to utterance edge
                                     │
                    KnowledgeUnit[] ──▶ graph · search · Q&A · JSON
```

### The four signals

- **semantic** — TextTiling (Hearst 1997) depth score over the transcript. Two details
  carry it: blocks are sized in *seconds of speech* (scale-invariant across speaking
  paces), and boundary strength is the **depth score**, not raw dissimilarity. In short
  speech `1 − cosine` sits near 1.0 everywhere and discriminates nothing; the depth score
  cancels that baseline out. **No embedding model.**
- **visual** — windowed HSV-histogram distance, `max`'d with a scene-cut kernel. Taking
  the max rather than adding a fourth term is what stops a cut and the histogram jump
  around it — the same physical event — from double-counting.
- **silence** — pauses between utterances. Free, independent, and the strongest single
  cue in talks.
- **speaker** — handovers from heuristic diarization (PyAV + numpy, no torch). Contributes
  exactly zero on single-presenter footage, and says so rather than inventing turns.

### Signals deliberately *not* built

- `action_change` / `entity_change` — **circular**. They need model output that only runs
  near candidate boundaries, which requires the score, which would require them.
- `topic_shift` — the same measurement as `semantic` at a different window.
- `scene_change` as its own term — double-counts `visual`.
- `scene_change` as its own term — double-counts `visual`, so cuts are folded in with a
  `max` instead.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/videos` | Upload; returns `{job_id, video_id}` immediately |
| `GET /api/jobs/{id}` | Stage, percent, per-stage timing trace |
| `GET /api/videos` · `/{id}` | Library and metadata |
| `GET /api/videos/{id}/units?config=` | Knowledge Units (all configs, or one) |
| `GET /api/videos/{id}/curves` | Signal curves, scores, utterances — drives the UI |
| `GET /api/videos/{id}/stream` | Video with **HTTP byte-range** support (seeking) |
| `GET /api/videos/{id}/keyframe/{unit}.jpg` | Per-unit keyframe |
| `POST /api/videos/{id}/search` | Timestamp-grounded moment search |
| `POST /api/videos/{id}/ask` | Q&A with cited, clickable evidence |
| `GET /api/videos/{id}/graph` | Event/entity graph |
| `GET /api/videos/{id}/trace` | Stage timings, providers, model-call counts |
| `GET /api/videos/{id}/export?config=&schema=&fmt=` | `schema=vke\|universal`, `fmt=json\|jsonl` |
| `GET /api/configs` | The three chunking configurations |

### Universal output

The problem statement asks for **universal** chunking. We scoped to video, so the
answer to "what makes it universal?" is a platform-neutral output contract:

```jsonc
{
  "id": "vke_multimodal_001",
  "source":   { "source_id": "fixture", "source_type": "video", "source_reference": "video://fixture" },
  "content":  { "primary": "…", "structured": { "title": "…", "entities": [...] } },
  "context":  { "preceding": "…", "following": "…", "carried_entities": [...] },
  "evidence": [ { "kind": "time_span", "ref": { "start": 25.44, "end": 52.42 },
                  "preview": "/api/videos/fixture/keyframe/vke_multimodal_001.jpg" } ],
  "relationships": { "previous": "…", "next": "…", "related": [...] },
  "metadata": { "modality": "video", "boundary": {...}, "quality": {...} },
  "confidence": 0.715,
  "provenance": {...}
}
```

Two decisions carry all the future compatibility, and both are free at export
time: **`evidence` is a list of typed `Locator`s** (a PDF chunk may span two
pages; it emits `kind: "page_region"` through the same field), and **`metadata`
is an open dict**. `BoundaryExplanation` travels with the unit because "why does
this chunk start here" is exactly as meaningful for a PDF as for a video.

A future PDF pipeline supplies its own extraction and its own boundary signals,
emits `KnowledgeUnit`s, and reuses storage, the API, the UI and this exporter
unchanged. That is the entire integration cost, paid up front:
`vke/universal.py`, ~120 lines.

---

## Knowledge Unit

```jsonc
{
  "id": "vke_multimodal_001",
  "video_id": "fixture",
  "span": { "start": 25.44, "end": 52.42, "duration": 26.98 },
  "title": "The session token expires after a while, so the user has to s…",
  "transcript": "…",
  "visual_context": "low on-screen text density, static, dark frame (edges 0.015, …)",
  "scene_ids": [1],
  "keyframe_url": "/api/videos/fixture/keyframe/vke_multimodal_001.jpg",
  "entities": ["user", "token", "expires", "password", "session"],
  "boundary": {
    "ts": 25.44, "score": 0.462, "threshold": 0.270,
    "snapped_from": 25.0,
    "signals": [
      { "name": "visual",   "raw": 0.994, "normalized": 1.00, "weight": 0.40, "contribution": 0.400 },
      { "name": "silence",  "raw": 0.311, "normalized": 0.31, "weight": 0.20, "contribution": 0.062 },
      { "name": "semantic", "raw": 0.000, "normalized": 0.00, "weight": 0.40, "contribution": 0.000 }
    ]
  },
  "summary": "The session token expires after a while… we revoke it immediately.",
  "prev_summary": "The user signs in with the username and a password…",
  "next_summary": "Now we move on to the database migration…",
  "carried_entities": ["user", "token", "password"],
  "quality": 0.715,
  "quality_parts": {
    "semantic_coherence": 0.185, "boundary_confidence": 0.857,
    "length_sanity": 0.819, "context_completeness": 1.0, "overall": 0.715
  },
  "flags": [],
  "related_unit_ids": ["vke_multimodal_000"],
  "prev_unit_id": "vke_multimodal_000",
  "next_unit_id": "vke_multimodal_002",
  "provenance": { "pipeline_version": "0.1.0", "weights": {...}, "providers": {...} }
}
```

`carried_entities` are the terms introduced *earlier* that this unit still leans
on — the ones a reader needs primed. `prev_summary`/`next_summary` are summarized,
never copy-pasted: duplicating adjacent text would re-introduce the redundancy
event chunking exists to remove.

`visual_context` is **measured, not generated**. With no vision model in the offline path,
inventing a scene description would be dishonest; the keyframe image does that job.

---

## Relationship to VideoRAG

`VideoRAG/` is **untouched** and still runs on a GPU box. We reuse it conceptually rather
than by vendoring:

> **VideoRAG is Baseline A.** Its `chunking_by_video_segments`
> ([_op.py:68](VideoRAG/VideoRAG-algorithm/videorag/_op.py#L68)) is 53 lines: fixed 30s
> segments packed to a token budget. We reimplement it faithfully as the comparison
> baseline and measure against it.

Its own modules could not be imported here anyway — `_videoutil/*` and
`vdb_nanovectordb.py` import `torch` / `imagebind` / `faster_whisper` at module load, and
none of that installs on Python 3.13 with 4GB VRAM.

Two defects in the baseline motivated the design: it recovers segment times by `eval()`-ing
a filename substring ([_op.py:721](VideoRAG/VideoRAG-algorithm/videorag/_op.py#L721)), and
never converts segment-relative ASR timestamps to absolute
([asr.py:27](VideoRAG/VideoRAG-algorithm/videorag/_videoutil/asr.py#L27)). Every timestamp
in VKE is absolute, which is what makes jump-to-moment exact.

---

## Layout

```
vke/
  schemas.py    the data contract          media.py      decode, features, keyframes
  config.py     weights + the 3 configs    asr.py        speech -> absolute timestamps
  signals.py    the 3 signals + fusion     texttiling.py Hearst depth scoring
  chunker.py    boundaries -> units        store.py      per-video JSON artifacts
  pipeline.py   orchestration              api.py        FastAPI + byte-range streaming
  enrich.py     context + quality scoring  universal.py  platform-neutral export
  texttiling.py Hearst depth scoring       diarize.py    heuristic speaker turns
  graph.py      event/entity graph         retrieve.py   search + grounded Q&A
  providers.py  optional LLM/VLM (one OpenAI-compatible adapter for all vendors)
static/index.html    the single screen (no build step, no CDN)
scripts/   make_fixture · spike · process · inspect_signals
tests/     57 tests
```

Nothing downstream of `chunker.py` imports `media`, `asr`, or `signals`. That single rule
is the adapter boundary: a future PDF pipeline supplies its own extraction and its own
boundary signals, emits `KnowledgeUnit`s, and inherits storage, the API, and the UI
unchanged.

---

## Verified on this machine

Python 3.13.14, Windows, RTX 3050 4GB (unused), no ffmpeg, no API keys.

- `faster-whisper base` int8 CPU — **10.2× realtime** (96s of audio in 9.4s). Bundles PyAV,
  so it decodes the mp4 audio track directly and **system ffmpeg is not required**.
- Byte-range streaming verified for all five cases: full request, explicit range,
  open-ended seek, suffix range, and 416 past EOF.
- Jump-to-moment confirmed in a real headless browser: clicking a unit seeks the `<video>`
  element to the unit's start.
- Re-processing hits the extraction cache: **269× realtime**.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layers, the adapter boundary, design decisions
- [docs/PIPELINE.md](docs/PIPELINE.md) — every stage with the actual maths
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — running, demoing, tuning, extending, troubleshooting

## Optional model providers

Everything above runs with no key. One env var upgrades it:

```bash
export VKE_VISION_PROVIDER=openai   # real visual descriptions + on-screen text
export VKE_LLM_PROVIDER=openai      # abstractive titles, cited answers
export VKE_API_KEY=sk-...
export VKE_BASE_URL=https://api.openai.com/v1
#      Gemini: https://generativelanguage.googleapis.com/v1beta/openai/
#      Ollama: http://localhost:11434/v1
```

One adapter covers every vendor, because they all speak the OpenAI chat-completions
schema. A missing key or a failed call falls back to the extractive path and is logged —
enrichment can never make the output worse.

## Known limitations

- **Offline, `visual_context` is measured, not described**, and every observation records
  `source: "heuristic"` so it can never be mistaken for a model's output. On-screen text
  needs a vision provider.
- **Diarization is heuristic.** It detects speaker *changes*, not identities, and reports
  one speaker with low confidence when it cannot hear a real difference.
- **The fixture is short (105s).** The block-size rule is principled rather than fitted,
  but the numbers above should be reproduced on a real 8–12 minute recording before being
  quoted.
- ImageBind-style joint audio-visual embeddings are not reproducible here (no torch).

## Next

A real 8–12 minute video with slides and a live demo section, hand-labelled once, is the
single highest-value remaining step — it converts the fixture result into a claim about
real footage.
