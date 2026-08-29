# Pipeline

What happens to a video, stage by stage, with the actual maths. For module
layout see [ARCHITECTURE.md](ARCHITECTURE.md); for commands see
[WORKFLOW.md](WORKFLOW.md).

---

## Overview

```
                                    mp4
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
   faster-whisper              OpenCV @2fps               PySceneDetect
   (PyAV decodes the           one decode pass            ContentDetector
    audio in place)                  │                            │
        │                            │                            │
   Utterance[]                 FrameFeature[]                SceneCut[]
   absolute ts                 hist·edges·motion            cut times
        │                            │                            │
        ├──── diarize.py ────────────┤                            │
        │     SpeakerTurn[]          │                            │
        ▼                            ▼                            ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                      UNIFIED TIMELINE                             │
   │        one absolute axis, grid step 0.5s, every signal aligned    │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
      s(t) = w_sem·semantic + w_vis·visual + w_sil·silence + w_spk·speaker
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  τ = μ(s) + k·σ(s)   →   local maxima   →   NMS (15s)             │
   │  →  snap to utterance edge (±2s)  →  enforce max (180s)           │
   │  →  refine: merge noise splits, split multi-topic runs            │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
                             KnowledgeUnit[]
                    + BoundaryExplanation on every one
                                   ▼
   ┌──────────┬───────────┬────────────┬───────────┬──────────────────┐
   │ keyframes│  Pass 2   │   enrich   │   graph   │  export / search │
   │  (jpg)   │ VLM · OCR │ context ·  │ 5 types · │  JSON · JSONL ·  │
   │          │ (optional)│  quality   │ 6 edges   │  Universal       │
   └──────────┴───────────┴────────────┴───────────┴──────────────────┘
```

---

## Stage 1 — Probe

`media.probe` opens the file once for duration, fps, resolution, and asks PyAV
whether an audio stream exists. A corrupt file fails here, before any expensive
work begins.

## Stage 2 — Speech → absolute timestamps

`asr.transcribe` runs `faster-whisper base` (int8, CPU) **directly on the mp4**.
faster-whisper bundles PyAV, so it decodes the audio track itself and system
ffmpeg is not required.

```python
Utterance(id="u0007", span=Span(start=25.44, end=29.10),
          text="The session token expires after a while…",
          confidence=0.91, speaker="speaker_00",
          words=[Word(text="The", start=25.44, end=25.58), …])
```

Every timestamp is absolute from t=0. This is the baseline defect being fixed:
VideoRAG emits segment-relative times and never converts them.

Fallback chain: faster-whisper → `.srt`/`.vtt` sidecar → nothing (a silent video
is valid, not an error). Only the `Utterance[]` source changes; nothing
downstream notices.

## Stage 3 — One visual decode pass

`media.extract_frames` walks the video once, sampling ~2 fps:

| Measurement | How | Used for |
|---|---|---|
| `hsv_hist` | 8×8×4 bins, normalized | the visual signal |
| `edge_density` | Canny, fraction of lit pixels | text/UI density proxy |
| `motion` | mean abs diff vs previous sample | activity |
| `brightness` | mean grey | scene description |

## Stage 4 — Scene cuts

`PySceneDetect` `ContentDetector`. Cuts are folded into the visual signal (see
below), never added as a separate term.

## Stage 5 — Diarization

`diarize.py` decodes audio with PyAV, takes three descriptors per utterance
(spectral centroid, bandwidth, RMS energy), normalizes, and runs k-means over
k ∈ [2, 4] choosing k by **silhouette**.

Silhouette matters: a between/within ratio rises monotonically with k, so it
always picks the largest k and reports four speakers for one voice. Below a
silhouette of 0.45 the module returns a single speaker and a confidence of 0.25 —
it says "I cannot hear a difference" rather than inventing turns.

---

## Stage 5b — Observe (semantic enrichment)

`detect.py` runs the only real vision inference in the tree. It happens **before**
chunking and is deliberately independent of it.

```
FrameFeature[] + SceneCut[]  ->  select_observation_frames()  ->  <=40 timestamps
                                          |
                             grab_frames() at FULL resolution
                                          |
                    +---------------------+---------------------+
                    |                                           |
             YOLOv10n (ONNX)                        PP-OCR (ONNX, optional)
             every selected frame              only the most edge-dense frames
                    |                                           |
                    +----------------> VisualObservation <------+
                       kind - value - source - ts - model - confidence - box
```

**Frames are chosen for the whole video, not per unit.** One detection pass then
serves all three configs, which costs a third of the alternative and keeps the
headline comparison an honest ablation: every config sees identical visual
evidence, so only the boundaries differ.

Selection is not uniform sampling — that would spend forty frames on one slide.
It seeds a frame just after each scene cut, then greedily takes the frame least
similar to those already held (the same Bhattacharyya distance the visual signal
uses), breaking ties toward the largest **time gap** so a long static shot cannot
swallow the budget.

OCR is placed by measurement: it runs on the most edge-dense frames of *this*
video, above a floor that only rejects the near-featureless. An absolute
threshold does not survive real footage — a clean UI with a few lines of large
text measures ~0.007 while a photograph of a street measures ~0.08, so a fixed
cut would skip the screenshot and OCR the photo.

Nothing here reaches Stage 6. See "Signals deliberately absent".

## Stage 6 — The four signals

All evaluated on a shared 0.5s grid, each normalized to [0, 1].

### semantic — TextTiling depth score

```
tokens  = content words with timestamps (stopwords removed)
block   = round(token_rate × 25s)          ← seconds of speech, not a token count
sim(i)  = cosine( TF(tokens[i-block:i]), TF(tokens[i:i+block]) )
depth(i)= (left_peak − sim(i)) + (right_peak − sim(i))   at local minima only
```

Three details are load-bearing, and each was a bug before it was a feature:

1. **Blocks are sized in seconds of speech.** A fixed token count is not
   comparable across speaking paces; a block must span roughly the length of
   topic you want to resolve.
2. **Depth score, not raw dissimilarity.** In short speech `1 − cosine` sits near
   1.0 almost everywhere and discriminates nothing. Depth cancels that baseline.
3. **Depth at local minima only.** Scoring every position turns a long flat
   valley — common when vocabulary is fully disjoint — into a wide plateau that
   then dominates normalization.

The comparison point is placed at the **midpoint between the two blocks**, not at
the first token after the gap. Blocks are counted in tokens, so a long pause
makes consecutive token times jump; anchoring on the later token would skip
straight over the silence where the boundary actually is.

### visual — windowed histogram change, absorbing cuts

```
hist(t)   = Bhattacharyya( mean hist over [t−4s, t), mean hist over [t, t+4s) )
cut(t)    = max over cuts of  exp( −(t − t_cut)² / 2σ² ),  σ = 1.5s
visual(t) = max( normalize(hist)(t), cut(t) )
```

`max`, not `+`. A scene cut and the histogram jump around it are the *same
physical event*; adding them would silently give the visual modality about twice
its intended weight.

### silence — pauses between utterances

```
for each gap:  amplitude = min(gap / 2.0, 1.0)
               placed as a gaussian at the gap midpoint
silence(t)   = max over gaps
```

Free (already in the ASR output), independent, and in practice the strongest
single cue in talks.

### speaker — handovers

A narrow gaussian (σ = 1s) at each detected speaker change, scaled by diarization
confidence. Contributes exactly zero on single-presenter footage, which is the
honest outcome for most demo material.

### Normalization

Robust min–max against the 5th/95th percentile — except when that range
degenerates. If a signal is non-zero at only a handful of grid points both
percentiles are 0, and naive percentile scaling would zero the entire array,
silently deleting the sharpest and most confident boundaries. In that case we
fall back to true min–max.

### Signals deliberately absent

| Rejected | Why |
|---|---|
| `action_change`, `entity_change` | **Circular.** They need model output that only runs near boundaries not yet found. |
| `topic_shift` | The same measurement as `semantic` at a different window. |
| `scene_change` as its own term | Double-counts `visual`. |

---

## Stage 7 — Boundary selection

1. **Fuse:** `s(t) = Σ wᵢ · signalᵢ(t)`
2. **Threshold:** `τ = μ(s) + k·σ(s)`, k = 1.0 — adapts per video rather than
   hard-coding a magic constant.
3. **Local maxima** above τ.
4. **Non-maximum suppression** within `MIN_EVENT_SECONDS` (15s), strongest first.
5. **Snap** each survivor to the nearest utterance start within ±2s, so a chunk
   never begins mid-sentence. The pre-snap position is recorded as
   `snapped_from`.
6. **Enforce max** (180s) by splitting at the strongest interior peak.
7. **Refine** (below).

Every accepted boundary carries a `BoundaryExplanation`: per-signal raw value,
normalized value, weight, and product — the arithmetic shown in the UI.

## Stage 8 — Refinement

Peak-picking is local, so it can over-segment a continuous explanation and
under-segment a long stretch that drifts.

- **MERGE** — drop a boundary between two short neighbours (< 22s) whose
  vocabulary overlap is ≥ 0.30. The split was noise, not a topic change.
- **SPLIT** — add a boundary inside a long unit (> 150s) whose halves share
  less than 0.12 of their vocabulary. A real change the local peak was too weak
  to expose.

Config A is deliberately **not** refined; improving the baseline would flatter
the comparison.

## Stage 9 — Units, keyframes, enrichment

`build_units` turns boundary intervals into `KnowledgeUnit`s with transcript,
measured visual context, scene ids, speakers, key terms and provenance.

`enrich` then adds what makes a unit readable alone: an extractive summary,
`prev_summary`/`next_summary` (summaries, never neighbour copy-paste),
`carried_entities` (terms introduced earlier that this unit leans on),
`related_unit_ids`, an interpretable quality score, and validator flags.

`attach_observations` folds every observation whose timestamp falls inside the
unit into it. `objects`, `ocr_text` and `actions` are **pure projections** of
`observations` and are never written any other way — so a label cannot appear
without the source, timestamp, model and confidence that produced it sitting
behind it. Duplicate sightings collapse to the most confident one.

`actions` is empty here and stays empty: action recognition is deferred, and
nothing derives an action from motion or edge density.

One keyframe JPEG is written per unit, 1.5s inside its span, all from a single
decode pass.

## Stage 10 — Pass 2 (optional)

Only when a provider is configured, and only on the displayed config's units:
one VLM call per keyframe returns description, on-screen text, objects and
actions together. Capped by `VKE_MAX_VISION_CALLS` (default 40).

Its answers **merge** into the evidence from Stage 5b rather than replacing it,
and arrive with `confidence=None` — a chat-completions response carries no
per-item score, and stamping 1.0 on it would make a guess indistinguishable from
a detector's measured 0.95. This is the only producer permitted to emit an
`action`, because it is the only semantic model in the pipeline.

## Stage 11 — Graph

Five node types (`Video`, `Scene`, `Event`, `Entity`, `Speaker`) and six edges
(`CONTAINS`, `PRECEDES`, `OCCURS_DURING`, `MENTIONS`, `SPOKEN_BY`, `RELATED_TO`).
`RELATED_TO` links non-adjacent events sharing ≥2 entities — the edge that
answers "which moments relate to authentication?" across a whole video.

The graph is used by retrieval, not just drawn.

## Stage 12 — Retrieval

```
query → TF-IDF over units          (lexical)
      + entity match               (+0.25 each)
      + on-screen-text match       (+0.20 each)
      → graph expansion, 1 hop     (+0.12 × decay)
      → temporal filter            (before/after/at)
      → EvidenceSet
```

Every hit carries `unit_id`, `span`, `score` and a human-readable `reason`, which
is what makes a result clickable and auditable.

`ask()` with an LLM configured writes an answer that must cite evidence numbers.
Offline it returns the top evidence and labels itself
`extractive_offline` — it does not fabricate a narrative.

---

## Timings

105s fixture, Python 3.13, CPU only:

| Stage | Time |
|---|---:|
| probe | 0.4s |
| asr | 16.6s |
| frames | 2.9s |
| scenes | 3.1s |
| diarize | 0.2s |
| observe (yolov10n + PP-OCR, CPU) | 2.3s |
| signals | 0.1s |
| chunking (×3 configs) | 0.01s |
| graph | 0.00s |
| keyframes | 1.2s |
| **total** | **~24s (4.3× realtime)** |

Re-running reuses the cached extraction: **~270× realtime**. Expensive model
calls in the offline path: **zero**.
