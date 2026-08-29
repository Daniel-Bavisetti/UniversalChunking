# Workflow

How to run, demo, extend and debug VKE. For internals see
[ARCHITECTURE.md](ARCHITECTURE.md) and [PIPELINE.md](PIPELINE.md).

---

## 1. Setup

```bash
pip install -r requirements.txt
```

No API keys, no GPU, no system ffmpeg, no torch. Verified on Python 3.13.14.

Prove the risky dependencies before trusting anything else:

```bash
python scripts/make_fixture.py     # builds data/fixture.mp4 + ground truth
python scripts/spike.py            # faster-whisper · OpenCV · PySceneDetect
```

`spike.py` prints PASS/FAIL per dependency. If it fails, nothing downstream will
work, and the failure tells you which fallback you need.

> On first run faster-whisper downloads its `base` model (~140MB). One time only.

---

## 2. Process a video

```bash
python scripts/process.py data/fixture.mp4
python scripts/process.py talk.mp4 --id my_talk       # custom id
python scripts/process.py talk.mp4 --force            # ignore the cache
```

Output goes to `data/store/<video_id>/`:

```
meta.json         duration, fps, resolution, has_audio
extraction.json   utterances · frame features · scene cuts · speaker turns
units.json        Knowledge Units, keyed by config
curves.json       signal curves + scores + transcript  (drives the UI)
graph.json        nodes and edges
traces.json       per-stage timings
keyframes/        one JPEG per unit
<video>.mp4       the staged source, so the API can stream it
```

Extraction is the only expensive stage and it is cached. Re-running to try new
weights takes under a second.

---

## 3. Run the UI

```bash
python -m uvicorn vke.api:app --port 8000
# http://localhost:8000
```

Drag a video onto the page to upload, or use the picker for one already processed.

**One screen, four tabs on the right:**

| Tab | What it shows |
|---|---|
| **Unit** | Keyframe, boundary arithmetic, quality breakdown, context, transcript |
| **Search** | Query moments or ask a question; every hit is clickable |
| **Graph** | Events chained in time, entities ringed around them |
| **Diagnostics** | Stage timings, providers in use, model-call counts |

Clicking anything with a timestamp seeks the player.

---

## 4. The demo

Three configurations, one code path, **one weight different**.

1. Select **Fixed 30s**. *"This is how video RAG chunks today — identical
   30-second boxes. It's VideoRAG's actual algorithm, reimplemented."* Play across
   a boundary: it cuts mid-sentence.
2. Select **Audio-only**. *"Now everything we do except look at the picture:
   topic shift, pauses, speaker changes. Better — but it never saw the screen."*
3. Select **VKE Multimodal**. Boundaries **move**. *"The only difference from the
   last one is that the visual weight went from zero to 0.40. Here's the
   arithmetic for this boundary."* Point at **Why this boundary**.
4. Click an event → the player seeks, the transcript highlights, the unit appears
   with its keyframe and its "context to read this alone".
5. **Export → Universal Units (JSON)**. *"Not a chat app — a knowledge layer any
   agent can consume, in a format a PDF pipeline would also emit."*

The claim is checkable, not rhetorical:

```bash
python scripts/evaluate.py
```

```
config            units  mean len    F1@2s    F1@5s     err  quality
Fixed 30s             4     26.2s     0.00     0.33    7.5s    0.480
Audio-only            4     26.2s     0.33     0.33    3.9s    0.595
VKE Multimodal        4     26.2s     0.67     1.00    0.9s    0.648

   reference  kind                Fixed 30s     Audio-only   VKE Multimodal
       25.5s  visual_only             FOUND         missed            FOUND
       52.8s  semantic_only          missed          FOUND            FOUND
       79.3s  both                   missed         missed            FOUND
```

---

## 5. Turning models on

Everything above runs with no key. To upgrade, copy the template and fill it in:

```bash
cp .env.example .env
```

```dotenv
# .env
VKE_LLM_PROVIDER=openai        # abstractive titles, cited answers
VKE_VISION_PROVIDER=openai     # real visual descriptions + on-screen text
VKE_API_KEY=sk-...
VKE_BASE_URL=https://api.openai.com/v1
VKE_LLM_MODEL=gpt-4o-mini
VKE_VISION_MODEL=gpt-4o-mini
VKE_MAX_VISION_CALLS=40
```

```bash
python scripts/process.py talk.mp4 --force
```

`.env` is loaded automatically (via `python-dotenv`) and is gitignored, so a real
key never ends up in a commit. A real shell/deploy environment variable always
takes precedence over `.env`, so this is safe to use in CI too — set the vars
there and `.env` is simply not read. Exporting the same variables directly in
the shell works identically, without a file:

```bash
export VKE_LLM_PROVIDER=openai VKE_VISION_PROVIDER=openai VKE_API_KEY=sk-...
python scripts/process.py talk.mp4 --force
```

One adapter covers every vendor, because they all speak the OpenAI
chat-completions schema:

| Vendor | `VKE_BASE_URL` |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Ollama (local) | `http://localhost:11434/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

Cost is bounded by `VKE_MAX_VISION_CALLS` (default 40) and only the displayed
config is enriched. Watch the spend in the **Diagnostics** tab.

If a key is missing or a call fails, the pipeline logs it and keeps the
extractive output. Model enrichment can never make the result worse.

---

## 6. Tuning boundaries

All knobs are in `vke/config.py`.

```python
CONFIG_C.weights           # semantic .40 · visual .40 · silence .20 · speaker .15
THRESHOLD_K = 1.0          # higher  -> fewer boundaries
MIN_EVENT_SECONDS = 15.0   # NMS radius and hard floor
MAX_EVENT_SECONDS = 180.0  # force a split beyond this
SEMANTIC_BLOCK_SECONDS = 25.0   # topic granularity to resolve
SNAP_WINDOW = 2.0          # utterance-edge snapping distance
```

**Look at the curve before changing anything:**

```bash
python scripts/inspect_signals.py                # fixture
python scripts/inspect_signals.py path/to.mp4    # your video
```

It prints each signal as a sparkline, the fused score for each config, the peaks
selected, and — with ground truth present — a per-boundary verdict. Tuning
weights without looking at this is guesswork.

> **If you change B or C's weights, keep them identical except for `visual`.**
> That single-variable difference is the entire comparison claim, and
> `test_audio_only_and_vke_differ_in_exactly_one_weight` will fail if it breaks.

---

## 7. Using a real video

The fixture proves correctness. A real recording proves the idea. Pick one with:

- a slide or screen change *without* a change in what is being said,
- a topic change *without* a visual change,
- ideally two speakers.

8–12 minutes is the sweet spot (~1 min to transcribe on CPU).

```bash
python scripts/process.py talk.mp4 --id talk
# hand-label the real boundaries once, ~30 minutes:
#   {"boundaries": [72.0, 185.5, 340.0],
#    "kinds": {"72.0": "visual_only", "185.5": "semantic_only", "340.0": "both"}}
python scripts/evaluate.py --video talk --truth labels.json --jsonl eval.jsonl
```

**Pre-process your demo video before presenting.** The cache makes a re-run
instant; a live upload on stage costs a minute of ASR.

---

## 8. Tests

```bash
python -m pytest tests/ -q          # 57 tests, ~2s
python -m pytest tests/ -q -k graph
```

Two are load-bearing and should be read before changing signal code:

- `test_configs_produce_different_boundaries` — if A, B and C ever agree, the
  comparison is meaningless.
- `test_audio_only_and_vke_differ_in_exactly_one_weight` — guards the ablation.

Also worth knowing: `test_robust_normalize_preserves_sparse_spikes` and
`test_texttiling_depth_does_not_plateau` both encode real bugs that silently
destroyed the semantic signal.

---

## 9. Extending

### A new boundary signal

1. Add a `*_curve()` in `signals.py` returning a `SignalCurve` normalized to [0,1].
2. Register it in `compute_curves`.
3. Give it a weight in `config.py` — **the same weight in B and C**, unless the
   signal is visual.
4. Add a unit test that it peaks where it should and stays silent where it should.

`BoundaryExplanation` picks it up automatically; the UI renders it without change.

### A new modality (PDF, slides, audio)

1. Write `pdf_extract.py` → your own observations.
2. Write `pdf_signals.py` → curves named e.g. `heading`, `font_change`
   (`SignalContribution.name` is `str`, so any name works).
3. Emit `KnowledgeUnit`s.
4. Emit `Locator(kind="page_region", ref={"page": 3, "bbox": [...]})`.

Storage, search, the graph, export and the UI work unchanged. Do not import
`media`/`asr`/`signals` downstream of `chunker.py` — that rule is the whole
integration contract.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `spike.py` fails on ASR | ctranslate2 wheel problem. Provide a `.srt` next to the video; the sidecar path takes over. |
| Video will not seek | Byte-range is broken. `curl -I -H "Range: bytes=0-1023" .../stream` must return `206`. |
| Boundaries look wrong | Run `inspect_signals.py` and look at the curve *before* touching weights. |
| Too many boundaries | Raise `THRESHOLD_K` or `MIN_EVENT_SECONDS`. |
| Semantic signal is flat | Not enough speech, or `SEMANTIC_BLOCK_SECONDS` is larger than your topics. |
| More speakers than exist | Silhouette threshold in `diarize.py`; raising it collapses to one speaker. |
| Port 8080 refuses to bind | Reserved on Windows. Use another port. |
| Unicode errors in scripts | `PYTHONIOENCODING=utf-8` — the console defaults to cp1252. |
| Changes not taking effect | Extraction is cached. Re-run with `--force`. |
