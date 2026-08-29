# Cleave — User Guide

How to run it, what you are looking at, and how to read what it tells you.
For how it is built inside, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## What Cleave is for

You have files — reports, papers, spreadsheets, recordings — and you want to feed them to
an AI system: a RAG app, a search index, an agent. Before any of that works, the files have
to be broken into pieces. That step is usually an afterthought, and it is where most of the
quality is lost: a table gets separated from its header, a caption from its figure, a
paragraph from the section that gave it meaning.

Cleave does that step properly. It reads the structure of a file first, works out what
belongs together, and cuts only where nothing important breaks. Each piece comes back
carrying its context and a short explanation of why it was cut where it was.

---

## Running it

### Requirements

Python 3.12 and [uv](https://docs.astral.sh/uv/). Everything else installs itself.

### Start

```bash
uv sync
uv run python -m cleave.app
```

Open **http://127.0.0.1:8321**.

The very first PDF takes about 15 seconds longer than usual while Docling downloads and
loads its layout models. After that it is a few seconds per file. *If you are demoing,
process one throwaway file first so nothing pays that cost in front of an audience.*

### Optional: audio

Audio transcription runs in a separate service, because the document stack and the audio
stack need conflicting versions of the same libraries. If you have the STT project:

```bash
cd ~/PycharmProjects/STT && .venv/bin/python -m uvicorn --factory stt.server.app:app_factory --port 8000
```

Cleave finds it automatically. Without it, audio uploads fail with a clear message and
everything else works normally.

### Optional: AI enrichment

A small minority of chunks — the ones whose meaning depends on context that isn't in them —
get an AI-written line of situating context. Cleave picks where that comes from
automatically, **preferring a local model** because it costs nothing and keeps your
documents on your machine:

1. **Local (Ollama)** — used if one is running with a model pulled.
2. **Gemini API** — used if a `GEMINI_API_KEY` is set.
3. **Neither** — everything still runs. You see which chunks *would* have benefited,
   flagged but not filled in.

Configuration lives in one place. Copy the template and fill in what you need:

```bash
cp .env.example .env
```

`cleave/__init__.py` loads that file before anything reads the environment, so every entry
point — the web app, `python -m cleave.evaluate`, the tests — sees the same settings. Real
environment variables win over the file, so `CLEAVE_LLM=none uv run …` still overrides it.

### Checking it actually works

The homepage opens with a **System status** panel, and the server logs the same line at
startup. It reports either:

```
LLM Enrichment: Active
```

or:

```
LLM Enrichment unavailable — deterministic mode
```

This is a **live probe**, not a config check: it sends a real schema-constrained request
through the same code path enrichment uses. That distinction matters — an API key can be
valid while the model it names has been retired, which returns a 404 only when you actually
call it. A presence check would report that as healthy; this reports it as down, with the
reason. Press **re-probe** to re-check without reloading, or query it directly:

```bash
curl -s localhost:8321/health?refresh=true | python3 -m json.tool
```

To run locally, once:

```bash
brew install ollama && ollama serve
```

```bash
ollama pull qwen3:4b
```

Cleave finds it on the next run. The home page's **Spend** section shows which provider is
active. To force a choice server-wide: `CLEAVE_LLM=none`, `CLEAVE_LLM=ollama`, or
`CLEAVE_LLM=gemini`.

The upload form also has a **"Summarize chunks with an LLM"** toggle, on by default. Turning
it off for a job skips enrichment entirely for that upload — no model call, no cost — while
routing, chunking and the veto logic run exactly the same. The result page notes when flagged
chunks stayed at tier 0 because the toggle was off, so it reads the same as "no provider
configured" rather than looking like a bug.

### Watching what it costs

Every model call is metered — by model, tokens in and out, and dollars. Each result page has
a **Model usage** table for that file; the home page keeps a lifetime ledger, also available
as JSON at `/api/usage`. Local models appear in the same table at zero, so you can compare
the two paths directly rather than guess.

Two things keep the bill small. The router only enriches chunks that need it (typically
around a fifth). And enrichment **batches** — the document is sent once per batch of six
chunks rather than once per chunk, which on the sample paper cut input tokens by 80% and
cost by 63% with no change to the output. `CLEAVE_ENRICH_BATCH` tunes the batch size.

---

## What you can put in

| Type | Formats |
|---|---|
| Documents | PDF, DOCX, PPTX, Markdown, HTML, TXT |
| Spreadsheets | CSV, XLSX (every sheet) |
| Audio | MP3, M4A, WAV, FLAC — needs the STT worker |
| Contract | JSON from another modality worker (see [CONTRACT.md](CONTRACT.md)) |

Up to 50 MB. Large PDFs work but are slower — table structure recognition runs on CPU on
Apple Silicon, so a 20-page paper is a comfortable size.

---

## Using it

### 1. The home page

Drop in a file and press **Cleave it**.

Above the upload box, if an evaluation has been run, sits the **Context Preservation
Scorecard** — the measured comparison between ordinary fixed-size chunking and Cleave, on
four things that either survive the cut or do not. Every number there comes from
`data/scorecard.json`; none of it is written by hand.

### 2. While it runs

A progress bar with the pipeline stages beneath it: **understand → extract → graph → route
& chunk → enrich → write**. Each lights up as it completes, so a slow step is visible rather
than mysterious. The page stops polling by itself when the job finishes.

### 3. The results

**The six numbers at the top** are the efficiency story:

| | |
|---|---|
| **units** | how many knowledge units the file produced |
| **no AI** | share handled by structure alone, with no model involved |
| **needed context** | units whose meaning depended on something outside them |
| **LLM calls** | how many actually ran |
| **cost** | measured, in dollars |
| **wall clock** | end to end |

A well-structured document lands around 70% "no AI"; a spreadsheet around 96%. That gap
*is* the argument: the more structure a file has, the less intelligence has to be bought.

**The routing decision** says which strategy was chosen and why, in a sentence, with the
signals that drove it. This changes per file — that is the point. You will see:

| Strategy | Chosen when | What it does |
|---|---|---|
| `structural` | the file has real headings | one unit per section, splitting at safe paragraph boundaries |
| `tabular` | the file is a dataset | a schema card per sheet, plus row groups that repeat the header |
| `temporal` | content has timestamps | one unit per speaker turn |
| `semantic` | flat prose, no headings | boundaries where the topic actually shifts |
| `paragraph_fallback` | flat prose, no embedding model | honest paragraph packing |
| `atomic` | any table or figure inside prose | kept whole with its caption |

**The search box** runs retrieval over the units. Worth trying a question whose answer
depends on context — the results show what each unit knew about itself. *Click the Search
button rather than pressing Enter.*

**The context graph** (click to open) draws every element of the file in reading order along
a line, with an arc for each relationship found. Amber arcs are captions bound to their
figures, blue arcs are cross-references like "see Table 3". Hover any arc for the evidence.
Every arc is something a boundary was not allowed to cut through.

**The filter row** narrows the units — by strategy, or to just those that needed context, or
just those where a cut was refused. Beside it are links to the raw `units.json`,
`graph.json` and `profile.json`, which is what a downstream system would actually consume.

### 4. Reading a unit

Each card shows, top to bottom:

- **Identity** — id, strategy, token count, page or timestamp, and whether it needed AI.
  `tier 0 · no AI` means structure was enough.
- **Heading path** — where this sits in the document: `3. Results › 3.2 Revenue`.
- **Situating context** (green, when present) — the one AI-written line, only on chunks that
  needed it.
- **Surrounding prose** (italic, faint) — the sentence before and after. Context, not
  content: it helps retrieval find the chunk without polluting what the chunk *is*.
- **The content** — prose is collapsible; row groups render as real scrollable tables with
  the header pinned; schema cards render as a formatted profile.
- **Relationships** — clickable, each with its evidence and confidence. `0.8 · bbox
  adjacency: caption 14pt below figure` tells you exactly how the link was found, so you can
  disagree with it.
- **Why this chunk exists** — the receipt. The strategy's reasoning, any **veto** (a cut the
  system refused, in amber), any **flag** (why it needed context, in red), and the cost.

---

## Things worth trying

**A research paper.** Watch `structural` routing, then open the graph: the arcs are the
paper's own cross-references. Filter to **vetoed cuts** to see boundaries that were refused
because they would have split a table from its header.

**A spreadsheet.** The first unit is a *schema card* — every column with its inferred type,
range, and distinct values, worked out from the data with no model. Then row groups, each
repeating the header. Note the cost: one LLM call for the whole file, because only the
schema card needed one.

**Prose with no headings.** Routes `semantic`, and boundaries land where the subject
actually changes rather than at a token count.

**An audio file.** Same output shape as a PDF, now with speaker turns and timestamps — one
contract across every modality.

---

## Measuring it yourself

```bash
uv run python -m cleave.evaluate tests/fixtures/executive_summary.pdf tests/fixtures/attention_paper.pdf
```

Writes `data/scorecard.json` and prints the comparison. Pass any documents you like. Four
metrics, all deterministic:

- **caption integrity** — did captions stay with their figures?
- **header integrity** — does every chunk of a table carry its header?
- **heading context** — does each paragraph's chunk know its section?
- **resolved references** — can "see Table 3" still find Table 3?

The homepage renders whatever this produced. If you change the demo files, re-run it — the
UI never shows a number that was not measured.

### Tests

```bash
CLEAVE_LLM=none uv run pytest -q
```

These assert the promises rather than the implementation: that a caption is never severed
from its figure, that a row group always carries its header, that no row is ever split, that
routing responds to the shape of the input, and that every unit can explain itself.

---

## When something goes wrong

| Symptom | Cause and fix |
|---|---|
| Audio job fails immediately | The STT worker is not running — start it on port 8000. |
| No `situating context` anywhere | No API key found, or `CLEAVE_LLM=none`. Chunks show as `needs context` instead; nothing else changes. |
| Search returns "embedding model unavailable" | `sentence-transformers` did not install. Everything except search still works. |
| First PDF is slow | One-time Docling model download. Subsequent runs are fast. |
| A large PDF takes a while | Table structure recognition is CPU-only on Apple Silicon. Fewer pages, or expect the wait. |
| Contract JSON rejected | The payload must declare `"contract": 1` and contain `elements` or `units`. |

Failures show the actual error on the job page rather than a spinner that never resolves.

---

## Where the output goes

Every job writes to `data/jobs/<job_id>/`:

| File | Contents |
|---|---|
| `units.json` | the knowledge units — what a downstream system consumes |
| `graph.json` | every element and relationship found |
| `profile.json` | the signals, the routing decision, and the run totals |

Each unit carries an `embed_text` field: exactly the text to embed, with its context already
in front. That is the intended integration point — embed `embed_text`, store the rest as
metadata, and every chunk your retriever returns arrives knowing where it came from.
