"""Context Preservation Scorecard: fixed 512/64 baseline vs Cleave, measured.

Every number is computed from the documents given on the command line and
written to data/scorecard.json. Nothing is predefined: the UI renders whatever
this file measured, flattering or not.

Metrics (each preserved/total over the corpus):
  caption integrity   captioned tables whose caption and body share a chunk
  header integrity    tables whose every row-bearing chunk also carries the header
  heading context     paragraphs whose chunk knows their governing heading
  resolved references 'Table N'/'Figure N' mentions whose target is co-located
                      (fixed) or explicitly linked (Cleave)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .chunkers import chunk
from .graph import ContextGraph, _REF_RE
from .ingest_document import IngestResult, ingest_document
from .models import _encoder

FIXED_TOKENS = 512
FIXED_OVERLAP = 64

_norm_re = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _norm_re.sub(" ", s.lower()).strip()


def fixed_chunks(ingest: IngestResult) -> list[str]:
    """The baseline every RAG tutorial ships: flatten to text, split by tokens."""
    text = "\n\n".join(e.text for e in ingest.elements if e.text)
    enc = _encoder()
    ids = enc.encode(text, disallowed_special=())
    out, start = [], 0
    while start < len(ids):
        out.append(enc.decode(ids[start:start + FIXED_TOKENS]))
        start += FIXED_TOKENS - FIXED_OVERLAP
    return out


@dataclass(slots=True)
class Tally:
    preserved: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.total += 1
        self.preserved += 1 if ok else 0


@dataclass(slots=True)
class ArmScore:
    caption: Tally = field(default_factory=Tally)
    header: Tally = field(default_factory=Tally)
    heading: Tally = field(default_factory=Tally)
    refs: Tally = field(default_factory=Tally)

    def to_dict(self) -> dict:
        metrics = [
            ("caption integrity", self.caption),
            ("header integrity", self.header),
            ("heading context", self.heading),
            ("resolved references", self.refs),
        ]
        ratios = [t.preserved / t.total for _, t in metrics if t.total]
        return {
            "cps_pct": round(100 * sum(ratios) / len(ratios), 1) if ratios else None,
            "metrics": [{"name": n, "preserved": t.preserved, "total": t.total}
                        for n, t in metrics],
        }


def score_document(path: str, fixed: ArmScore, cleave: ArmScore) -> None:
    # use_llm=False keeps the scorecard what it claims to be: no model calls, no
    # judgement, the same probe strings searched in both arms. Letting figure
    # vision run here would make the measurement depend on a paid API and on
    # which day it ran.
    ingest = ingest_document(path, use_llm=False)
    graph = ContextGraph(ingest.elements)
    units, _profile = chunk(ingest, graph)

    f_chunks = [norm(c) for c in fixed_chunks(ingest)]
    c_units = [(norm(u.content), u) for u in units]
    by_unit_id = {u.id: u for u in units}

    def f_find(probe: str) -> list[int]:
        return [i for i, c in enumerate(f_chunks) if probe and probe in c]

    def c_find(probe: str):
        return [u for text_n, u in c_units if probe and probe in text_n]

    tables = [e for e in ingest.elements if e.kind == "table"]
    headings = {e.id: e for e in ingest.elements if e.kind == "heading"}

    # ── caption + header integrity (tables) ──
    for t in tables:
        grid: list[list[str]] = t.meta.get("grid", [])
        header_probe = norm(" ".join(t.meta.get("header_row", [])))[:60]
        row_probes = [norm(" ".join(r))[:60] for r in grid[1:]]
        row_probes = [p for p in row_probes if len(p) > 15]
        cap_ids = graph.captions_of(t.id)
        cap_text = " ".join(graph.by_id[c].text for c in cap_ids if c in graph.by_id)
        cap_probe = norm(cap_text)[:60]
        body_probe = row_probes[0] if row_probes else header_probe

        if cap_probe and body_probe:
            fixed.caption.add(any(cap_probe in f_chunks[i] for i in f_find(body_probe)))
            cleave.caption.add(any(body_probe in text_n and cap_probe in text_n
                                   for text_n, _u in c_units))

        if header_probe and row_probes:
            f_bearing = {i for p in row_probes for i in f_find(p)}
            fixed.header.add(bool(f_bearing) and
                             all(header_probe in f_chunks[i] for i in f_bearing))
            c_bearing = {u.id for p in row_probes for u in c_find(p)}
            cleave.header.add(bool(c_bearing) and
                              all(header_probe in norm(by_unit_id[uid].content)
                                  for uid in c_bearing))

    # ── heading context (paragraphs with a governing heading) ──
    for e in ingest.elements:
        if e.kind != "paragraph" or len(e.text) < 60 or not e.parent_id:
            continue
        head = headings.get(e.parent_id)
        if not head:
            continue
        probe = norm(e.text)[:80]
        head_probe = norm(head.text)[:60]
        hits = f_find(probe)
        fixed.heading.add(bool(hits) and any(head_probe in f_chunks[i] for i in hits))
        c_hits = c_find(probe)
        cleave.heading.add(bool(c_hits) and any(
            head_probe in norm(u.content)
            or any(head_probe in norm(h) for h in u.context.heading_path)
            for u in c_hits))

    # ── resolved references ──
    for e in ingest.elements:
        if e.kind not in ("paragraph", "list_item"):
            continue
        targets = graph.references_from(e.id)
        if not targets:
            continue
        probe = norm(e.text)[:80]
        for target_id, _d in targets:
            t_el = graph.by_id[target_id]
            t_grid = t_el.meta.get("grid", [])
            t_probe = norm(" ".join(t_grid[0]))[:60] if t_grid else ""
            if not t_probe:  # figures: caption text is the only textual anchor
                caps = graph.captions_of(target_id)
                t_probe = norm(" ".join(graph.by_id[c].text for c in caps))[:60]
            if not t_probe:
                continue
            hits = f_find(probe)
            fixed.refs.add(bool(hits) and any(t_probe in f_chunks[i] for i in hits))
            c_hits = c_find(probe)
            cleave.refs.add(bool(c_hits) and any(
                t_probe in norm(u.content)
                or any(r.type.value == "references" for r in u.relationships)
                for u in c_hits))


def main(paths: list[str]) -> dict:
    fixed, cleave = ArmScore(), ArmScore()
    for p in paths:
        print(f"scoring {p} …")
        score_document(p, fixed, cleave)
    result = {
        "documents": [Path(p).name for p in paths],
        "baseline": f"fixed {FIXED_TOKENS} tokens / {FIXED_OVERLAP} overlap",
        "fixed": fixed.to_dict(),
        "cleave": cleave.to_dict(),
    }
    out = Path(__file__).resolve().parent.parent / "data" / "scorecard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"\nwritten to {out}")
    return result


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "tests/fixtures/executive_summary.pdf",
        "tests/fixtures/attention_paper.pdf",
    ]
    main(args)
