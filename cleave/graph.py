"""Context graph: element-level relationships as a logical in-memory graph.

The graph is a means, not the deliverable — it constrains chunk boundaries,
assembles heading context, and resolves references. Edges carry confidence and
evidence because those are what the demo (and a downstream consumer) can audit.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import networkx as nx

from .models import ContentElement

log = logging.getLogger(__name__)

_REF_RE = re.compile(r"\b(Table|Figure|Fig\.?)\s+(\d+)", re.IGNORECASE)

#: max vertical gap (PDF points) for treating a caption as adjacent to a float
_CAPTION_GAP_PT = 60.0


class ContextGraph:
    def __init__(self, elements: list[ContentElement]):
        self.elements = elements
        self.by_id = {e.id: e for e in elements}
        # MultiDiGraph, not DiGraph: two elements can stand in more than one
        # relation at once, and a caption is very often the element immediately
        # before its figure. On a DiGraph the later `next` edge overwrote the
        # `captions` edge on that same pair — silently unlinking exactly the
        # pair this pipeline exists to keep together.
        self.g = nx.MultiDiGraph()
        for e in elements:
            self.g.add_node(e.id, kind=e.kind)
        self._build()

    # ───────── construction ─────────

    def _build(self) -> None:
        self._hierarchy()
        self._captions()
        self._references()
        self._reading_order()

    def _add(self, src: str, dst: str, type_: str, confidence: float, evidence: str) -> None:
        # Parallel edges of *different* types are the point; a repeat of the
        # same type is not, so it is collapsed rather than accumulated.
        if self._edge(src, dst, type_) is not None:
            return
        self.g.add_edge(src, dst, type=type_, confidence=confidence, evidence=evidence)

    def _edge(self, src: str, dst: str, type_: str) -> dict[str, Any] | None:
        if not self.g.has_edge(src, dst):
            return None
        for d in self.g.get_edge_data(src, dst).values():
            if d["type"] == type_:
                return d
        return None

    def _hierarchy(self) -> None:
        for e in self.elements:
            if e.parent_id and e.parent_id in self.by_id:
                self._add(e.parent_id, e.id, "parent", 1.0, "heading ancestry")

    def _captions(self) -> None:
        captions = [e for e in self.elements if e.kind == "caption"]
        claimed: set[str] = set()
        # pass 1: Docling's own caption references — deterministic
        for e in self.elements:
            if e.kind not in ("table", "figure"):
                continue
            for cid in e.meta.get("caption_ids", []):
                if cid in self.by_id:
                    self._add(cid, e.id, "captions", 1.0, "Docling caption reference")
                    self._add(e.id, cid, "captioned_by", 1.0, "Docling caption reference")
                    claimed.add(cid)
        # pass 2: bbox adjacency for floats Docling left uncaptioned
        for e in self.elements:
            if e.kind not in ("table", "figure") or self.captions_of(e.id):
                continue
            best: tuple[float, ContentElement] | None = None
            for c in captions:
                if c.id in claimed or c.page != e.page or not (c.bbox and e.bbox):
                    continue
                gap = _vertical_gap(e.bbox, c.bbox)
                if gap <= _CAPTION_GAP_PT and _h_overlap(e.bbox, c.bbox) > 0:
                    if best is None or gap < best[0]:
                        best = (gap, c)
            if best:
                gap, c = best
                ev = f"bbox adjacency: caption {gap:.0f}pt from {e.kind} on page {e.page}"
                self._add(c.id, e.id, "captions", 0.8, ev)
                self._add(e.id, c.id, "captioned_by", 0.8, ev)
                claimed.add(c.id)
                e.meta.setdefault("caption_ids", []).append(c.id)

    def _references(self) -> None:
        """'Table 3' / 'Figure 2' mentions → edges to the actual float.
        Resolution: caption text containing the label wins; ordinal fallback."""
        tables = [e for e in self.elements if e.kind == "table"]
        figures = [e for e in self.elements if e.kind == "figure"]

        def resolve(word: str, num: int) -> ContentElement | None:
            pool = tables if word.lower().startswith("t") else figures
            stem = "Table" if pool is tables else r"Fig(?:ure)?\.?"
            label = re.compile(rf"\b{stem}\s+{num}\b", re.I)
            for f in pool:
                for cid in f.meta.get("caption_ids", []):
                    cap = self.by_id.get(cid)
                    if cap and label.search(cap.text):
                        return f
            return pool[num - 1] if 0 < num <= len(pool) else None

        for e in self.elements:
            if e.kind not in ("paragraph", "list_item", "caption"):
                continue
            for m in _REF_RE.finditer(e.text):
                target = resolve(m.group(1), int(m.group(2)))
                # Checked per relation type: a reading-order edge between the
                # same two elements must not suppress a genuine reference.
                if (target and target.id != e.id
                        and self._edge(e.id, target.id, "references") is None):
                    self._add(e.id, target.id, "references", 0.9,
                              f"text mentions {m.group(0)!r}")

    def _reading_order(self) -> None:
        for a, b in zip(self.elements, self.elements[1:]):
            self._add(a.id, b.id, "next", 1.0, "reading order")

    # ───────── queries ─────────

    def heading_path(self, el_id: str) -> list[str]:
        path: list[str] = []
        seen: set[str] = set()
        cur = self.by_id.get(el_id)
        while cur and cur.parent_id and cur.parent_id not in seen:
            seen.add(cur.parent_id)
            parent = self.by_id.get(cur.parent_id)
            if parent is None:
                break
            path.append(parent.text)
            cur = parent
        return list(reversed(path))

    def captions_of(self, float_id: str) -> list[str]:
        return [src for src, _, d in self.g.in_edges(float_id, data=True)
                if d["type"] == "captions"]

    def surrounding_text(self, el_id: str, max_chars: int = 320) -> tuple[str | None, str | None]:
        """Nearest prose before and after an element.

        A table lifted out of its page loses the sentence that introduced it —
        "revenue by region is shown below" is often the only thing that says
        what the table is for. These windows are context, never content: they
        ride along in `Context.leading`/`trailing` and are embedded, not shown
        as the chunk's own text.
        """
        try:
            idx = next(i for i, e in enumerate(self.elements) if e.id == el_id)
        except StopIteration:
            return None, None
        prose = ("paragraph", "list_item")

        def scan(rng) -> str | None:
            for i in rng:
                e = self.elements[i]
                if e.kind in prose and len(e.text) > 40:
                    return e.text[:max_chars]
                if e.kind == "heading":
                    break          # a heading is a hard context boundary
            return None

        return scan(range(idx - 1, -1, -1)), scan(range(idx + 1, len(self.elements)))

    def references_to(self, el_id: str) -> list[tuple[str, dict[str, Any]]]:
        return [(src, d) for src, _, d in self.g.in_edges(el_id, data=True)
                if d["type"] == "references"]

    def references_from(self, el_id: str) -> list[tuple[str, dict[str, Any]]]:
        return [(dst, d) for _, dst, d in self.g.out_edges(el_id, data=True)
                if d["type"] == "references"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"id": e.id, "kind": e.kind, "page": e.page,
                 "text": e.text[:120] + ("…" if len(e.text) > 120 else "")}
                for e in self.elements
            ],
            "edges": [
                {"source": s, "target": t, "type": d["type"],
                 "confidence": round(d["confidence"], 3), "evidence": d["evidence"]}
                for s, t, d in self.g.edges(data=True)
            ],
        }


def _vertical_gap(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> float:
    """Min vertical distance between two boxes, robust to bbox origin
    (Docling PDF boxes are bottom-left origin, so t > b)."""
    a_lo, a_hi = min(a[1], a[3]), max(a[1], a[3])
    b_lo, b_hi = min(b[1], b[3]), max(b[1], b[3])
    if a_hi < b_lo:
        return b_lo - a_hi
    if b_hi < a_lo:
        return a_lo - b_hi
    return 0.0


def _h_overlap(a: tuple[float, float, float, float],
               b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
