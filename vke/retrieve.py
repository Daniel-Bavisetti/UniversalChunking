"""Search and timestamp-grounded question answering over Knowledge Units.

Retrieval exists to show the units are useful downstream; it is not the product.
So it is deliberately four short stages rather than a research pipeline:

    query -> lexical (TF-IDF) + entity match
          -> graph expansion (events sharing entities with the matches)
          -> optional temporal filter
          -> EvidenceSet, every hit carrying a span and a reason

Every result is timestamp-grounded, which is what makes a hit clickable. Offline,
`ask` returns evidence and an extractive answer and says so; it never fabricates
a narrative. With an LLM configured it writes a real answer that must cite the
unit ids it used.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .graph import EVENT, Graph
from .schemas import Evidence, KnowledgeUnit, Span
from .text import tokenize

TIME_PATTERNS = [
    (re.compile(r"\b(\d{1,2}):(\d{2})\b"), lambda m: int(m[1]) * 60 + int(m[2])),
    (re.compile(r"\b(\d+)\s*(?:min|minute)s?\b", re.I), lambda m: int(m[1]) * 60),
    (re.compile(r"\b(\d+)\s*(?:sec|second)s?\b", re.I), lambda m: int(m[1])),
]

BEFORE_WORDS = ("before", "prior to", "leading up to", "preceding")
AFTER_WORDS = ("after", "following", "next", "then")


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
class UnitIndex:
    """TF-IDF over units. Small enough that brute force is the right answer."""

    def __init__(self, units: list[KnowledgeUnit]) -> None:
        self.units = units
        self.docs: list[list[str]] = [tokenize(u.to_embedding_text()) for u in units]

        df: dict[str, int] = defaultdict(int)
        for doc in self.docs:
            for term in set(doc):
                df[term] += 1
        n = max(len(self.docs), 1)
        self.idf = {t: math.log(1 + n / (1 + c)) + 1.0 for t, c in df.items()}

        self.vectors: list[dict[str, float]] = []
        for doc in self.docs:
            tf: dict[str, float] = defaultdict(float)
            for term in doc:
                tf[term] += 1.0
            vec = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vectors.append({t: v / norm for t, v in vec.items()})

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        if not terms:
            return [0.0] * len(self.units)
        qtf: dict[str, float] = defaultdict(float)
        for t in terms:
            qtf[t] += 1.0
        qvec = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in qtf.items()}
        norm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        qvec = {t: v / norm for t, v in qvec.items()}

        out: list[float] = []
        for vec in self.vectors:
            shared = set(qvec) & set(vec)
            out.append(sum(qvec[t] * vec[t] for t in shared))
        return out


# --------------------------------------------------------------------------- #
# query understanding (rules, not a model)
# --------------------------------------------------------------------------- #
def parse_time_hint(query: str) -> tuple[float | None, str | None]:
    """Extract an absolute time and a before/after intent, if present."""
    seconds: float | None = None
    for pattern, convert in TIME_PATTERNS:
        m = pattern.search(query)
        if m:
            seconds = float(convert(m))
            break
    lowered = query.lower()
    direction = None
    if any(w in lowered for w in BEFORE_WORDS):
        direction = "before"
    elif any(w in lowered for w in AFTER_WORDS):
        direction = "after"
    return seconds, direction


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def search(
    query: str,
    units: list[KnowledgeUnit],
    graph: Graph | None = None,
    top_k: int = 6,
    index: UnitIndex | None = None,
) -> list[Evidence]:
    if not units:
        return []

    idx = index or UnitIndex(units)
    lexical = idx.score(query)
    terms = set(tokenize(query))

    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = defaultdict(list)

    for unit, base in zip(units, lexical):
        if base > 0.01:
            scores[unit.id] = base
            reasons[unit.id].append(f"text match ({base:.2f})")

        hit_entities = terms & {e.lower() for e in unit.entities}
        if hit_entities:
            scores[unit.id] = scores.get(unit.id, 0.0) + 0.25 * len(hit_entities)
            reasons[unit.id].append(f"entity: {', '.join(sorted(hit_entities))}")

        hit_ocr = terms & {w.lower() for line in unit.ocr_text for w in line.split()}
        if hit_ocr:
            scores[unit.id] = scores.get(unit.id, 0.0) + 0.2 * len(hit_ocr)
            reasons[unit.id].append(f"on-screen text: {', '.join(sorted(hit_ocr))}")

    # --- graph expansion: reach events the words alone would miss ----------- #
    if graph is not None and scores:
        seeds = [f"event:{uid}" for uid in scores]
        seeds += [f"entity:{t}" for t in terms if f"entity:{t}" in graph.nodes]
        for node_id, decay in graph.expand(seeds, hops=1, node_types=[EVENT]).items():
            uid = node_id.split(":", 1)[1]
            if uid not in scores:
                scores[uid] = 0.12 * decay
                reasons[uid].append("related via shared entities (graph)")

    # --- temporal filter ---------------------------------------------------- #
    at, direction = parse_time_hint(query)
    by_id = {u.id: u for u in units}
    if at is not None:
        anchor = next((u for u in units if u.span.contains(at)), None)
        if anchor is not None:
            for uid in list(scores):
                unit = by_id[uid]
                if direction == "before" and unit.span.end > anchor.span.start:
                    scores.pop(uid, None)
                elif direction == "after" and unit.span.start < anchor.span.end:
                    scores.pop(uid, None)
                elif direction is None and not unit.span.overlaps(anchor.span):
                    scores[uid] *= 0.5
            if direction:
                reasons[anchor.id].append(f"temporal anchor at {at:.0f}s")

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [
        Evidence(
            unit_id=uid,
            video_id=by_id[uid].video_id,
            span=by_id[uid].span,
            score=round(score, 4),
            reason="; ".join(reasons[uid]) or "ranked by similarity",
            snippet=_snippet(by_id[uid], terms),
        )
        for uid, score in ranked
    ]


def _snippet(unit: KnowledgeUnit, terms: set[str], width: int = 190) -> str:
    """The part of the transcript nearest a query term."""
    text = unit.transcript or unit.visual_context
    if not text:
        return unit.title
    lowered = text.lower()
    position = next(
        (lowered.find(t) for t in terms if lowered.find(t) >= 0), -1)
    if position < 0:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, position - width // 3)
    end = min(len(text), start + width)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


# --------------------------------------------------------------------------- #
# question answering
# --------------------------------------------------------------------------- #
ANSWER_SYSTEM = (
    "Answer the question using ONLY the numbered evidence provided. Every claim "
    "must cite the evidence number it came from, like [2]. If the evidence does "
    "not answer the question, say so plainly. Be concise."
)


def ask(
    question: str,
    units: list[KnowledgeUnit],
    graph: Graph | None = None,
    llm=None,
    top_k: int = 5,
) -> dict:
    """Return an answer plus the timestamp-grounded evidence behind it."""
    evidence = search(question, units, graph=graph, top_k=top_k)
    if not evidence:
        return {
            "question": question,
            "answer": "Nothing in this video matches that question.",
            "answer_source": "no_evidence",
            "evidence": [],
        }

    if llm is not None and getattr(llm, "name", "offline") != "offline":
        block = "\n\n".join(
            f"[{i+1}] {_fmt(e.span)} — {e.snippet}"
            for i, e in enumerate(evidence)
        )
        answer = llm.complete(
            f"Question: {question}\n\nEvidence:\n{block}",
            system=ANSWER_SYSTEM,
            max_tokens=350,
        )
        if answer:
            return {"question": question, "answer": answer,
                    "answer_source": "llm",
                    "evidence": [e.model_dump() for e in evidence]}

    # Offline: report what was found and where. Fabricating a narrative with no
    # model behind it would be worse than an honest pointer to the evidence.
    top = evidence[0]
    return {
        "question": question,
        "answer": (
            f"Best match {_fmt(top.span)} — {top.snippet} "
            f"({len(evidence)} relevant moment"
            f"{'s' if len(evidence) != 1 else ''} found.)"
        ),
        "answer_source": "extractive_offline",
        "evidence": [e.model_dump() for e in evidence],
    }


def _fmt(span: Span) -> str:
    def mmss(s: float) -> str:
        return f"{int(s // 60):02d}:{int(s % 60):02d}"
    return f"{mmss(span.start)}–{mmss(span.end)}"
