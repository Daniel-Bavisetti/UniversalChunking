"""P2: make each unit readable on its own.

A chunk that only makes sense in sequence is not a reusable knowledge unit. This
adds the minimum context needed to read one cold, plus an interpretable quality
score so a consumer can tell good units from poor ones.

Context is SUMMARIZED, never copy-pasted from neighbours: duplicating adjacent
text inflates every chunk and re-introduces exactly the redundancy that
event-based chunking is supposed to remove.
"""

from __future__ import annotations

import math

from .schemas import KnowledgeUnit
from .text import tokenize

MIN_USEFUL_SECONDS = 8.0
IDEAL_SECONDS = 45.0


# --------------------------------------------------------------------------- #
# extractive summary
# --------------------------------------------------------------------------- #
def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.replace("?", ".").replace("!", ".").split("."):
        s = " ".join(raw.split())
        if len(s) > 12:
            out.append(s)
    return out


def summarize(text: str, max_chars: int = 180) -> str:
    """Highest term-frequency sentences, kept in their original order."""
    sentences = _sentences(text)
    if not sentences:
        return ""

    freq: dict[str, int] = {}
    for tok in tokenize(text):
        freq[tok] = freq.get(tok, 0) + 1

    def score(sentence: str) -> float:
        toks = tokenize(sentence)
        return sum(freq.get(t, 0) for t in toks) / len(toks) if toks else 0.0

    ranked = sorted(range(len(sentences)), key=lambda i: -score(sentences[i]))
    chosen: list[int] = []
    total = 0
    for i in ranked:
        if total + len(sentences[i]) > max_chars and chosen:
            break
        chosen.append(i)
        total += len(sentences[i])
    return ". ".join(sentences[i] for i in sorted(chosen)).strip(". ") + "."


# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #
def _lexical_cohesion(text: str) -> float:
    """Vocabulary overlap between the two halves of the unit.

    A coherent unit talks about one thing throughout, so its halves share
    vocabulary. Low overlap suggests the unit spans two unrelated events.
    """
    toks = tokenize(text)
    if len(toks) < 8:
        return 0.5  # too little evidence to judge; do not punish or reward
    mid = len(toks) // 2
    a, b = set(toks[:mid]), set(toks[mid:])
    if not a or not b:
        return 0.5
    return len(a & b) / len(a | b)


def _boundary_confidence(unit: KnowledgeUnit) -> float:
    b = unit.boundary
    if not b.signals:
        return 0.0  # a fixed clock cut has no evidence behind it, and says so
    if b.threshold <= 0:
        return 0.5
    return max(0.0, min(1.0, b.score / (b.threshold * 2.0)))


def _length_sanity(seconds: float) -> float:
    """Penalize units too short to be meaningful; taper very long ones."""
    if seconds < MIN_USEFUL_SECONDS:
        return seconds / MIN_USEFUL_SECONDS * 0.5
    return float(min(1.0, math.exp(-abs(seconds - IDEAL_SECONDS) / (IDEAL_SECONDS * 2))))


def score_quality(unit: KnowledgeUnit) -> dict[str, float]:
    parts = {
        "semantic_coherence": round(_lexical_cohesion(unit.transcript), 3),
        "boundary_confidence": round(_boundary_confidence(unit), 3),
        "length_sanity": round(_length_sanity(unit.span.duration), 3),
        "context_completeness": round(
            (0.5 if unit.transcript.strip() else 0.0)
            + (0.25 if unit.prev_summary or unit.prev_unit_id is None else 0.0)
            + (0.25 if unit.next_summary or unit.next_unit_id is None else 0.0),
            3,
        ),
    }
    parts["overall"] = round(sum(parts.values()) / len(parts), 3)
    return parts


# --------------------------------------------------------------------------- #
# the pass
# --------------------------------------------------------------------------- #
def enrich(units: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
    """Fill in summary, context, carried entities, relations and quality."""
    for unit in units:
        unit.summary = summarize(unit.transcript)

    by_id = {u.id: u for u in units}

    seen: set[str] = set()
    for unit in units:
        # Entities already introduced earlier are the ones a reader needs primed;
        # entities first appearing here are self-explanatory in context.
        unit.carried_entities = [e for e in unit.entities if e in seen]
        seen.update(unit.entities)

        prev = by_id.get(unit.prev_unit_id or "")
        nxt = by_id.get(unit.next_unit_id or "")
        unit.prev_summary = prev.summary if prev else None
        unit.next_summary = nxt.summary if nxt else None

    # Related units share substantive vocabulary but need not be adjacent. This
    # is what a graph would have given us, for one comprehension.
    for unit in units:
        mine = set(unit.entities)
        unit.related_unit_ids = [
            other.id
            for other in units
            if other.id != unit.id and len(mine & set(other.entities)) >= 2
        ][:5]

    for unit in units:
        parts = score_quality(unit)
        unit.quality = parts["overall"]
        unit.quality_parts = parts
    return units


# --------------------------------------------------------------------------- #
# optional LLM polish
# --------------------------------------------------------------------------- #
TITLE_SYSTEM = (
    "You title segments of a video. Reply with ONLY the title: 3 to 7 words, no "
    "quotes, no trailing punctuation. Describe what happens in the segment."
)


def polish_titles(units: list[KnowledgeUnit], llm) -> list[KnowledgeUnit]:
    """Replace extractive titles with abstractive ones when an LLM is available.

    Strictly an upgrade path: if the call fails or returns something implausible,
    the extractive title stays. This can never make the output worse.
    """
    for unit in units:
        if not unit.transcript.strip():
            continue
        prompt = "\n".join([
            f"Segment {unit.span.start:.0f}s-{unit.span.end:.0f}s.",
            f"Transcript: {unit.transcript[:1200]}",
            f"On screen: {unit.visual_context[:200]}",
        ])
        title = llm.complete(prompt, system=TITLE_SYSTEM, max_tokens=32).strip()
        title = title.strip('"').strip("'").rstrip(".").strip()
        if 3 <= len(title) <= 90 and "\n" not in title:
            unit.title = title
    return units


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(unit: KnowledgeUnit) -> list[str]:
    """Flag units a consumer should treat with suspicion."""
    flags: list[str] = []
    if unit.span.duration < MIN_USEFUL_SECONDS:
        flags.append("too_short")
    if not unit.transcript.strip():
        flags.append("no_speech")
    if unit.quality_parts.get("semantic_coherence", 1.0) < 0.08 and \
            len(tokenize(unit.transcript)) > 30:
        flags.append("possible_multiple_events")
    if unit.boundary.signals and unit.boundary.score < unit.boundary.threshold:
        flags.append("weak_boundary")
    return flags
