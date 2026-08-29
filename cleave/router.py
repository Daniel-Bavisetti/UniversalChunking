"""Adaptive routing: profile the content, choose a strategy, choose safe cuts.

Three routing checks in priority order (temporal → structural → flat fallback),
atomic elements carved out first regardless. Escalation is orthogonal: cheap
triggers mark which chunks *deserve* an LLM, whether or not one is ever called.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .graph import ContextGraph
from .models import ContentElement, Profile, count_tokens

log = logging.getLogger(__name__)

TARGET_TOKENS = 500
MAX_TOKENS = 700

_TEXT_KINDS = ("paragraph", "list_item", "code", "caption")

# Sentences that lean on context that may live outside the chunk.
_ANAPHORA_RES = [
    re.compile(r"\bas (shown|described|noted|mentioned|discussed) (above|earlier|previously|below)\b", re.I),
    re.compile(r"\bthis (table|figure|section|chart|diagram|approach|result)\b", re.I),
    re.compile(r"\bthe (former|latter)\b", re.I),
    re.compile(r"^(This|These|It|They)\b"),
]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def is_tabular(elements: list[ContentElement]) -> bool:
    """True when the input IS a dataset rather than prose that contains tables.

    The discriminator is the share of content carried by tables: a report with
    a couple of tables still reads as prose, a spreadsheet is nothing else.
    """
    tables = [e for e in elements if e.kind == "table"]
    if not tables or not any(e.meta.get("grid") for e in tables):
        return False
    prose = [e for e in elements if e.kind in ("paragraph", "list_item")]
    table_tokens = sum(count_tokens(e.text) for e in tables)
    prose_tokens = sum(count_tokens(e.text) for e in prose)
    return table_tokens > 0 and table_tokens >= 4 * prose_tokens


def build_profile(elements: list[ContentElement]) -> Profile:
    text_els = [e for e in elements if e.kind in _TEXT_KINDS]
    headings = [e for e in elements if e.kind == "heading"]
    p = Profile(
        element_count=len(elements),
        text_element_count=len(text_els),
        heading_count=len(headings),
        heading_density=len(headings) / max(1, len(text_els)),
        table_count=sum(1 for e in elements if e.kind == "table"),
        figure_count=sum(1 for e in elements if e.kind == "figure"),
        caption_count=sum(1 for e in elements if e.kind == "caption"),
        has_timestamps=any(e.t0 is not None for e in elements),
        total_tokens=sum(count_tokens(e.text) for e in elements),
    )
    p.is_tabular = is_tabular(elements)
    if p.is_tabular:
        grids = [e.meta.get("grid") or [] for e in elements if e.kind == "table"]
        p.row_count = sum(max(0, len(g) - 1) for g in grids)
        p.column_count = max((len(g[0]) for g in grids if g), default=0)
    p.route, p.route_reason = _route(p)
    return p


def _route(p: Profile) -> tuple[str, str]:
    if p.has_timestamps:
        return "temporal", "content carries timestamps — chunking follows speakers and time"
    if p.is_tabular:
        sheets = f"{p.table_count} sheet{'s' if p.table_count != 1 else ''}"
        return "tabular", (
            f"{sheets}, {p.row_count:,} rows × {p.column_count} columns and almost no prose — "
            "this is a dataset: chunk by row groups, repeat the header, profile the schema"
        )
    if p.heading_count >= 3 and p.heading_density >= 0.03:
        structural_reason = (
            f"{p.heading_count} headings over {p.text_element_count} text elements "
            f"(density {p.heading_density:.2f}) — document structure is trustworthy, "
            "sections define chunks"
        )
        try:
            from .semantic import available  # noqa: PLC0415

            if available():
                return "hybrid", structural_reason + (
                    "; oversized sections split where the topic drifts, "
                    "not just nearest the token target"
                )
        except Exception:
            pass
        return "structural", structural_reason
    base_reason = (
        f"only {p.heading_count} headings for {p.text_element_count} text elements — "
        "no usable hierarchy"
    )
    try:
        from .semantic import available  # noqa: PLC0415

        if available():
            return "semantic", base_reason + ", grouping by embedding topic drift"
    except Exception:
        pass
    return "paragraph_fallback", base_reason + ", packing at paragraph boundaries"


# ───────── cut selection with hard vetoes ─────────

#: How far (in tokens) a semantic cut may stray from the target before token
#: discipline wins again. Wide enough for drift to matter, narrow enough that
#: chunk sizes stay predictable.
SEMANTIC_SLACK_TOKENS = 200


@dataclass(slots=True)
class CutResult:
    index: int | None                 # boundary BEFORE elements[index]; None = keep whole
    vetoes: list[str] = field(default_factory=list)
    overflow: bool = False
    similarity: float | None = None   # set when embedding drift chose the cut


def choose_cut(region: list[ContentElement], graph: ContextGraph,
               target_tokens: int = TARGET_TOKENS,
               sims: list[float] | None = None) -> CutResult:
    """Pick the paragraph boundary closest to the token target that severs no
    hard relationship. Overflow beats severance: if every candidate is vetoed,
    the region stays whole and says so.

    With ``sims`` (adjacent-element similarity, sims[i-1] for a cut at i), the
    veto-safe candidates within SEMANTIC_SLACK_TOKENS of the target compete on
    meaning instead: the cut lands where similarity is lowest — a topic drift —
    rather than merely nearest the token count. Vetoes stay hard either way."""
    tokens = [count_tokens(e.text) for e in region]
    candidates = list(range(1, len(region)))
    if not candidates:
        return CutResult(index=None, overflow=True)

    def tokens_before(i: int) -> int:
        return sum(tokens[:i])

    vetoes: list[str] = []
    soft: list[int] = []
    clean: list[int] = []
    for i in sorted(candidates, key=lambda i: abs(tokens_before(i) - target_tokens)):
        before, after = region[i - 1], region[i]
        # HARD: never strand a heading at the end of a chunk
        if before.kind == "heading":
            vetoes.append(
                f"cut before {after.id} rejected: would strand heading {before.text[:60]!r}"
            )
            continue
        # HARD: never separate a caption from its float (safety net — floats are
        # normally carved out before text regions form)
        cap_pair = _caption_pair(before, after, graph)
        if cap_pair:
            vetoes.append(f"cut before {after.id} rejected: severs {cap_pair}")
            continue
        # SOFT: prefer not to cut inside a consecutive list run
        if before.kind == "list_item" and after.kind == "list_item":
            soft.append(i)
            continue
        clean.append(i)
        if sims is None:
            return CutResult(index=i, vetoes=vetoes)

    if clean:
        nearest = clean[0]  # clean preserves nearest-to-target order
        window = [i for i in clean
                  if abs(tokens_before(i) - target_tokens) <= SEMANTIC_SLACK_TOKENS
                  and 0 <= i - 1 < len(sims)]
        if window:
            drift = min(window, key=lambda i: (sims[i - 1],
                                               abs(tokens_before(i) - target_tokens)))
            return CutResult(index=drift, vetoes=vetoes, similarity=sims[drift - 1])
        return CutResult(index=nearest, vetoes=vetoes)

    if soft:
        i = min(soft, key=lambda i: abs(tokens_before(i) - target_tokens))
        vetoes.append(f"no clean boundary — cut inside list run before {region[i].id} (least-bad)")
        return CutResult(index=i, vetoes=vetoes)
    return CutResult(index=None, vetoes=vetoes, overflow=True)


def _caption_pair(a: ContentElement, b: ContentElement, graph: ContextGraph) -> str | None:
    for x, y in ((a, b), (b, a)):
        if x.kind == "caption" and y.kind in ("table", "figure"):
            if x.id in graph.captions_of(y.id):
                return f"CAPTIONS {x.id} ↔ {y.id}"
    return None


# ───────── escalation (flags only — executing LLM calls is a stretch stage) ─────────

def anaphora_rate(text: str) -> float:
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return 0.0
    hits = sum(1 for s in sentences if any(rx.search(s) for rx in _ANAPHORA_RES))
    return hits / len(sentences)


def escalation_flags(content: str, heading_path: list[str], strategy: str,
                     kind: str = "text", has_caption: bool = True) -> list[str]:
    flags: list[str] = []
    rate = anaphora_rate(content)
    if rate > 0.10:
        flags.append(f"anaphora rate {rate:.2f} — meaning leans on context outside the chunk")
    if not heading_path and strategy not in ("temporal", "atomic"):
        flags.append("orphan — no heading ancestry to situate it")
    if kind in ("table", "figure") and not has_caption:
        flags.append(f"uncaptioned {kind} — nothing states what it shows")
    return flags
