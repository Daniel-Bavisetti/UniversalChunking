"""Strategy implementations + knowledge-unit assembly.

`chunk()` is the pipeline entry: carve atomic floats, run the routed strategy
over the text stream, project graph edges onto units, attach receipts.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .graph import ContextGraph
from .ingest_document import IngestResult
from .models import (
    ChunkingDecision,
    ContentElement,
    Context,
    KnowledgeUnit,
    Modality,
    Profile,
    Provenance,
    RelationType,
    Relationship,
    Temporal,
    count_tokens,
)
from .router import (
    MAX_TOKENS,
    TARGET_TOKENS,
    build_profile,
    choose_cut,
    escalation_flags,
)

log = logging.getLogger(__name__)


def chunk(ingest: IngestResult, graph: ContextGraph) -> tuple[list[KnowledgeUnit], Profile]:
    elements = ingest.elements
    profile = build_profile(elements)
    if not ingest.title:
        # Docling found no TITLE item — the first heading is the honest fallback
        ingest.title = next((e.text for e in elements if e.kind == "heading"), None)

    units: list[KnowledgeUnit] = []
    el_to_unit: dict[str, str] = {}
    members_by_unit: dict[str, list[str]] = {}
    counter = 0

    def new_unit_id() -> str:
        nonlocal counter
        uid = f"ku_{counter:04d}"
        counter += 1
        return uid

    def base_provenance(first: ContentElement) -> Provenance:
        return Provenance(
            source_uri=ingest.source_uri, source_sha256=ingest.sha256,
            page=first.page, bbox=first.bbox,
        )

    # 1 — atomic floats first, regardless of route
    consumed: set[str] = set()
    for e in elements:
        if e.kind not in ("table", "figure"):
            continue
        cap_ids = graph.captions_of(e.id)
        captions = [graph.by_id[c] for c in cap_ids if c in graph.by_id]
        consumed.add(e.id)
        consumed.update(c.id for c in captions)
        for u in _atomic_units(e, captions, graph, new_unit_id, base_provenance,
                               ingest.title):
            units.append(u)
            members_by_unit[u.id] = [e.id] + [c.id for c in captions]
            # first unit wins: an element split across units belongs to its start
            el_to_unit.setdefault(e.id, u.id)
            for c in captions:
                el_to_unit.setdefault(c.id, u.id)

    # 2 — the routed strategy over the remaining stream
    stream = [e for e in elements if e.id not in consumed]
    if profile.route == "tabular":
        # tables ARE the content here, so undo the atomic carve-out above and
        # let the tabular path own them
        units.clear()
        el_to_unit.clear()
        members_by_unit.clear()
        counter = 0
        text_units = _tabular_units([e for e in elements if e.kind == "table"],
                                    new_unit_id, base_provenance, ingest)
    elif profile.route == "temporal":
        text_units = _temporal_units(stream, graph, new_unit_id, base_provenance, ingest.title)
    elif profile.route in ("structural", "hybrid"):
        text_units = _structural_units(stream, graph, new_unit_id, base_provenance,
                                       ingest.title, profile, strategy=profile.route)
    else:
        text_units = _packed_units(stream, graph, new_unit_id, base_provenance,
                                   ingest.title, profile, strategy="paragraph_fallback")
    for u, members in text_units:
        units.append(u)
        members_by_unit[u.id] = list(members)
        for eid in members:
            el_to_unit.setdefault(eid, u.id)

    # Atomic floats were emitted before the routed stream, so restore reading
    # order by each unit's earliest member element. Units that share an element
    # (a table split into row groups) tie on position and keep emission order.
    order = {e.id: i for i, e in enumerate(elements)}
    emitted = {u.id: i for i, u in enumerate(units)}
    first_pos = {
        u.id: min((order.get(eid, 1 << 30) for eid in members_by_unit.get(u.id, ())),
                  default=1 << 30)
        for u in units
    }
    units.sort(key=lambda u: (first_pos[u.id], emitted[u.id]))

    _project_relationships(units, el_to_unit, graph)
    log.info("chunked %s: %d units via %s", ingest.source_uri, len(units), profile.route)
    return units, profile


# ───────── atomic ─────────

def _atomic_units(e: ContentElement, captions: list[ContentElement],
                  graph: ContextGraph, new_unit_id, base_provenance, title):
    cap_text = " ".join(c.text for c in captions).strip()
    heading_path = graph.heading_path(e.id)
    grid: list[list[str]] = e.meta.get("grid", [])
    header: list[str] = e.meta.get("header_row", [])
    visual: dict = e.meta.get("visual") or {}
    leading, trailing = graph.surrounding_text(e.id)

    def make(content: str, reason: str, vetoed: list[str], part: str | None = None):
        # A figure that vision understood is no longer "uncaptioned" in the sense
        # the flag means — something now states what it shows.
        described = bool(cap_text) or bool(e.meta.get("visual", {}).get("description"))
        flags = escalation_flags(content, heading_path, "atomic",
                                 kind=e.kind, has_caption=described)
        return KnowledgeUnit(
            id=new_unit_id(),
            content=content,
            modality=Modality.DOCUMENT,
            context=Context(document_title=title, heading_path=heading_path,
                            leading=leading, trailing=trailing),
            provenance=base_provenance(e),
            decision=ChunkingDecision(
                strategy="atomic", reason=reason, vetoed_cuts=vetoed,
                escalation_flags=flags,
                signals={"caption_confidence":
                         1.0 if e.meta.get("caption_ids") else (0.8 if cap_text else 0.0)},
            ),
            entities=list(visual.get("visual_entities", []))[:8],
            metadata={
                "element_kind": e.kind,
                **({"part": part} if part else {}),
                # Provenance for every visual claim in the content: which models
                # ran, which did not and why. An empty field stays explainable.
                **({"visual": visual, "visual_type": visual.get("visual_type")}
                   if visual else {}),
                **({"visual_skipped": e.meta["visual_skipped"]}
                   if e.meta.get("visual_skipped") else {}),
            },
            token_count=count_tokens(content),
        )

    # A figure's text is its visual understanding (see figures.py / vision.py).
    # Before that existed this was always "" for figures, which is how a figure
    # unit ended up as a placeholder with nothing to retrieve.
    body = e.text
    full = "\n\n".join(x for x in (cap_text, body) if x)
    if not full:
        full = f"[uncaptioned {e.kind} on page {e.page}]"

    if count_tokens(full) <= MAX_TOKENS or not grid:
        if e.kind == "figure" and e.meta.get("visual"):
            vtype = (e.meta.get("visual_type") or "figure").replace("_", " ")
            reason = (f"{vtype} kept whole with its caption and what the picture "
                      "shows — severing the pair is vetoed"
                      if cap_text else
                      f"{vtype} kept as one unit — its content is what vision read "
                      "off the picture")
        else:
            reason = (f"{e.kind} kept whole with its caption — severing the pair is vetoed"
                      if cap_text else f"{e.kind} kept as one unit")
        yield make(full, reason, vetoed=[])
        return

    # oversized table: split by rows, header repeated on every continuation
    head_md = "| " + " | ".join(header) + " |\n|" + "---|" * max(1, len(header)) if header else ""
    rows = grid[1:] if header and grid and grid[0] == header else grid
    batch: list[str] = []
    part_no = 1

    def flush():
        nonlocal batch, part_no
        if not batch:
            return None
        rows_md = "\n".join("| " + " | ".join(r) + " |" for r in batch)
        content = "\n".join(x for x in (cap_text, head_md, rows_md) if x)
        u = make(
            content,
            reason=(f"table exceeds {MAX_TOKENS} tokens — split by rows; cutting through "
                    "the header is vetoed, so it repeats on every part"),
            vetoed=[f"cut inside table {e.id} without header rejected: header repeated instead"],
            part=f"{part_no}",
        )
        batch, part_no = [], part_no + 1
        return u

    budget = count_tokens(head_md) + count_tokens(cap_text)
    acc = budget
    for r in rows:
        row_md = "| " + " | ".join(r) + " |"
        t = count_tokens(row_md)
        if batch and acc + t > TARGET_TOKENS:
            u = flush()
            if u:
                yield u
            acc = budget
        batch.append(r)
        acc += t
    u = flush()
    if u:
        yield u


# ───────── structural ─────────

def _structural_units(stream, graph, new_unit_id, base_provenance, title, profile,
                      strategy: str = "structural"):
    """One unit per innermost section; oversized sections split at veto-checked
    paragraph boundaries, children inheriting the full heading path.

    strategy="hybrid" keeps the same regions and the same vetoes but lets
    embedding drift pick WHERE an oversized section splits (see _emit_region)."""
    sections: list[list[ContentElement]] = []
    cur: list[ContentElement] = []
    for e in stream:
        if e.kind == "heading":
            if cur:
                sections.append(cur)
            cur = [e]
        else:
            cur.append(e)
    if cur:
        sections.append(cur)

    out = []
    for sec in sections:
        out.extend(_emit_region(sec, graph, new_unit_id, base_provenance, title,
                                strategy=strategy, profile=profile))
    return out


# ───────── flat prose fallback ─────────

def _packed_units(stream, graph, new_unit_id, base_provenance, title, profile,
                  strategy: str):
    """Pack the whole stream to ~TARGET tokens, boundaries only between
    elements, same veto rules. When the embedding model is available, topic
    drift picks the groups first (strategy upgrades to 'semantic')."""
    try:
        from .semantic import semantic_groups  # noqa: PLC0415

        groups = semantic_groups(stream)
    except Exception:
        groups = None
    if groups and len(groups) > 1:
        out = []
        for g in groups:
            out.extend(_emit_region(g, graph, new_unit_id, base_provenance, title,
                                    strategy="semantic", profile=profile))
        return out
    return _emit_region(stream, graph, new_unit_id, base_provenance, title,
                        strategy=strategy, profile=profile)


def _emit_region(region, graph, new_unit_id, base_provenance, title,
                 strategy: str, profile: Profile):
    out = []
    queue = [r for r in (region,) if r]
    while queue:
        reg = queue.pop(0)
        tokens = sum(count_tokens(e.text) for e in reg)
        if tokens <= MAX_TOKENS:
            out.append(_text_unit(reg, graph, new_unit_id, base_provenance, title,
                                  strategy, reason=_whole_reason(reg, strategy, tokens),
                                  vetoed=[], overflow=False, profile=profile))
            continue
        sims = None
        if strategy == "hybrid":
            try:
                from .semantic import boundary_similarities  # noqa: PLC0415

                sims = boundary_similarities(reg)
            except Exception:
                sims = None
        cut = choose_cut(reg, graph, sims=sims)
        if cut.index is None:
            out.append(_text_unit(reg, graph, new_unit_id, base_provenance, title,
                                  strategy,
                                  reason=(f"{tokens} tokens with no safe boundary — kept whole "
                                          "(overflow beats severing a relationship)"),
                                  vetoed=cut.vetoes, overflow=True, profile=profile))
            continue
        left, right = reg[:cut.index], reg[cut.index:]
        if cut.similarity is not None:
            reason = (f"section of {tokens} tokens split where the topic drifts "
                      f"(adjacent similarity {cut.similarity:.2f}) inside the "
                      f"veto-safe window around {TARGET_TOKENS} tokens")
        else:
            reason = (f"section of {tokens} tokens split at the paragraph "
                      f"boundary nearest {TARGET_TOKENS} tokens")
        out.append(_text_unit(left, graph, new_unit_id, base_provenance, title,
                              strategy,
                              reason=reason,
                              vetoed=cut.vetoes, overflow=False, profile=profile))
        queue.insert(0, right)
    return out


def _whole_reason(reg, strategy: str, tokens: int) -> str:
    if strategy in ("structural", "hybrid"):
        head = next((e.text for e in reg if e.kind == "heading"), None)
        if head:
            return f"section {head[:60]!r} is {tokens} tokens — under budget, kept whole"
        return f"section is {tokens} tokens — under budget, kept whole"
    if strategy == "semantic":
        return (f"topic-coherent run of {len(reg)} paragraphs ({tokens} tokens) — "
                "boundary where embedding similarity dips below mean − 1σ")
    return f"{len(reg)} paragraphs packed to {tokens} tokens at paragraph boundaries"


def _text_unit(members, graph, new_unit_id, base_provenance, title,
               strategy: str, reason: str, vetoed: list[str], overflow: bool,
               profile: Profile):
    content = "\n\n".join(e.text for e in members if e.text)
    anchor = members[0]
    heading_path = graph.heading_path(anchor.id)
    if anchor.kind == "heading":
        # a section unit is situated by its ancestors PLUS its own heading
        heading_path = heading_path + [anchor.text]
    flags = escalation_flags(content, heading_path, strategy)
    unit = KnowledgeUnit(
        id=new_unit_id(),
        content=content,
        modality=Modality.DOCUMENT,
        context=Context(document_title=title, heading_path=heading_path),
        provenance=base_provenance(anchor),
        decision=ChunkingDecision(
            strategy=strategy, reason=reason, vetoed_cuts=vetoed,
            escalation_flags=flags,
            signals={
                "tokens": float(sum(count_tokens(e.text) for e in members)),
                "heading_density": profile.heading_density,
                "overflow": 1.0 if overflow else 0.0,
            },
        ),
        token_count=count_tokens(content),
    )
    return unit, [e.id for e in members]


# ───────── tabular (CSV / XLSX) ─────────

def _tabular_units(tables, new_unit_id, base_provenance, ingest):
    """One schema card per sheet, then header-repeating row groups.

    The schema card is the only unit here that can benefit from an LLM (it is
    the one place a human-readable "what is this dataset" line adds something
    structure cannot supply), so it is the only one flagged. Row groups are
    fully self-describing and stay tier 0 — which is why a 480-row file costs
    one LLM call rather than forty.
    """
    from .tabular import (  # noqa: PLC0415
        TABULAR_MAX_TOKENS,
        profile_table,
        render_group,
        row_groups,
    )

    source_name = Path(ingest.source_uri).name
    out = []
    for t in tables:
        grid = t.meta.get("grid") or []
        if not grid:
            continue
        header = t.meta.get("header_row") or grid[0]
        sheet = t.meta.get("sheet")
        prof = profile_table(grid, header, sheet)
        heading_path = [source_name] + ([sheet] if sheet else [])

        # ── schema card ──
        card_id = new_unit_id()
        card = KnowledgeUnit(
            id=card_id,
            content=prof.schema_card(source_name),
            modality=Modality.DOCUMENT,
            context=Context(document_title=ingest.title, heading_path=heading_path),
            provenance=base_provenance(t),
            decision=ChunkingDecision(
                strategy="tabular",
                reason=(f"schema card for {'sheet ' + repr(sheet) if sheet else 'the dataset'} — "
                        f"{prof.row_count:,} rows × {prof.column_count} columns profiled by type, "
                        "range and cardinality so the dataset is searchable by shape"),
                signals={"rows": float(prof.row_count), "columns": float(prof.column_count)},
                escalation_flags=["dataset summary — a one-line description of what these "
                                  "columns represent cannot be derived from structure alone"],
            ),
            metadata={"element_kind": "schema_card", **prof.to_meta()},
            token_count=count_tokens(prof.schema_card(source_name)),
        )
        out.append((card, [t.id]))

        # ── row groups ──
        groups = row_groups(grid, header)
        for gi, (start, rows) in enumerate(groups, start=1):
            content = render_group(header, rows)
            first_row = start + 2          # +1 for header, +1 for 1-based rows
            last_row = first_row + len(rows) - 1
            unit = KnowledgeUnit(
                id=new_unit_id(),
                content=content,
                modality=Modality.DOCUMENT,
                context=Context(document_title=ingest.title, heading_path=heading_path),
                provenance=base_provenance(t),
                decision=ChunkingDecision(
                    strategy="tabular",
                    reason=(f"rows {first_row:,}–{last_row:,} of "
                            f"{'sheet ' + repr(sheet) if sheet else source_name} "
                            f"({gi} of {len(groups)}) — cut between rows, never through one"),
                    signals={"rows_in_chunk": float(len(rows)),
                             "first_row": float(first_row), "last_row": float(last_row)},
                    vetoed_cuts=["a row group without its header is data with no schema — "
                                 "header repeated instead of split away"],
                ),
                relationships=[Relationship(RelationType.HAS_SCHEMA, card_id, 1.0,
                                            "column types and ranges for these rows")],
                metadata={"element_kind": "row_group", "sheet": sheet,
                          "first_row": first_row, "last_row": last_row,
                          "columns": header},
                token_count=count_tokens(content),
            )
            if unit.token_count > TABULAR_MAX_TOKENS:
                unit.decision.signals["overflow"] = 1.0
            out.append((unit, [t.id]))
            card.relationships.append(Relationship(
                RelationType.SCHEMA_OF, unit.id, 1.0,
                f"describes rows {first_row:,}–{last_row:,}"))
    return out


# ───────── temporal (stretch — exercised by audio/video elements) ─────────

def _temporal_units(stream, graph, new_unit_id, base_provenance, title):
    """Speaker change = boundary; turns <3s merge into the neighbour; runs
    >120s split at the largest inter-segment pause."""
    segs = [e for e in stream if e.t0 is not None]
    if not segs:
        return []
    turns: list[list[ContentElement]] = []
    for e in segs:
        if turns and turns[-1][-1].speaker == e.speaker:
            turns[-1].append(e)
        else:
            turns.append([e])
    # Absorb micro-turns — but never across a speaker change. A two-second
    # answer is still that person's answer, and folding it into the previous
    # speaker's turn would attribute their words to someone else, which is the
    # one thing temporal chunking exists to prevent. Only unattributed
    # fragments and same-speaker stutters merge.
    merged: list[list[ContentElement]] = []
    for t in turns:
        dur = (t[-1].t1 or 0) - (t[0].t0 or 0)
        same_voice = merged and (t[0].speaker is None
                                 or t[0].speaker == merged[-1][-1].speaker)
        if merged and dur < 3.0 and same_voice:
            merged[-1].extend(t)
        else:
            merged.append(t)

    out = []
    for t in merged:
        for span in _split_long_turn(t):
            content = " ".join(e.text for e in span)
            speaker = span[0].speaker
            unit = KnowledgeUnit(
                id=new_unit_id(),
                content=content,
                modality=Modality.AUDIO,
                context=Context(document_title=title),
                provenance=base_provenance(span[0]),
                decision=ChunkingDecision(
                    strategy="temporal",
                    reason=(f"speaker turn ({speaker or 'unknown'}), "
                            f"{(span[-1].t1 or 0) - (span[0].t0 or 0):.0f}s — "
                            "turn boundaries are chunk boundaries"),
                    escalation_flags=escalation_flags(content, [], "temporal"),
                ),
                temporal=Temporal(start_s=span[0].t0 or 0.0, end_s=span[-1].t1 or 0.0,
                                  speaker=speaker),
                metadata=_merge_span_meta(span),
                token_count=count_tokens(content),
            )
            out.append((unit, [e.id for e in span]))
    return out


def _merge_span_meta(span) -> dict:
    """Combine element metadata across a turn.

    Visual summaries are joined rather than overwritten: when a span covers
    several scenes, "what was on screen while this was said" is all of them,
    and keeping only the last would quietly drop the rest.
    """
    meta: dict = {}
    visuals: list[str] = []
    semantics: list[dict] = []
    for e in span:
        for k, v in e.meta.items():
            if k == "visual_summary":
                if v and v not in visuals:
                    visuals.append(v)
            elif k == "semantics":
                # One turn can ask a question AND assign a task — every
                # labelled utterance in the span survives, none overwrites.
                semantics.append(v)
            else:
                meta[k] = v
    if visuals:
        meta["visual_summary"] = " · ".join(visuals)
    if semantics:
        meta["semantics"] = semantics
    return meta


def _split_long_turn(turn, max_s: float = 120.0):
    dur = (turn[-1].t1 or 0) - (turn[0].t0 or 0)
    if dur <= max_s or len(turn) < 2:
        yield turn
        return
    gaps = [(float((b.t0 or 0) - (a.t1 or 0)), i)
            for i, (a, b) in enumerate(zip(turn, turn[1:]), start=1)]
    _, cut = max(gaps)
    yield from _split_long_turn(turn[:cut], max_s)
    yield from _split_long_turn(turn[cut:], max_s)


# ───────── relationship projection ─────────

def _project_relationships(units, el_to_unit, graph: ContextGraph) -> None:
    """Element edges → unit edges where endpoints land in different units,
    restricted to the kept types. Hierarchy stays denormalized as heading_path."""
    by_id = {u.id: u for u in units}
    for src_el, dst_el, d in graph.g.edges(data=True):
        if d["type"] != "references":
            continue
        su, du = el_to_unit.get(src_el), el_to_unit.get(dst_el)
        if not su or not du or su == du:
            continue
        unit = by_id[su]
        if any(r.target_id == du and r.type == RelationType.REFERENCES
               for r in unit.relationships):
            continue
        unit.relationships.append(Relationship(
            type=RelationType.REFERENCES, target_id=du,
            confidence=d["confidence"], evidence=d["evidence"],
        ))
    for a, b in zip(units, units[1:]):
        a.relationships.append(Relationship(RelationType.NEXT, b.id, 1.0, "reading order"))
        b.relationships.append(Relationship(RelationType.PREVIOUS, a.id, 1.0, "reading order"))
