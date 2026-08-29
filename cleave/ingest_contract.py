"""Contract import: knowledge from an external modality worker.

This is the seam described in CONTRACT.md. A separate repository (the video
pipeline) parses something Cleave cannot, and hands back either normalized
ContentElements — in which case Cleave still does the graph, routing and
chunking — or finished KnowledgeUnits, which are rendered as-is.

Keeping this in the core means the video work never has to touch Cleave, and
Cleave never has to grow a dependency on it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .ingest_document import IngestResult
from .models import (
    ChunkingDecision,
    ContentElement,
    Context,
    KnowledgeUnit,
    Modality,
    Provenance,
    RelationType,
    Relationship,
    Temporal,
    count_tokens,
    sha256_of,
)

log = logging.getLogger(__name__)

SUPPORTED_CONTRACT = 1


def load_contract(path: str | Path) -> tuple[IngestResult | None, list[KnowledgeUnit]]:
    """Read a contract file.

    Returns ``(ingest, [])`` for element payloads — the caller runs the normal
    pipeline over them — or ``(None, units)`` for finished units.
    """
    path = Path(path)
    payload = json.loads(path.read_text())

    # Only a file that DECLARES itself a contract is held to the contract.
    # Everything else is ordinary JSON — records, an API response, a config —
    # and gets chunked as data rather than rejected for not being something it
    # never claimed to be.
    if not isinstance(payload, dict) or "contract" not in payload:
        return _ingest_generic_json(path, payload), []

    version = payload.get("contract")
    if version != SUPPORTED_CONTRACT:
        raise ValueError(
            f"contract version {version!r} is not supported; this build speaks "
            f"contract {SUPPORTED_CONTRACT} — set \"contract\": 1, or remove the "
            "key entirely to have the file chunked as plain JSON data"
        )
    source_uri = payload.get("source_uri") or str(path)
    prefix = path.stem[:12]

    if payload.get("units"):
        units = [_unit_from(d, source_uri, prefix, i)
                 for i, d in enumerate(payload["units"])]
        log.info("contract import: %d finished units from %s", len(units), path.name)
        return None, units

    elements = [_element_from(d, prefix, i)
                for i, d in enumerate(payload.get("elements", []))]
    if not elements:
        raise ValueError("contract file contains neither 'units' nor 'elements'")

    from .cleaning import clean_elements  # noqa: PLC0415

    report = clean_elements(elements)
    log.info("contract import: %d elements from %s — %s",
             len(elements), path.name, report.summary())
    return IngestResult(
        elements=elements,
        title=payload.get("title") or path.stem,
        source_uri=source_uri,
        sha256=sha256_of(str(path)),
        warnings=[f"imported via contract v{version} from {path.name}"],
        cleaning=report.to_dict(),
    ), []


def _element_from(d: dict[str, Any], prefix: str, i: int) -> ContentElement:
    bbox = d.get("bbox")
    return ContentElement(
        id=f"{prefix}_{d.get('id', i)}",
        kind=d.get("kind", "other"),
        text=(d.get("text") or "").strip(),
        level=d.get("level"),
        parent_id=(f"{prefix}_{d['parent_id']}" if d.get("parent_id") else None),
        page=d.get("page"),
        bbox=tuple(bbox) if bbox else None,
        t0=d.get("t0"),
        t1=d.get("t1"),
        speaker=d.get("speaker"),
        meta=d.get("meta") or {},
    )


def _unit_from(d: dict[str, Any], source_uri: str, prefix: str, i: int) -> KnowledgeUnit:
    ctx = d.get("context") or {}
    prov = d.get("provenance") or {}
    dec = d.get("decision") or {}
    temp = d.get("temporal")
    content = d.get("content", "")
    return KnowledgeUnit(
        id=f"{prefix}_{d.get('id', i)}",
        content=content,
        modality=Modality(d.get("modality", "video")),
        context=Context(
            document_title=ctx.get("document_title"),
            heading_path=list(ctx.get("heading_path") or []),
            situating_summary=ctx.get("situating_summary"),
            leading=ctx.get("leading"),
            trailing=ctx.get("trailing"),
            tier=int(ctx.get("tier", 0)),
        ),
        provenance=Provenance(
            source_uri=prov.get("source_uri") or source_uri,
            source_sha256=prov.get("source_sha256"),
            page=prov.get("page"),
        ),
        decision=ChunkingDecision(
            strategy=dec.get("strategy", "imported"),
            reason=dec.get("reason", "produced by an external modality worker"),
            signals={k: float(v) for k, v in (dec.get("signals") or {}).items()},
            vetoed_cuts=list(dec.get("vetoed_cuts") or []),
            escalation_flags=list(dec.get("escalation_flags") or []),
            llm_calls=int(dec.get("llm_calls", 0)),
            cost_usd=float(dec.get("cost_usd", 0.0)),
        ),
        relationships=[
            Relationship(
                type=RelationType(r["type"]),
                target_id=f"{prefix}_{r['target_id']}",
                confidence=float(r.get("confidence", 1.0)),
                evidence=r.get("evidence"),
            )
            for r in (d.get("relationships") or [])
            if r.get("type") in {t.value for t in RelationType}
        ],
        temporal=(Temporal(start_s=float(temp["start_s"]), end_s=float(temp["end_s"]),
                           speaker=temp.get("speaker")) if temp else None),
        entities=list(d.get("entities") or []),
        metadata=d.get("metadata") or {},
        token_count=int(d.get("token_count") or count_tokens(content)),
    )


# ───────── generic JSON (no contract declared) ─────────

def _ingest_generic_json(path: Path, payload: Any) -> IngestResult:
    """Ordinary JSON, chunked as data.

    Two shapes are recognized, because they are what JSON in the wild almost
    always is:

    * an **array of records** (or a dict whose largest value is one) becomes a
      table grid — which downstream triggers the tabular route: a schema card
      profiling every column, then header-repeating row groups. A JSON export
      of 500 orders gets exactly the treatment the CSV of it would.
    * anything else is walked into ``key: value`` lines under headings taken
      from the top-level keys, so nesting survives as hierarchy instead of
      being flattened into one blob.
    """
    elements: list[ContentElement] = []
    counter = 0

    def new_id() -> str:
        nonlocal counter
        eid = f"el_{counter:04d}"
        counter += 1
        return eid

    records, records_key = _find_records(payload)
    if records is not None:
        grid = _records_to_grid(records)
        elements.append(ContentElement(
            id=new_id(), kind="table",
            text="\n".join("| " + " | ".join(r) + " |" for r in grid),
            meta={"grid": grid, "header_row": grid[0],
                  "sheet": records_key},
        ))
        # Scalar keys alongside the array (counts, cursors, metadata) are
        # context worth keeping, not noise.
        if isinstance(payload, dict):
            scalars = {k: v for k, v in payload.items()
                       if k != records_key and not isinstance(v, (dict, list))}
            if scalars:
                lines = "\n".join(f"{k}: {v}" for k, v in scalars.items())
                elements.append(ContentElement(
                    id=new_id(), kind="paragraph", text=lines))
    else:
        for line_group in _walk_json(payload):
            kind, level, text = line_group
            elements.append(ContentElement(
                id=new_id(), kind=kind, level=level, text=text))

    from .cleaning import clean_elements  # noqa: PLC0415

    report = clean_elements(elements)
    elements = [e for e in elements if e.text or e.kind == "table"]
    shape = (f"{len(records)} records" if records is not None
             else f"{len(elements)} elements")
    log.info("generic JSON %s: %s", path.name, shape)
    return IngestResult(
        elements=elements,
        title=path.stem,
        source_uri=str(path),
        sha256=sha256_of(str(path)),
        warnings=[],
        cleaning=report.to_dict(),
    )


def _find_records(payload: Any) -> tuple[list[dict] | None, str | None]:
    """The array of homogeneous flat-ish records, if this JSON is one.

    → (records, key) — key is None when the payload IS the array.
    """
    def is_records(v: Any) -> bool:
        return (isinstance(v, list) and len(v) >= 2
                and all(isinstance(r, dict) for r in v)
                and len({frozenset(r) for r in v}) <= max(3, len(v) // 4))

    if is_records(payload):
        return payload, None
    if isinstance(payload, dict):
        candidates = [(k, v) for k, v in payload.items() if is_records(v)]
        if candidates:
            key, records = max(candidates, key=lambda kv: len(kv[1]))
            return records, key
    return None, None


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)[:120]
    return str(v)


def _records_to_grid(records: list[dict]) -> list[list[str]]:
    header: list[str] = []
    for r in records:
        for k in r:
            if k not in header:
                header.append(k)
    return [header] + [[_cell(r.get(k)) for k in header] for r in records]


def _walk_json(payload: Any, key: str = "", depth: int = 0):
    """Depth-first walk → (kind, level, text) triples. Dict keys at the top two
    levels become headings, so the router can see the document's real shape."""
    if isinstance(payload, dict):
        if key and depth <= 2:
            yield ("heading", depth, key)
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                yield from _walk_json(v, k, depth + 1)
            else:
                yield ("paragraph", None, f"{k}: {_cell(v)}")
    elif isinstance(payload, list):
        if key and depth <= 2:
            yield ("heading", depth, key)
        for i, v in enumerate(payload):
            if isinstance(v, (dict, list)):
                yield from _walk_json(v, f"{key}[{i}]", depth + 1)
            else:
                yield ("list_item", None, _cell(v))
    else:
        yield ("paragraph", None, _cell(payload))
