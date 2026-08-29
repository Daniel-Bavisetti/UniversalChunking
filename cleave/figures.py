"""Figures inside documents: turning a placeholder back into content.

Docling locates every picture in a PDF and hands back its bounding box, but not
what it shows. Until now that produced the worst unit in the system — literally
``[uncaptioned figure on page 7]`` — a chunk with provenance, a heading path, a
position in the graph, and nothing whatsoever for a retriever to match on. On a
paper whose whole argument is in its figures, those were the chunks that
mattered most and said least.

This module crops each picture out of the rendered page and runs the shared
visual stack over it, then writes the result back onto the element so that
everything downstream — cut vetoes, caption pairing, reference resolution,
embedding — treats a figure like any other content-bearing element.

Two things keep the cost sane:

  * **Only figures that could carry meaning.** Rules, logos, bullet glyphs and
    page furniture are filtered by size before any model runs.
  * **The document's own words are used as grounding.** The caption and the
    sentence that introduced the figure are passed to the vision model, so its
    description agrees with the document rather than floating free of it — and
    a figure that already has a rich caption asks a cheaper question.
"""

from __future__ import annotations

import io
import logging

from .models import ContentElement

log = logging.getLogger(__name__)

#: A page can carry a surprising number of decorative pictures. This bounds the
#: worst case per document; the ones skipped say so in the element meta.
MAX_FIGURES_PER_DOC = 24


def _to_bytes(pil_image) -> bytes | None:
    try:
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        log.warning("could not serialize figure image (%s)", exc)
        return None


def enrich_figures(elements: list[ContentElement], images_by_element: dict[str, object],
                   *, title: str | None = None, use_llm: bool = True,
                   ledger=None) -> dict:
    """Fill in every figure element that has a picture behind it.

    ``images_by_element`` maps element id → a PIL image already cropped by
    Docling. Returns a small report for the job record.

    The element's ``text`` is replaced with a readable rendering of what the
    picture contains. That is deliberate: ``text`` is what gets embedded,
    counted, and shown, and a figure whose text is a placeholder is a figure
    that cannot be retrieved.
    """
    from .vision import load_bgr, understand  # noqa: PLC0415

    report = {"figures": 0, "understood": 0, "skipped": 0, "llm_calls": 0,
              "cost_usd": 0.0, "reasons": {}}
    if not images_by_element:
        return report

    by_id = {e.id: e for e in elements}
    order = {e.id: i for i, e in enumerate(elements)}

    def caption_text(el: ContentElement) -> str:
        return " ".join(
            by_id[cid].text for cid in el.meta.get("caption_ids", []) if cid in by_id
        ).strip()

    def nearby_prose(el: ContentElement) -> str:
        """The closest sentence before the figure — usually the one that says
        what it is for ("Figure 3 compares the two decoders …")."""
        idx = order.get(el.id)
        if idx is None:
            return ""
        for i in range(idx - 1, max(-1, idx - 6), -1):
            prev = elements[i]
            if prev.kind in ("paragraph", "list_item") and len(prev.text) > 40:
                return prev.text[:300]
        return ""

    processed = 0
    for el in elements:
        if el.kind != "figure" or el.id not in images_by_element:
            continue
        report["figures"] += 1
        if processed >= MAX_FIGURES_PER_DOC:
            el.meta["visual_skipped"] = (
                f"beyond the {MAX_FIGURES_PER_DOC}-figure budget for one document")
            report["skipped"] += 1
            continue

        data = _to_bytes(images_by_element[el.id])
        if data is None:
            el.meta["visual_skipped"] = "figure image could not be read"
            report["skipped"] += 1
            continue

        bgr = load_bgr(data)
        from .vision import is_meaningful  # noqa: PLC0415

        if not is_meaningful(bgr):
            el.meta["visual_skipped"] = "too small to be content — treated as decoration"
            report["skipped"] += 1
            continue

        caption = caption_text(el)
        hint_bits = [b for b in (f"document: {title}" if title else "",
                                 nearby_prose(el)) if b]
        seen = understand(data, caption=caption, context_hint=" | ".join(hint_bits),
                          use_llm=use_llm, ledger=ledger)
        processed += 1
        report["llm_calls"] += seen.llm_calls
        report["cost_usd"] += seen.cost_usd
        for producer, why in seen.skipped.items():
            report["reasons"].setdefault(producer, why)

        el.meta["visual"] = seen.to_dict()
        el.meta["visual_type"] = seen.visual_type
        if seen.is_empty:
            continue

        # The caption stays the caption; this is the figure's own content.
        el.text = seen.as_text()
        report["understood"] += 1

    report["cost_usd"] = round(report["cost_usd"], 6)
    if report["figures"]:
        log.info("figures: %d found, %d understood, %d skipped (%d model call(s))",
                 report["figures"], report["understood"], report["skipped"],
                 report["llm_calls"])
    return report
