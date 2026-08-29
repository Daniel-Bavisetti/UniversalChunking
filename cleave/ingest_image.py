"""Image ingestion — Level 3, standalone visual content.

A photograph, a screenshot, a scanned page, an infographic or a diagram
arrives with no structure to extract: there are no headings to nest under and
no reading order to follow. Everything a retrieval system will ever know about
it has to be *produced*, which is the one place in this pipeline where a model
is not an optimisation but the only source of content.

That makes the routing honest rather than special-cased. The image becomes a
small set of elements — a figure carrying the visual understanding, and a
caption element carrying any text OCR read off it — and the ordinary graph,
router and chunker take it from there. A scanned document with a lot of text
routes differently from a photograph, because the elements genuinely differ.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .ingest_document import IngestResult
from .models import ContentElement, sha256_of

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

#: OCR lines beyond this count mean the picture is really a document that
#: happens to be stored as pixels. Its text becomes paragraphs so the normal
#: prose strategies apply, instead of one undifferentiated blob.
SCANNED_PAGE_LINES = 12


def ingest_image(path: str | Path, *, use_llm: bool = True, ledger=None) -> IngestResult:
    path = Path(path)
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"unsupported image type: {path.suffix}")

    from .vision import understand  # noqa: PLC0415

    seen = understand(path, context_hint=f"filename: {path.name}",
                      use_llm=use_llm, ledger=ledger)
    warnings: list[str] = []
    for producer, why in seen.skipped.items():
        warnings.append(f"{producer} did not run — {why}")

    elements: list[ContentElement] = []
    counter = 0

    def new_id() -> str:
        nonlocal counter
        eid = f"el_{counter:04d}"
        counter += 1
        return eid

    # A scanned page is a document, not a picture of one. Its OCR lines become
    # real paragraphs so headings, packing and semantic drift all still apply.
    scanned = len(seen.ocr_text) >= SCANNED_PAGE_LINES
    if scanned:
        log.info("%s: %d OCR lines — treating as a scanned page",
                 path.name, len(seen.ocr_text))
        for line in seen.ocr_text:
            if line.strip():
                elements.append(ContentElement(
                    id=new_id(), kind="paragraph", text=line.strip(), page=1,
                    meta={"source": "ocr"},
                ))

    caption_id: str | None = None
    if seen.description:
        caption_id = new_id()
        elements.append(ContentElement(
            id=caption_id, kind="caption", text=seen.description, page=1,
            meta={"source": "vision_model"},
        ))

    figure_id = new_id()
    figure = ContentElement(
        id=figure_id,
        kind="figure",
        # The figure's own text is the full rendering, so the unit is readable
        # standing alone rather than being an empty placeholder with a caption.
        text=seen.as_text(),
        page=1,
        meta={
            "visual": seen.to_dict(),
            "visual_type": seen.visual_type,
            "caption_ids": [caption_id] if caption_id else [],
            "is_scanned_page": scanned,
        },
    )
    elements.append(figure)

    if seen.is_empty:
        warnings.append(
            "no visual understanding was produced — the unit records the file and "
            "its provenance but has no content a retriever can match on")

    from .cleaning import clean_elements  # noqa: PLC0415

    report = clean_elements(elements)
    elements = [e for e in elements if e.text or e.kind == "figure"]

    title = seen.description[:80] if seen.description else path.stem
    log.info("image %s: type=%s ocr=%d objects=%d producers=%s",
             path.name, seen.visual_type, len(seen.ocr_text), len(seen.objects),
             ",".join(seen.producers) or "none")

    return IngestResult(
        elements=elements,
        title=title,
        source_uri=str(path),
        sha256=sha256_of(str(path)),
        warnings=warnings,
        cleaning=report.to_dict(),
    )
