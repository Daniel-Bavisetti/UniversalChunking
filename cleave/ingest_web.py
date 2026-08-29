"""Web ingestion: a URL becomes structured elements.

The web is the one input where fetching and understanding are usually conflated,
and where most pipelines lose the most: a page is requested, the HTML is
stripped to text, and the navigation, cookie banner and footer arrive in the
index alongside the article. Everything that made the page navigable — the
heading hierarchy, the tables, the figures and their captions — is gone before
the first chunk is cut.

So the fetch layer here has exactly one job: **produce clean, structured
Markdown**. It does not chunk. Cleave's own profiler, graph, router and cut
vetoes are the differentiator, and handing boundary decisions to a crawler's
built-in splitter would throw that away.

Two fetchers, chosen by what the page actually is:

  * **Trafilatura** — the fast path. Article, blog post, docs page, news: static
    HTML where the problem is boilerplate removal, not rendering. No browser,
    no event loop, tens of milliseconds.
  * **Crawl4AI** — the capable path. JavaScript-rendered apps, single-page
    documentation, anything where the content does not exist until a browser
    runs. Costs a headless browser launch, so it is not the default.

The profiler tries the cheap one first and escalates only on evidence — a page
that yielded too little text, or that shipped an empty body with a script
bundle. That is the same principle the chunker uses for LLM calls: the
expensive tool is aimed, not sprayed, and the decision is recorded.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .ingest_document import IngestResult
from .models import ContentElement

log = logging.getLogger(__name__)

USER_AGENT = os.environ.get(
    "CLEAVE_WEB_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36 Cleave/0.1",
)
FETCH_TIMEOUT_S = float(os.environ.get("CLEAVE_WEB_TIMEOUT", "45"))

#: Below this many characters, a "successful" static extraction is treated as
#: evidence the page needs a browser rather than as a short article.
MIN_STATIC_CHARS = int(os.environ.get("CLEAVE_WEB_MIN_CHARS", "600"))

#: Hard ceiling on one page, so a pathological URL cannot dominate a job.
MAX_PAGE_CHARS = int(os.environ.get("CLEAVE_WEB_MAX_CHARS", "400000"))

_SCRIPT_HEAVY = re.compile(r"<(script|noscript)[\s>]", re.I)


def is_url(text: str) -> bool:
    try:
        parsed = urlparse(text.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


@dataclass(slots=True)
class FetchResult:
    markdown: str
    title: str | None
    fetcher: str                  # trafilatura | crawl4ai
    reason: str                   # why this fetcher, in one sentence
    warnings: list[str]


# ───────── fetchers ─────────

def _fetch_static(url: str) -> tuple[str, str | None, str]:
    """Trafilatura: main-content extraction with boilerplate removed.

    → (markdown, title, failure_reason)
    """
    try:
        import trafilatura  # noqa: PLC0415
    except Exception as exc:
        return "", None, f"trafilatura not installed ({type(exc).__name__})"

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "", None, "the page could not be downloaded"
        # include_* keep exactly the structure the chunker later relies on:
        # headings define sections, tables become grids, links carry references.
        #
        # favor_precision is deliberately NOT set. It looks like the safe
        # choice, and it is the wrong one here: on a Wikipedia article it
        # removes ~1,700 characters of boilerplate and, with them, all 14
        # headings — which drops the page from the structural route onto the
        # weaker semantic one. Trading a document's hierarchy for slightly less
        # boilerplate is the exact loss this project exists to prevent.
        markdown = trafilatura.extract(
            downloaded, output_format="markdown", include_tables=True,
            include_images=True, include_links=True, include_formatting=True,
            url=url,
        ) or ""
        title = None
        try:
            meta = trafilatura.extract_metadata(downloaded)
            title = getattr(meta, "title", None) if meta else None
        except Exception:
            pass
        if _SCRIPT_HEAVY.search(downloaded) and len(markdown) < MIN_STATIC_CHARS:
            return markdown, title, "page body is script-driven"
        return markdown, title, ""
    except Exception as exc:
        return "", None, f"{type(exc).__name__}: {exc}"


def _fetch_rendered(url: str) -> tuple[str, str | None, str]:
    """Crawl4AI: a real browser, for pages that only exist after JavaScript."""
    try:
        import asyncio  # noqa: PLC0415

        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig  # noqa: PLC0415
    except Exception as exc:
        return "", None, f"crawl4ai not installed ({type(exc).__name__})"

    async def run() -> tuple[str, str | None]:
        browser = BrowserConfig(headless=True, user_agent=USER_AGENT)
        async with AsyncWebCrawler(config=browser) as crawler:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(page_timeout=int(FETCH_TIMEOUT_S * 1000)),
            )
            markdown = ""
            raw = getattr(result, "markdown", None)
            if raw is not None:
                # Newer builds return an object carrying several variants; the
                # fit/filtered one is the de-boilerplated body.
                markdown = (getattr(raw, "fit_markdown", None)
                            or getattr(raw, "raw_markdown", None)
                            or str(raw))
            title = (getattr(result, "metadata", {}) or {}).get("title")
            return markdown or "", title

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return (*asyncio.run(run()), "")
        # Already inside an event loop (a FastAPI worker): use a private one on
        # its own thread rather than fighting the running loop.
        import concurrent.futures  # noqa: PLC0415

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            markdown, title = pool.submit(lambda: asyncio.run(run())).result(
                timeout=FETCH_TIMEOUT_S * 2)
        return markdown, title, ""
    except Exception as exc:
        return "", None, f"{type(exc).__name__}: {exc}"


def fetch(url: str, *, force: str | None = None) -> FetchResult:
    """Static first, browser on evidence. ``force`` pins one fetcher."""
    warnings: list[str] = []

    if force != "crawl4ai":
        markdown, title, why = _fetch_static(url)
        enough = len(markdown) >= MIN_STATIC_CHARS
        if markdown and enough and not why:
            return FetchResult(markdown, title, "trafilatura",
                               f"static HTML yielded {len(markdown):,} characters of "
                               "main content — no browser needed", warnings)
        if force == "trafilatura":
            return FetchResult(markdown, title, "trafilatura",
                               why or "static extraction, forced", warnings)
        warnings.append(
            f"static extraction returned {len(markdown):,} characters"
            + (f" ({why})" if why else "") + " — escalating to a rendered fetch")

    markdown, title, why = _fetch_rendered(url)
    if markdown:
        return FetchResult(markdown, title, "crawl4ai",
                           "page needed JavaScript rendering to produce its content",
                           warnings)

    warnings.append(f"rendered fetch failed: {why}")
    # Fall back to whatever static managed, rather than failing the job.
    markdown, title, static_why = _fetch_static(url)
    return FetchResult(markdown, title, "trafilatura",
                       f"rendered fetch unavailable ({why}); using static extraction",
                       warnings + ([static_why] if static_why else []))


# ───────── ingestion ─────────

def ingest_web(url: str, *, use_llm: bool = True, ledger=None,
               force: str | None = None) -> IngestResult:
    """URL → elements, via Markdown and the ordinary document parser.

    The fetched Markdown is handed to Docling rather than parsed here, so a web
    page produces exactly the same element kinds as a DOCX: headings with
    levels, tables with grids, list items, captions. Everything downstream then
    treats the page as a document, because structurally it now is one.
    """
    if not is_url(url):
        raise ValueError(f"not an http(s) URL: {url!r}")

    result = fetch(url, force=force)
    if not result.markdown.strip():
        raise RuntimeError(
            f"no content could be extracted from {url} — " + "; ".join(result.warnings))

    markdown = result.markdown[:MAX_PAGE_CHARS]
    truncated = len(result.markdown) > MAX_PAGE_CHARS

    import tempfile  # noqa: PLC0415

    from .ingest_document import ingest_document  # noqa: PLC0415

    host = urlparse(url).netloc.replace(":", "_")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", urlparse(url).path.strip("/"))[:48] or "index"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / f"{host}_{slug}.md"
        page.write_text(markdown, encoding="utf-8")
        ingest = ingest_document(page, use_llm=use_llm, ledger=ledger)

    # The temp file was a transport detail; provenance must point at the URL.
    ingest.source_uri = url
    ingest.title = result.title or ingest.title or host
    ingest.warnings = list(ingest.warnings) + result.warnings
    if truncated:
        ingest.warnings.append(
            f"page exceeded {MAX_PAGE_CHARS:,} characters and was truncated")

    _annotate(ingest.elements, url, result)
    log.info("web %s: %d elements via %s — %s",
             url, len(ingest.elements), result.fetcher, result.reason)
    return ingest


def _annotate(elements: list[ContentElement], url: str, result: FetchResult) -> None:
    """Stamp the fetch decision onto the stream so it reaches the receipt."""
    for el in elements:
        el.meta.setdefault("source_url", url)
    if elements:
        elements[0].meta["fetcher"] = result.fetcher
        elements[0].meta["fetch_reason"] = result.reason
