"""Markdown table rendering, in one place.

Three call sites had grown their own copy: two byte-identical ``_table_markdown``
functions, a ``_row_md``, and an inline header fragment. They also each made the
same guess about whether Docling repeated the header as ``grid[0]``, which is the
kind of duplication that drifts silently.

``header_md`` and ``table_markdown`` are deliberately *not* the same function.
A header fragment is emitted for a table that is being split across several
chunks, where the header repeats above each part; it is not a table on its own,
and its separator width is guarded differently. Folding them together would
change the text of every split table.
"""

from __future__ import annotations

from collections.abc import Sequence


def row_md(row: Sequence[str]) -> str:
    """One markdown table row."""
    return "| " + " | ".join(row) + " |"


def table_markdown(grid: list[list[str]]) -> str:
    """A full markdown table, separator included when there is more than one row."""
    if not grid:
        return ""
    lines = [row_md(row) for row in grid]
    if len(lines) > 1:
        lines.insert(1, "|" + "---|" * len(grid[0]))
    return "\n".join(lines)


def header_md(header: Sequence[str]) -> str:
    """A header row plus separator, to repeat above each part of a split table.

    ``max(1, ...)`` guards a header that came back empty — the separator still
    has to be a valid row, or the continuation renders as prose.
    """
    if not header:
        return ""
    return row_md(header) + "\n|" + "---|" * max(1, len(header))


def body_rows(grid: list[list[str]], header: Sequence[str]) -> list[list[str]]:
    """The data rows, dropping the header row when the grid repeats it.

    Docling sometimes includes the header as ``grid[0]`` and sometimes does not.
    """
    return grid[1:] if grid and header and grid[0] == list(header) else grid
