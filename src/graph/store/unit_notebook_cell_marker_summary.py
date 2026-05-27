"""Summarize notebook-style cell markers in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, unit_id

_CELL_RE = re.compile(r"^\s*(#|//)\s*%%(?:\s*\[markdown\])?", re.IGNORECASE)
_REGION_RE = re.compile(r"^\s*<!--\s*#(region|endregion)\b.*-->\s*$", re.IGNORECASE)


def summarize_unit_notebook_cell_markers(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = units_with = markers = markdown_cells = unbalanced = 0
    marker_types: Counter[str] = Counter()
    prefixes: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    for index, unit in enumerate(units):
        total += 1
        uid = unit_id(unit) or str(index)
        found = False
        balance = 0
        for line_no, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            marker_type = prefix = ""
            cell = _CELL_RE.match(line)
            region = _REGION_RE.match(line)
            if cell:
                marker_type = "cell"
                prefix = cell.group(1)
                if "[markdown]" in line.casefold():
                    markdown_cells += 1
            elif region:
                marker_type = region.group(1).casefold()
                prefix = "html"
                balance += 1 if marker_type == "region" else -1
            else:
                continue
            found = True
            markers += 1
            marker_types[marker_type] += 1
            prefixes[prefix] += 1
            if len(examples) < sample_limit:
                examples.append({"unit_id": uid, "line": line_no, "marker_type": marker_type, "text": line.strip()})
        if found:
            units_with += 1
        if balance != 0:
            unbalanced += 1
    return {"total_units": total, "units_with_markers": units_with, "total_markers": markers, "marker_type_counts": dict(sorted(marker_types.items())), "language_prefix_counts": dict(sorted(prefixes.items())), "markdown_cell_count": markdown_cells, "unbalanced_region_units": unbalanced, "examples": examples}
