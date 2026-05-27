"""CSV export for citation-like markers in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "marker_type", "marker_text", "line_number", "occurrence_index"]
_PATTERNS = [
    ("numeric_bracket", re.compile(r"\[(?:\d{1,3}(?:[-,]\s*\d{1,3})*)\]")),
    ("doi", re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)),
    ("footnote_ref", re.compile(r"\[\^[^\]]+\]")),
    ("author_year", re.compile(r"\([A-Z][A-Za-z]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z]+)?(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}[a-z]?\)")),
]


def export_units_to_citation_marker_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        occurrences: list[tuple[int, int, str, str]] = []
        for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            for marker_type, pattern in _PATTERNS:
                occurrences.extend((line_number, match.start(), marker_type, match.group(0)) for match in pattern.finditer(line))
        for index, (line_number, _pos, marker_type, marker_text) in enumerate(sorted(occurrences), start=1):
            rows.append({"unit_id": unit_id(unit), "title": title, "marker_type": marker_type, "marker_text": marker_text, "line_number": line_number, "occurrence_index": index})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["occurrence_index"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}
