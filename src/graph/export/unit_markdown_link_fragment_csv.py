"""CSV export for Markdown links with URL fragments or internal anchors."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line", "link_text", "destination", "fragment", "is_internal"]
_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def export_unit_markdown_link_fragment_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["destination"]), sort_key(row["link_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        for match in _LINK_RE.finditer(line):
            destination = field_value(match.group(2))
            fragment = _fragment(destination)
            if fragment:
                rows.append(
                    {
                        "unit_id": uid,
                        "title": title,
                        "line": line_number,
                        "link_text": field_value(match.group(1)),
                        "destination": destination,
                        "fragment": fragment,
                        "is_internal": "true" if destination.startswith("#") else "false",
                    }
                )
    return rows


def _fragment(destination: str) -> str:
    if destination.startswith("#"):
        return destination[1:]
    parsed = urlparse(destination)
    return parsed.fragment
