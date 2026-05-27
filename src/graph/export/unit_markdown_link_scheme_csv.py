"""CSV export for Markdown links grouped by URL scheme."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "link_text", "scheme", "target", "scheme_type"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|<([A-Za-z][A-Za-z0-9+.-]*:[^>\s]+)>")


def export_units_to_markdown_link_scheme_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _LINK_RE.finditer(line):
            link_text = match.group(1) or ""
            target = match.group(2) or match.group(3) or ""
            scheme = _scheme(target)
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "link_text": field_value(link_text), "scheme": scheme, "target": field_value(target), "scheme_type": _scheme_type(scheme)})
    return rows


def _scheme(target: str) -> str:
    parsed = urlparse(target)
    return parsed.scheme.casefold() if parsed.scheme else "relative"


def _scheme_type(scheme: str) -> str:
    if scheme in {"http", "https"}:
        return "web"
    if scheme == "mailto":
        return "mail"
    if scheme == "file":
        return "file"
    if scheme in {"relative", ""}:
        return "unknown"
    return "app"

