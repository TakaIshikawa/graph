"""CSV export for HTML meta tags embedded in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "name", "property", "http_equiv", "content_value"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_META_RE = re.compile(r"<meta\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""(?P<name>[A-Za-z_:][\w:.-]*)\s*=\s*(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^\s"'=<>`]+))""")


def export_unit_markdown_html_meta_tag_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["name"]), sort_key(row["property"]), sort_key(row["http_equiv"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _META_RE.finditer(line):
            attrs = _attrs(match.group("attrs"))
            rows.append({"unit_id": uid, "title": title, "source": source, "line_number": line_number, "name": attrs.get("name", ""), "property": attrs.get("property", ""), "http_equiv": attrs.get("http-equiv", ""), "content_value": attrs.get("content", "")})
    return rows


def _attrs(raw: str) -> dict[str, str]:
    return {match.group("name").casefold(): field_value(match.group("dq") or match.group("sq") or match.group("bare")) for match in _ATTR_RE.finditer(raw)}
