"""CSV export for HTML anchor relation attributes in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "href", "rel_value", "rel_token", "line_number"]
_ANCHOR_RE = re.compile(r"<a\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)", re.DOTALL)


def export_unit_markdown_link_relation_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["href"]), sort_key(row["rel_token"])))
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
        for match in _ANCHOR_RE.finditer(line):
            attrs = {m.group("name").casefold(): field_value(m.group("value")) for m in _ATTR_RE.finditer(match.group("attrs"))}
            href = attrs.get("href", "")
            rel_value = attrs.get("rel", "")
            if not href or not rel_value:
                continue
            for token in rel_value.split():
                rows.append({"unit_id": uid, "title": title, "href": href, "rel_value": rel_value, "rel_token": token, "line_number": line_number})
    return rows
