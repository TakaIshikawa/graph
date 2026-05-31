"""CSV export for HTML iframe tags in unit content."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "src", "title_attr", "width", "height", "loading", "sandbox"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_IFRAME_RE = re.compile(r"<iframe\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""")


def export_units_to_html_iframe_inventory_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in _content_lines(str(get(unit, "content") or "")):
        for match in _IFRAME_RE.finditer(line):
            attrs = _attrs(match.group("attrs"))
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "line_number": line_number,
                    "src": attrs.get("src", ""),
                    "title_attr": attrs.get("title", ""),
                    "width": attrs.get("width", ""),
                    "height": attrs.get("height", ""),
                    "loading": attrs.get("loading", ""),
                    "sandbox": attrs.get("sandbox", ""),
                }
            )
    return rows


def _attrs(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(text):
        value = next((group for group in match.groups()[1:] if group is not None), "")
        attrs[match.group(1).lower()] = field_value(html.unescape(value))
    return attrs


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
