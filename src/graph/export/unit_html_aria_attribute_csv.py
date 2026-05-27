"""CSV export for HTML ARIA attributes in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "tag", "attribute", "value", "line_number"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9:-]*)([^<>]*)>")
_ATTR_RE = re.compile(r"""\s(aria-[A-Za-z0-9_.:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""", re.IGNORECASE)


def export_units_to_html_aria_attribute_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["attribute"])))
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
        for tag_match in _TAG_RE.finditer(line):
            tag, attrs = tag_match.groups()
            for attr_match in _ATTR_RE.finditer(attrs):
                rows.append(
                    {
                        "unit_id": uid,
                        "title": title,
                        "tag": tag.casefold(),
                        "attribute": attr_match.group(1).casefold(),
                        "value": field_value(next((item for item in attr_match.groups()[1:] if item is not None), "")),
                        "line_number": line_number,
                    }
                )
    return rows


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
