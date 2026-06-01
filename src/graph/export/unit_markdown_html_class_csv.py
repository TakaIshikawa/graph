"""CSV export for HTML class attributes in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "tag", "class_name", "class_count"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<\s*([A-Za-z][A-Za-z0-9:-]*)\b([^<>]*)>")
_CLASS_RE = re.compile(r"""(?:^|\s)class\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", re.IGNORECASE)


def export_units_to_markdown_html_class_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["class_name"])))
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
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for tag_match in _TAG_RE.finditer(line):
            class_match = _CLASS_RE.search(tag_match.group(2))
            if not class_match:
                continue
            classes = [class_name for class_name in (class_match.group(1) or class_match.group(2) or class_match.group(3) or "").split() if class_name]
            for class_name in classes:
                rows.append(
                    {
                        "unit_id": uid,
                        "title": title,
                        "line_number": line_number,
                        "tag": field_value(tag_match.group(1).lower()),
                        "class_name": field_value(class_name),
                        "class_count": len(classes),
                    }
                )
    return rows
