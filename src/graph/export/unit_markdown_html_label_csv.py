"""CSV export for Markdown-embedded HTML label elements."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "for_id", "text", "has_control_child"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_LABEL_RE = re.compile(r"<label\b(?P<attrs>[^>]*)>(?P<body>.*?)</label\s*>", re.IGNORECASE)
_FOR_RE = re.compile(r"\bfor\s*=\s*(['\"])(?P<value>.*?)\1", re.IGNORECASE)
_CONTROL_RE = re.compile(r"<\s*(input|select|textarea|button)\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def export_unit_markdown_html_label_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["for_id"]), sort_key(row["text"])))
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
        for match in _LABEL_RE.finditer(line):
            for_match = _FOR_RE.search(match.group("attrs"))
            body = match.group("body")
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "source": source,
                    "line_number": line_number,
                    "for_id": field_value(for_match.group("value") if for_match else ""),
                    "text": field_value(html.unescape(_TAG_RE.sub("", body))),
                    "has_control_child": "true" if _CONTROL_RE.search(body) else "false",
                }
            )
    return rows
