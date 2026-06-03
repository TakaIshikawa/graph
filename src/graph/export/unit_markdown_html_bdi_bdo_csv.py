"""CSV export for Markdown-embedded HTML bdi and bdo elements."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "tag", "dir", "text", "has_dir"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_ELEMENT_RE = re.compile(r"<(?P<tag>bdi|bdo)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)\s*>", re.IGNORECASE)
_DIR_RE = re.compile(r"\bdir\s*=\s*(['\"])(?P<value>.*?)\1", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def export_unit_markdown_html_bdi_bdo_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["text"])))
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
        for match in _ELEMENT_RE.finditer(line):
            body = match.group("body")
            if not field_value(_TAG_RE.sub("", body)):
                continue
            dir_match = _DIR_RE.search(match.group("attrs"))
            direction = field_value(dir_match.group("value") if dir_match else "").casefold()
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "source": source,
                    "line_number": line_number,
                    "tag": match.group("tag").casefold(),
                    "dir": direction,
                    "text": field_value(html.unescape(_TAG_RE.sub("", body))),
                    "has_dir": "true" if direction else "false",
                }
            )
    return rows
