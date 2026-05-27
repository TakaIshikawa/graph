"""CSV export for blockquote attribution coverage."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "quote_index", "start_line", "end_line", "has_attribution", "attribution_text", "quote_preview"]
_ATTR_RE = re.compile(r"^\s*(?:[>-]\s*)?(?:--|—|-)\s*(.+)$|^\s*(?:source|citation):\s*(.+)$", re.IGNORECASE)


def export_units_to_blockquote_attribution_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        lines = str(get(unit, "content") or "").splitlines()
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        for index, block in enumerate(_blocks(lines), start=1):
            start, end, texts = block
            attribution = _attribution(texts[-1]) or _following_attribution(lines, end)
            rows.append({"unit_id": unit_id(unit), "title": title, "quote_index": index, "start_line": start, "end_line": end, "has_attribution": "true" if attribution else "false", "attribution_text": attribution, "quote_preview": field_value(" ".join(texts))[:120]})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["quote_index"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _blocks(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    blocks: list[tuple[int, int, list[str]]] = []
    start = 0
    texts: list[str] = []
    for index, line in enumerate(lines, start=1):
        if line.lstrip().startswith(">"):
            if not texts:
                start = index
            texts.append(field_value(line.lstrip()[1:].strip()))
        elif texts:
            blocks.append((start, index - 1, texts))
            texts = []
    if texts:
        blocks.append((start, len(lines), texts))
    return blocks


def _attribution(text: str) -> str:
    match = _ATTR_RE.match(text)
    return field_value(next((group for group in match.groups() if group), "")) if match else ""


def _following_attribution(lines: list[str], end_line: int) -> str:
    for line in lines[end_line:]:
        if not field_value(line):
            continue
        return _attribution(line)
    return ""
