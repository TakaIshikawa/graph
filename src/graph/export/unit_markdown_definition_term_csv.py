"""CSV export for Markdown definition terms and definitions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "term_line_number", "definition_line_number", "term_text", "definition_preview"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^\s*:\s*(.*)$")


def export_unit_markdown_definition_term_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["term_line_number"]), int(row["definition_line_number"]), sort_key(row["term_text"])))
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
    content = str(get(unit, "content") or data.get("content") or "")
    rows: list[dict[str, str | int]] = []
    in_fence = False
    term_text = ""
    term_line = 0
    pending: dict[str, str | int] | None = None
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _DEF_RE.match(line)
        if match and term_text:
            if pending:
                rows.append(pending)
            pending = {"unit_id": uid, "title": title, "source": source, "term_line_number": term_line, "definition_line_number": line_number, "term_text": term_text, "definition_preview": field_value(match.group(1))}
            continue
        if pending and (line.startswith(" ") or line.startswith("\t")) and line.strip():
            pending["definition_preview"] = field_value(f"{pending['definition_preview']} {line.strip()}")
            continue
        if pending:
            rows.append(pending)
            pending = None
        if line.strip():
            term_text = field_value(line)
            term_line = line_number
        else:
            term_text = ""
            term_line = 0
    if pending:
        rows.append(pending)
    return rows
