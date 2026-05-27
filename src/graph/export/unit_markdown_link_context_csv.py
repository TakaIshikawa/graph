"""CSV export for Markdown links with bounded surrounding context."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "label", "target", "line_number", "context_before", "context_after"]
_INLINE_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_REF_USE_RE = re.compile(r"(?<!!)\[([^\]]+)\]\[([^\]]*)\]")
_REF_DEF_RE = re.compile(r"^\s*\[([^\]]+)\]:\s*(\S+)")


def export_units_to_markdown_link_context_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None, context_chars: int = 40) -> str | dict[str, Any]:
    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row} for row in _links(str(get(unit, "content") or ""), context_chars))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"]), sort_key(row["label"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _links(content: str, context_chars: int) -> list[dict[str, str | int]]:
    definitions: dict[str, str] = {}
    lines = content.splitlines()
    for line in lines:
        match = _REF_DEF_RE.match(line)
        if match:
            definitions[match.group(1).casefold()] = match.group(2)
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(lines, start=1):
        if _REF_DEF_RE.match(line):
            match = _REF_DEF_RE.match(line)
            assert match is not None
            rows.append(_row(line, match.start(), match.end(), match.group(1), match.group(2), line_number, context_chars))
            continue
        for match in _INLINE_RE.finditer(line):
            rows.append(_row(line, match.start(), match.end(), match.group(1), match.group(2), line_number, context_chars))
        for match in _REF_USE_RE.finditer(line):
            label = match.group(1)
            ref = match.group(2) or label
            target = definitions.get(ref.casefold(), "")
            rows.append(_row(line, match.start(), match.end(), label, target, line_number, context_chars))
    return rows


def _row(line: str, start: int, end: int, label: str, target: str, line_number: int, context_chars: int) -> dict[str, str | int]:
    return {"label": field_value(label), "target": field_value(target), "line_number": line_number, "context_before": field_value(line[max(0, start - context_chars) : start]), "context_after": field_value(line[end : end + context_chars])}
