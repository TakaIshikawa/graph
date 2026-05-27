"""CSV export for HTML comments in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line", "comment_text", "context"]
_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.S)


def export_unit_markdown_html_comment_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["comment_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    content = str(get(unit, "content") or "")
    line_starts = _line_starts(content)
    return [
        {
            "unit_id": uid,
            "title": title,
            "line": _line_for(line_starts, match.start()),
            "comment_text": field_value(match.group(1)),
            "context": field_value(_line_at(content, match.start())),
        }
        for match in _COMMENT_RE.finditer(content)
    ]


def _line_starts(text: str) -> list[int]:
    return [0] + [match.end() for match in re.finditer(r"\n", text)]


def _line_for(starts: list[int], offset: int) -> int:
    return sum(1 for start in starts if start <= offset)


def _line_at(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    return text[start:] if end == -1 else text[start:end]
