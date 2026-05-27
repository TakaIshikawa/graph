"""CSV export for per-unit content size counts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, inline_text, render_csv, sort_key, unit_id, write_csv

_FIELDS = ["unit_id", "title", "word_count", "line_count", "paragraph_count", "character_count"]
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def export_units_to_word_count_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDS)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    return {
        "unit_id": unit_id(unit),
        "title": inline_text(get(unit, "title")),
        "word_count": len(_WORD_RE.findall(content)),
        "line_count": len(content.splitlines()) if content else 0,
        "paragraph_count": len([block for block in re.split(r"(?:\r?\n[ \t]*){2,}", content.strip()) if block.strip()]),
        "character_count": len(content),
    }
