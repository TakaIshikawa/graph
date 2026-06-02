"""CSV export for Markdown inline hashtags."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "tag_text", "normalized_tag", "line_number", "column", "context"]
_TAG_RE = re.compile(r"(?<![\w/#])#([A-Za-z][A-Za-z0-9_-]*)")


def export_units_to_markdown_hashtag_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["column"]), sort_key(row["normalized_tag"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_project") or metadata(unit).get("source"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        for match in _TAG_RE.finditer(line):
            if _inside_code_span(line, match.start()) or _inside_url(line, match.start()):
                continue
            tag_text = f"#{match.group(1)}"
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "title": title,
                    "source": source,
                    "tag_text": tag_text,
                    "normalized_tag": match.group(1).casefold().replace("_", "-"),
                    "line_number": line_number,
                    "column": match.start() + 1,
                    "context": field_value(line)[:160],
                }
            )
    return rows


def _inside_code_span(line: str, offset: int) -> bool:
    return line[:offset].count("`") % 2 == 1


def _inside_url(line: str, offset: int) -> bool:
    prefix = line[:offset]
    token_start = max(prefix.rfind(" "), prefix.rfind("("), prefix.rfind("[")) + 1
    token = line[token_start:offset]
    return token.startswith(("http://", "https://")) or "://" in token
