"""CSV export for HTML time elements embedded in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "datetime", "text", "has_datetime"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TIME_RE = re.compile(r"<time\b(?P<attrs>[^>]*)>(?P<text>.*?)</time\s*>", re.IGNORECASE | re.DOTALL)
_DATETIME_RE = re.compile(r"""(?:^|\s)datetime\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def export_unit_markdown_html_time_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["datetime"]), sort_key(row["text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    rows: list[dict[str, str | int | bool]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _TIME_RE.finditer(line):
            datetime_match = _DATETIME_RE.search(match.group("attrs"))
            datetime_value = field_value(datetime_match.group(1) or datetime_match.group(2) or datetime_match.group(3)) if datetime_match else ""
            rows.append({"unit_id": uid, "title": title, "source": source, "line_number": line_number, "datetime": datetime_value, "text": field_value(_TAG_RE.sub(" ", match.group("text"))), "has_datetime": bool(datetime_value)})
    return rows
