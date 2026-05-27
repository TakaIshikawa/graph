"""CSV export for Markdown reference-style link usages in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "label", "link_text", "usage_type", "line_number"]
_REF_DEF_RE = re.compile(r"^[ \t]{0,3}\[[^\]\n]+]:")
_FULL_OR_COLLAPSED_RE = re.compile(r"(?<!!)\[([^\]\n]+)]\[([^\]\n]*)]")
_SHORTCUT_RE = re.compile(r"(?<!!)(?<!])\[([^\]\n]+)](?![\[(])")
_USAGE_ORDER = {"full": 0, "collapsed": 1, "shortcut": 2}


def export_unit_markdown_reference_usage_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), _USAGE_ORDER[str(row["usage_type"])], sort_key(row["label"]), sort_key(row["link_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _REF_DEF_RE.match(line):
            continue
        for usage in _usages(line):
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, **usage})
    return rows


def _usages(line: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    masked = list(line)
    for match in _FULL_OR_COLLAPSED_RE.finditer(line):
        link_text = field_value(match.group(1))
        label = field_value(match.group(2)) or link_text
        usage_type = "collapsed" if not field_value(match.group(2)) else "full"
        if label and link_text:
            rows.append({"label": label, "link_text": link_text, "usage_type": usage_type})
        masked[match.start() : match.end()] = " " * (match.end() - match.start())
    for match in _SHORTCUT_RE.finditer("".join(masked)):
        label = field_value(match.group(1))
        if label:
            rows.append({"label": label, "link_text": label, "usage_type": "shortcut"})
    return rows
