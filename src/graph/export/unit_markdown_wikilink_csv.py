"""CSV export for Obsidian-style Markdown wikilinks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "target", "section", "alias", "line_number", "context"]
_WIKILINK_RE = re.compile(r"(?<!\\)\[\[([^\[\]\n]+)\]\]")


def export_units_to_markdown_wikilink_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"]), sort_key(row["section"]), sort_key(row["alias"])))
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
        for match in _WIKILINK_RE.finditer(line):
            parsed = _parse_wikilink(match.group(1))
            if parsed is None:
                continue
            target, section, alias = parsed
            rows.append({"unit_id": uid, "title": title, "target": target, "section": section, "alias": alias, "line_number": line_number, "context": field_value(line)})
    return rows


def _parse_wikilink(raw: str) -> tuple[str, str, str] | None:
    if "|" in raw:
        destination, alias = raw.split("|", 1)
    else:
        destination, alias = raw, ""
    destination = destination.strip()
    alias = alias.strip()
    if not destination or not destination.strip("#") or "|" in alias:
        return None
    if "#" in destination:
        target, section = destination.split("#", 1)
    else:
        target, section = destination, ""
    target = target.strip()
    section = section.strip()
    if not target:
        return None
    return target, section, alias
