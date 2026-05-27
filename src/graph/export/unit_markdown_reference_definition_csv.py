"""CSV export for Markdown reference-style link definitions in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "label", "destination", "link_title", "line_number", "duplicate_label"]
_REF_DEF_RE = re.compile(r"^[ \t]{0,3}\[([^\]\n]+)\]:[ \t]*(.*)$")


def export_unit_markdown_reference_definition_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(
        key=lambda row: (
            sort_key(row["unit_id"]),
            int(row["line_number"]),
            sort_key(row["label"]),
            sort_key(row["destination"]),
        )
    )
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    definitions = _definitions(str(get(unit, "content") or ""))
    label_counts = Counter(_normalized_label(definition["label"]) for definition in definitions)
    return [
        {
            "unit_id": uid,
            "title": title,
            "label": definition["label"],
            "destination": definition["destination"],
            "link_title": definition["link_title"],
            "line_number": definition["line_number"],
            "duplicate_label": "true" if label_counts[_normalized_label(definition["label"])] > 1 else "false",
        }
        for definition in definitions
    ]


def _definitions(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        match = _REF_DEF_RE.match(line)
        if not match:
            continue
        destination, link_title = _destination_and_title(match.group(2))
        if not destination:
            continue
        rows.append(
            {
                "label": field_value(match.group(1)),
                "destination": destination,
                "link_title": link_title,
                "line_number": line_number,
            }
        )
    return rows


def _destination_and_title(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if not text:
        return "", ""
    if text.startswith("<"):
        end = text.find(">")
        if end == -1:
            return "", ""
        destination = text[1:end]
        rest = text[end + 1 :].strip()
    else:
        parts = text.split(None, 1)
        destination = parts[0]
        rest = parts[1].strip() if len(parts) > 1 else ""
    return field_value(destination), _title(rest)


def _title(text: str) -> str:
    if len(text) < 2:
        return ""
    pairs = {'"': '"', "'": "'", "(": ")"}
    closing = pairs.get(text[0])
    if closing is None or not text.endswith(closing):
        return ""
    return field_value(text[1:-1])


def _normalized_label(text: object) -> str:
    return re.sub(r"\s+", " ", field_value(text)).casefold()
