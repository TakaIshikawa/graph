"""CSV cross-reference for Markdown footnote references and definitions."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "footnote_label", "reference_count", "definition_line", "first_reference_line", "missing_definition", "unused_definition"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEFINITION_RE = re.compile(r"^\s*\[\^([^\]\n]+)\]:")
_REFERENCE_RE = re.compile(r"\[\^([^\]\n]+)\]")


def export_unit_markdown_footnote_backlinks_to_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write footnote reference/definition cross-reference rows."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row} for row in _footnote_rows(_content(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["footnote_label"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _footnote_rows(content: str) -> list[dict[str, str | int]]:
    references: Counter[str] = Counter()
    first_reference: dict[str, int] = {}
    definitions: dict[str, int] = {}
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        definition = _DEFINITION_RE.match(line)
        if definition:
            definitions.setdefault(definition.group(1), line_number)
            continue
        for match in _REFERENCE_RE.finditer(line):
            label = match.group(1)
            references[label] += 1
            first_reference.setdefault(label, line_number)
    rows = []
    for label in sorted(set(references) | set(definitions), key=sort_key):
        rows.append(
            {
                "footnote_label": field_value(label),
                "reference_count": references[label],
                "definition_line": definitions.get(label, ""),
                "first_reference_line": first_reference.get(label, ""),
                "missing_definition": "true" if label not in definitions else "false",
                "unused_definition": "true" if label not in references else "false",
            }
        )
    return rows
