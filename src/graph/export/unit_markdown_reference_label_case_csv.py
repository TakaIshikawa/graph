"""CSV export for reference labels that vary by case."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "normalized_label", "observed_labels", "definition_count", "reference_count", "has_case_conflict"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^\s*\[([^\]]+)\]:\s+\S+")
_REF_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\[([^\]\n]+)\]")


def export_units_to_markdown_reference_label_case_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["normalized_label"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"labels": set(), "definitions": 0, "references": 0})
    for _line_number, line in _content_lines(str(get(unit, "content") or "")):
        definition = _DEF_RE.match(line)
        if definition:
            _add(groups, definition.group(1), "definitions")
        for reference in _REF_RE.finditer(line):
            _add(groups, reference.group(1), "references")
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows = []
    for normalized, data in groups.items():
        labels = sorted(data["labels"], key=sort_key)
        rows.append(
            {
                "unit_id": uid,
                "title": title,
                "normalized_label": normalized,
                "observed_labels": "|".join(labels),
                "definition_count": data["definitions"],
                "reference_count": data["references"],
                "has_case_conflict": "true" if len({label.casefold() for label in labels}) == 1 and len(labels) > 1 else "false",
            }
        )
    return rows


def _add(groups: dict[str, dict[str, Any]], label: str, count_key: str) -> None:
    clean = field_value(label)
    if not clean:
        return
    group = groups[clean.casefold()]
    group["labels"].add(clean)
    group[count_key] += 1


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
