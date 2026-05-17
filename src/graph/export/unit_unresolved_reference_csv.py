"""CSV export for unresolved wiki-style unit references."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "reference",
    "normalized_reference",
    "reference_count",
    "matched_candidate_count",
]
_WIKI_REFERENCE_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_unresolved_reference_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write unresolved wiki-style references as deterministic CSV."""
    unit_list = list(units)
    rows = _unresolved_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _unresolved_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    candidates = Counter[str]()
    for unit in units:
        unit_id = _unit_id(unit)
        title = _field_value(_get(unit, "title"))
        if unit_id:
            candidates[_normalize(unit_id)] += 1
        if title:
            candidates[_normalize(title)] += 1

    rows: list[dict[str, str | int]] = []
    for unit in units:
        references = Counter(_references(unit))
        for reference, count in references.items():
            normalized = _normalize(reference)
            matched_count = candidates.get(normalized, 0)
            if matched_count:
                continue
            rows.append(
                {
                    "unit_id": _unit_id(unit),
                    "title": _field_value(_get(unit, "title")),
                    "reference": reference,
                    "normalized_reference": normalized,
                    "reference_count": count,
                    "matched_candidate_count": matched_count,
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["unit_id"]),
            _sort_key(row["normalized_reference"]),
            _sort_key(row["reference"]),
        ),
    )


def _references(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    texts = [_field_value(_get(unit, "content"))]
    texts.extend(_metadata_strings(_metadata(unit)))
    references: list[str] = []
    for text in texts:
        for match in _WIKI_REFERENCE_RE.finditer(text):
            target = match.group(1).split("|", 1)[0]
            reference = _field_value(target)
            if reference:
                references.append(reference)
    return references


def _metadata_strings(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        values: list[str] = []
        for item in value.values():
            values.extend(_metadata_strings(item))
        return values
    if isinstance(value, Iterable):
        values: list[str] = []
        for item in value:
            values.extend(_metadata_strings(item))
        return values
    return []


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize(value: object) -> str:
    return _inline_text(value).casefold()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
