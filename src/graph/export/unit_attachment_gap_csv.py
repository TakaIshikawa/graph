"""CSV export for units with expected but missing attachment references."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "source_project", "expected_attachment_count", "found_attachment_count", "gap_type", "detail"]
_COUNT_KEYS = ("attachment_count", "file_count", "image_count", "document_count")
_FLAG_KEYS = ("has_attachment", "has_attachments", "attachments_expected")
_REF_KEYS = ("attachment_urls", "attachment_url", "attachments", "files", "file_paths", "file_path", "image_urls", "document_urls")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_attachment_gap_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write units whose attachment metadata lacks usable references."""
    unit_list = list(units)
    rows = _gap_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _gap_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        expected, expected_keys = _expected_count(unit)
        found = _found_count(unit)
        if expected <= 0 or found >= expected:
            continue
        gap_type = "missing_references" if found == 0 else "partial_references"
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                "expected_attachment_count": expected,
                "found_attachment_count": found,
                "gap_type": gap_type,
                "detail": ";".join(sorted(expected_keys, key=_sort_key)),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["unit_id"])))


def _expected_count(unit: KnowledgeUnit | Mapping[str, Any]) -> tuple[int, set[str]]:
    metadata = _metadata(unit)
    count = 0
    keys: set[str] = set()
    for key in _COUNT_KEYS:
        value = _integer(_casefold_get(metadata, key))
        if value > 0:
            count += value
            keys.add(key)
    if count == 0:
        for key in _FLAG_KEYS:
            if _truthy(_casefold_get(metadata, key)):
                count = 1
                keys.add(key)
                break
    return count, keys


def _found_count(unit: KnowledgeUnit | Mapping[str, Any]) -> int:
    metadata = _metadata(unit)
    refs: set[str] = set()
    for key in _REF_KEYS:
        refs.update(_field_value(value) for value in _flatten(_casefold_get(metadata, key)) if _field_value(value))
    return len(refs)


def _flatten(value: object) -> list[object]:
    if value is None or isinstance(value, bytes) or isinstance(value, Mapping):
        return []
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flatten(entry)]
    return [value]


def _integer(value: object) -> int:
    try:
        return max(0, int(float(_field_value(value))))
    except ValueError:
        return 0


def _truthy(value: object) -> bool:
    text = _field_value(value).casefold()
    return text in {"1", "true", "yes", "y"}


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)

