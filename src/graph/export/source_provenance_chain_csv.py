"""CSV export for source provenance chain metadata."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

_FIELDNAMES = ["source_id", "source_name", "adapter", "imported_from", "parent_source", "original_url", "import_batch", "provenance_depth_hint"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_provenance_chain_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source-level provenance fields from direct attributes and metadata."""
    source_list = list(sources)
    rows = _provenance_rows(source_list)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _provenance_rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for source in sources:
        imported_from = _lookup(source, "imported_from")
        parent_source = _lookup(source, "parent_source")
        original_url = _lookup(source, "original_url")
        rows.append(
            {
                "source_id": _source_id(source),
                "source_name": _source_name(source),
                "adapter": _lookup(source, "adapter"),
                "imported_from": imported_from,
                "parent_source": parent_source,
                "original_url": original_url,
                "import_batch": _lookup(source, "import_batch"),
                "provenance_depth_hint": len([value for value in (imported_from, parent_source, original_url) if value]),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_id"]), _sort_key(row["source_name"])))


def _lookup(source: Mapping[str, Any] | object, key: str) -> str:
    direct = _field_value(_get(source, key))
    if direct:
        return direct
    metadata = _get(source, "metadata")
    if isinstance(metadata, Mapping):
        return _field_value(_casefold_get(metadata, key))
    return ""


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


def _source_id(source: Mapping[str, Any] | object) -> str:
    return _field_value(_get(source, "id")) or _field_value(_get(source, "source_id"))


def _source_name(source: Mapping[str, Any] | object) -> str:
    return _field_value(_get(source, "name")) or _field_value(_get(source, "title"))


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
