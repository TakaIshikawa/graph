"""CSV export for unit content and attachment MIME type inventory."""

from __future__ import annotations

import csv
import mimetypes
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["mime_type", "count", "unit_ids", "source_keys"]
_MIME_KEYS = {"mime_type", "content_type", "media_type", "attachment_mime_type"}
_EXTENSION_KEYS = {
    "attachment",
    "attachments",
    "content_file",
    "content_path",
    "file",
    "files",
    "link",
    "links",
    "path",
    "paths",
    "url",
    "urls",
}
_EXTENSION_MIME_OVERRIDES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
_UNKNOWN = "unknown"
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_mime_type_inventory_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write an aggregate inventory of unit MIME type hints."""
    unit_list = list(units)
    rows = _inventory_rows(unit_list)
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


def _inventory_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "unit_ids": set(), "source_keys": set()})
    for unit in units:
        unit_id = _unit_id(unit)
        hints = _mime_hints(unit)
        if not hints:
            hints = [(_UNKNOWN, "missing")]
        for mime_type, source_key in hints:
            bucket = buckets[mime_type or _UNKNOWN]
            bucket["count"] += 1
            if unit_id:
                bucket["unit_ids"].add(unit_id)
            bucket["source_keys"].add(source_key)

    rows: list[dict[str, str | int]] = []
    for mime_type, bucket in buckets.items():
        rows.append(
            {
                "mime_type": mime_type,
                "count": bucket["count"],
                "unit_ids": "; ".join(sorted(bucket["unit_ids"], key=_sort_key)),
                "source_keys": "; ".join(sorted(bucket["source_keys"], key=_sort_key)),
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["mime_type"]))


def _mime_hints(unit: KnowledgeUnit | Mapping[str, Any]) -> list[tuple[str, str]]:
    hints: list[tuple[str, str]] = []
    metadata = _metadata(unit)

    for raw_key, value in metadata.items():
        key = _field_value(raw_key)
        normalized_key = _normalized_key(key)
        if normalized_key in _MIME_KEYS:
            values = _flatten(value)
            if not values:
                hints.append((_UNKNOWN, key))
            for item in values:
                hints.append((_normalize_mime(item) or _UNKNOWN, key))

    if any(mime_type != _UNKNOWN for mime_type, _source_key in hints):
        return hints

    for raw_key, value in metadata.items():
        key = _field_value(raw_key)
        if not _is_extension_source_key(key):
            continue
        for item in _flatten(value):
            mime_type = _mime_from_path(item)
            if mime_type:
                hints.append((mime_type, f"{key}:extension"))

    return hints


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _flatten(value: object) -> list[object]:
    if value is None or isinstance(value, bytes) or isinstance(value, Mapping):
        return []
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flatten(entry)]
    return [value]


def _normalize_mime(value: object) -> str:
    text = _field_value(value)
    if not text:
        return ""
    mime_type = text.split(";", 1)[0].strip().casefold()
    return mime_type if "/" in mime_type else text.casefold()


def _mime_from_path(value: object) -> str:
    text = _field_value(value)
    if not text:
        return ""
    parsed = urlparse(text)
    candidate = parsed.path if parsed.scheme else text
    suffix = Path(candidate).suffix.casefold()
    if suffix in _EXTENSION_MIME_OVERRIDES:
        return _EXTENSION_MIME_OVERRIDES[suffix]
    mime_type, _encoding = mimetypes.guess_type(candidate)
    return _normalize_mime(mime_type)


def _is_extension_source_key(key: str) -> bool:
    normalized_key = _normalized_key(key)
    return (
        normalized_key in _EXTENSION_KEYS
        or "attachment" in normalized_key
        or normalized_key.endswith("_file")
        or normalized_key.endswith("_path")
    )


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


def _normalized_key(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_")


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
