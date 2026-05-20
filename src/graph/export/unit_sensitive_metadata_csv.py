"""CSV export for sensitive-looking unit metadata keys."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source", "unit_id", "title", "metadata_key", "redacted_value", "risk_label"]
_DEFAULT_KEY_PATTERNS = ("password", "secret", "token", "api_key", "access_key", "private_key", "credential", "auth")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_sensitive_metadata_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    key_patterns: Iterable[str] | None = None,
) -> str | dict[str, Any]:
    """Return or write redacted findings for sensitive-looking metadata keys."""
    compiled_patterns = _compiled_key_patterns(key_patterns)
    unit_list = list(units)
    rows = _sensitive_rows(unit_list, compiled_patterns)
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


def _compiled_key_patterns(patterns: Iterable[str] | None) -> list[re.Pattern[str]]:
    values = list(_DEFAULT_KEY_PATTERNS if patterns is None else patterns)
    if not values:
        raise ValueError("key_patterns must contain at least one non-empty pattern")
    compiled: list[re.Pattern[str]] = []
    for pattern in values:
        text = _inline_text(pattern)
        if not text:
            raise ValueError("key_patterns must contain only non-empty patterns")
        compiled.append(re.compile(text, re.IGNORECASE))
    return compiled


def _sensitive_rows(
    units: list[KnowledgeUnit | Mapping[str, Any]],
    patterns: list[re.Pattern[str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        metadata = _metadata(unit)
        for key, value in metadata.items():
            metadata_key = _field_value(key)
            if not metadata_key or not _is_sensitive_key(metadata_key, patterns):
                continue
            rows.append(
                {
                    "source": _field_value(_get(unit, "source_project")) or "Unknown",
                    "unit_id": _unit_id(unit),
                    "title": _field_value(_get(unit, "title")),
                    "metadata_key": metadata_key,
                    "redacted_value": _redacted_value(value),
                    "risk_label": _risk_label(metadata_key),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source"]),
            _sort_key(row["unit_id"]),
            _sort_key(row["metadata_key"]),
        ),
    )


def _is_sensitive_key(key: str, patterns: list[re.Pattern[str]]) -> bool:
    searchable = f"{key} {_normalized_key(key)}"
    return any(pattern.search(searchable) for pattern in patterns)


def _risk_label(key: str) -> str:
    normalized = _normalized_key(key)
    if any(token in normalized for token in ("password", "private_key", "secret", "credential")):
        return "high"
    if any(token in normalized for token in ("api_key", "access_key", "token")):
        return "medium"
    return "review"


def _redacted_value(value: object) -> str:
    if value is None:
        return "empty"
    if isinstance(value, bytes):
        length = len(value)
    else:
        length = len(_value_text(value))
    if length == 0:
        return "empty"
    if length <= 4:
        return f"short:{length}"
    return f"long:{length}"


def _value_text(value: object) -> str:
    if isinstance(value, Mapping):
        return f"{len(value)} keys"
    if isinstance(value, list | tuple | set):
        return "; ".join(_value_text(item) for item in value)
    return _field_value(value)


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _render_csv(rows: list[dict[str, str]]) -> str:
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


def _normalized_key(value: object) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", _field_value(value))
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
