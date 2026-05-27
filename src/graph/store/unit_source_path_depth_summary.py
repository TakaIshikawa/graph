"""Source path depth summary for store units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

PATH_KEYS = ("source_path", "path", "file_path", "url_path")
DEEP_PATH_THRESHOLD = 4


def summarize_unit_source_path_depth(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        grouped[_source(unit)].append(unit)

    rows: list[dict[str, Any]] = []
    for source in sorted(grouped, key=_sort_key):
        depths = [_path_depth(_path(unit)) for unit in grouped[source]]
        valid = [depth for depth in depths if depth is not None]
        rows.append(
            {
                "source": source,
                "source_project": source,
                "unit_count": len(grouped[source]),
                "path_count": len(valid),
                "missing_path_count": len(grouped[source]) - len(valid),
                "min_depth": min(valid) if valid else 0,
                "max_depth": max(valid) if valid else 0,
                "average_depth": round(sum(valid) / len(valid), 2) if valid else 0.0,
                "root_level_count": sum(1 for depth in valid if depth == 1),
                "deep_path_count": sum(1 for depth in valid if depth >= DEEP_PATH_THRESHOLD),
            }
        )
    return {"rows": rows, "source_summaries": rows, "total_units": sum(row["unit_count"] for row in rows)}


def _path(unit: Mapping[str, Any] | object) -> str:
    metadata = _metadata(unit)
    for key in PATH_KEYS:
        text = _text(_get(unit, key)) or _text(metadata.get(key))
        if text:
            return text
    return ""


def _path_depth(path: str) -> int | None:
    if not path:
        return None
    parsed = urlparse(path)
    if parsed.scheme and parsed.netloc:
        path = parsed.path
    path = path.replace("\\", "/").strip("/")
    if not path:
        return 0
    return len([part for part in path.split("/") if part])


def _source(unit: Mapping[str, Any] | object) -> str:
    metadata = _metadata(unit)
    return _text(_get(unit, "source_project")) or _text(_get(unit, "source")) or _text(metadata.get("source")) or "unknown"


def _metadata(value: Mapping[str, Any] | object) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
