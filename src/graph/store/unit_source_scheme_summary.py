"""Source URI scheme summary for store units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

SOURCE_KEYS = ("source_url", "url", "external_url", "source_path", "path")


def summarize_unit_source_schemes(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_count": 0, "example_unit_ids": [], "example_sources": []})
    total = 0
    for index, unit in enumerate(units):
        total += 1
        source = _source(unit)
        scheme = _scheme(source)
        group = groups[scheme]
        group["unit_count"] += 1
        if len(group["example_unit_ids"]) < sample_limit:
            group["example_unit_ids"].append(_unit_id(unit, index))
        if source and len(group["example_sources"]) < sample_limit:
            group["example_sources"].append(source)
    rows = [{"scheme": scheme, **groups[scheme]} for scheme in sorted(groups, key=_sort_key)]
    return {"total_units": total, "rows": rows}


def _source(unit: Any) -> str:
    meta = _metadata(unit)
    for key in SOURCE_KEYS:
        text = _text(_get(unit, key)) or _text(meta.get(key))
        if text:
            return text
    return ""


def _scheme(source: str) -> str:
    if not source:
        return "missing"
    if any(char.isspace() for char in source) and "://" in source:
        return "malformed"
    parsed = urlparse(source)
    if parsed.scheme:
        if parsed.scheme in {"http", "https"} and not parsed.netloc:
            return "malformed"
        return parsed.scheme.casefold()
    return "relative"


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(unit: Any, index: int) -> str:
    return _text(_get(unit, "id") or _metadata(unit).get("id")) or str(index)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(getattr(value, "value", value)).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
