"""Summarize source rate-limit hints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_KEYS = ("retry_after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset")


def summarize_source_rate_limit_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    hinted = [row for row in rows if row["hints"]]
    low = [row for row in hinted if _int(row["hints"].get("x-ratelimit-remaining")) <= 10 and row["hints"].get("x-ratelimit-remaining") != ""]
    limit = max(0, sample_limit)
    key_counts = Counter(key for row in hinted for key in row["hints"])
    provider_counts = Counter(row["provider"] for row in hinted if row["provider"])
    return {
        "total_sources": len(source_list),
        "sources_with_rate_limit_hints": len(hinted),
        "sources_without_rate_limit_hints": len(source_list) - len(hinted),
        "key_counts": dict(sorted(key_counts.items())),
        "provider_counts": dict(sorted(provider_counts.items())),
        "samples": sorted(low, key=lambda row: (_int(row["hints"].get("x-ratelimit-remaining")), sort_key(row["source_id"])))[:limit],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    data = metadata(source)
    hints = {key: value for key in _KEYS if (value := _value(source, data, key))}
    return {"source_id": source_id(source) or str(index), "provider": field_value(get(source, "provider") or data.get("provider")), "hints": hints}


def _value(source: Mapping[str, Any] | object, data: Mapping[str, Any], key: str) -> str:
    names = {key, key.replace("-", "_"), "".join(part.title() if index else part for index, part in enumerate(key.replace("-", "_").split("_")))}
    for name in names:
        value = field_value(get(source, name)) or field_value(data.get(name))
        if value:
            return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for name, value in container.items():
                if str(name).casefold().replace("_", "-") == key:
                    return field_value(value)
    return ""


def _int(value: object) -> int:
    try:
        return int(field_value(value))
    except ValueError:
        return 10**9
