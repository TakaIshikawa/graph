"""Summarize Sec-Fetch metadata headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADERS = {
    "site": "sec-fetch-site",
    "mode": "sec-fetch-mode",
    "dest": "sec-fetch-dest",
    "user": "sec-fetch-user",
}
_EXPECTED_VALUES = {
    "site": {"same-origin", "same-site", "cross-site", "none"},
    "mode": {"cors", "navigate", "no-cors", "same-origin", "websocket"},
    "dest": {
        "audio",
        "audioworklet",
        "document",
        "embed",
        "empty",
        "font",
        "frame",
        "iframe",
        "image",
        "manifest",
        "object",
        "paintworklet",
        "report",
        "script",
        "serviceworker",
        "sharedworker",
        "style",
        "track",
        "video",
        "worker",
        "xslt",
    },
    "user": {"?0", "?1"},
}


def summarize_source_sec_fetch_metadata(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if any(row[name] for name in _HEADERS)]
    samples = sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    return {
        "total_sources": len(source_list),
        "sources_with_sec_fetch_metadata": len(present),
        "missing_metadata_count": len(source_list) - len(present),
        "site_value_counts": _value_counts(rows, "site"),
        "mode_value_counts": _value_counts(rows, "mode"),
        "dest_value_counts": _value_counts(rows, "dest"),
        "user_value_counts": _value_counts(rows, "user"),
        "cross_site_navigation_count": sum(1 for row in rows if row["site"] == "cross-site" and row["mode"] == "navigate"),
        "unusual_value_count": sum(1 for row in rows for name in _HEADERS if row[name] and row[name] not in _EXPECTED_VALUES[name]),
        "rows": sorted(present, key=lambda row: sort_key(row["source_id"])),
        "samples": samples,
    }


def _value_counts(rows: list[dict[str, str]], name: str) -> dict[str, int]:
    return dict(sorted(Counter(row[name] for row in rows if row[name]).items()))


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    row = {"source_id": source_id(source) or str(index)}
    for name, header in _HEADERS.items():
        row[name] = field_value(_lookup_header(source, header)).casefold()
    return row


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> Any:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = get(container, key) if container_name == "source" else container.get(key)
            if field_value(value):
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return value
    return ""
