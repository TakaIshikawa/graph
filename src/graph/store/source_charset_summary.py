"""Summarize source charset metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from email.message import Message
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id


def summarize_source_charsets(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["charset"]]
    non_utf8 = [row for row in present if row["charset"] != "utf-8"]
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "sources_with_charset": len(present),
        "charset_counts": dict(sorted(Counter(row["charset"] for row in present).items())),
        "missing_charset_count": len(source_list) - len(present),
        "non_utf8_count": len(non_utf8),
        "samples": sorted(non_utf8, key=lambda row: (sort_key(row["source_id"]), sort_key(row["charset"])))[:limit],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    value = _charset(source)
    return {"source_id": source_id(source) or str(index), "charset": value}


def _charset(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for key in ("charset", "encoding"):
        value = _normalize(get(source, key)) or _normalize(data.get(key))
        if value:
            return value
    for key in ("content_type", "content-type"):
        value = _content_type_charset(get(source, key)) or _content_type_charset(data.get(key))
        if value:
            return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                name = str(key).casefold().replace("_", "-")
                if name in {"content-type", "charset", "encoding"}:
                    parsed = _content_type_charset(value) if name == "content-type" else _normalize(value)
                    if parsed:
                        return parsed
    return ""


def _content_type_charset(value: object) -> str:
    text = field_value(value)
    if not text:
        return ""
    message = Message()
    message["content-type"] = text
    return _normalize(message.get_param("charset", header="content-type"))


def _normalize(value: object) -> str:
    text = field_value(value).casefold()
    return "utf-8" if text == "utf8" else text
