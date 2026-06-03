"""Summarize Content-Type charsets in sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "content-type"


def summarize_source_content_type_charsets(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    sources_with = 0
    for index, source in enumerate(source_list):
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        media_type, charset, status = _parse(value)
        key = (media_type, charset, status)
        row = rows_by_key.setdefault(
            key,
            {"media_type": media_type, "charset": charset, "charset_status": status, "count": 0, "source_ids": [], "examples": []},
        )
        row["count"] += 1
        sid = source_id(source) or str(index)
        if sid not in row["source_ids"] and len(row["source_ids"]) < max(0, sample_limit):
            row["source_ids"].append(sid)
        if value not in row["examples"] and len(row["examples"]) < max(0, sample_limit):
            row["examples"].append(value)
    rows = sorted(rows_by_key.values(), key=lambda row: (sort_key(row["media_type"]), sort_key(row["charset"]), sort_key(row["charset_status"])))
    return {
        "total_sources": len(source_list),
        "sources_with_content_type": sources_with,
        "missing_content_type_count": len(source_list) - sources_with,
        "rows": rows,
    }


def _parse(value: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in value.split(";")]
    media_type = field_value(parts[0]).casefold() if parts else ""
    charset = ""
    malformed = False
    for param in parts[1:]:
        if not param:
            continue
        if "=" not in param:
            malformed = True
            continue
        key, raw = param.split("=", 1)
        if key.strip().casefold() != "charset":
            continue
        text = raw.strip()
        if not text:
            malformed = True
            continue
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        charset = field_value(text).casefold()
    if charset:
        return media_type, charset, "present"
    return media_type, "", "malformed" if malformed else "missing"


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
