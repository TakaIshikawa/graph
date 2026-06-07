"""Summarize Set-Cookie security attributes in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "set-cookie"


def summarize_source_set_cookie_security(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    cookies = [cookie for row in rows for cookie in row["cookies"]]
    samples = sorted([{"source_id": cookie["source_id"], "cookie_name": cookie["name"], "attributes": cookie["attributes"]} for cookie in cookies], key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    return {
        "total_sources": len(source_list),
        "sources_with_set_cookie": sum(1 for row in rows if row["cookies"]),
        "missing_set_cookie_count": sum(1 for row in rows if not row["cookies"]),
        "cookie_count": len(cookies),
        "secure_count": sum(1 for cookie in cookies if cookie["secure"]),
        "httponly_count": sum(1 for cookie in cookies if cookie["httponly"]),
        "samesite_counts": dict(sorted(Counter(cookie["samesite"] for cookie in cookies if cookie["samesite"]).items())),
        "missing_secure_count": sum(1 for cookie in cookies if not cookie["secure"]),
        "missing_httponly_count": sum(1 for cookie in cookies if not cookie["httponly"]),
        "missing_samesite_count": sum(1 for cookie in cookies if not cookie["samesite"]),
        "partitioned_count": sum(1 for cookie in cookies if cookie["partitioned"]),
        "rows": rows,
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    sid = source_id(source) or str(index)
    return {"source_id": sid, "cookies": [_parse_cookie(value, sid) for value in _cookie_values(_lookup_header(source, _HEADER)) if value.strip()]}


def _cookie_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part for line in value.splitlines() for part in line.split(",") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [field_value(item) for item in value]
    return [field_value(value)] if field_value(value) else []


def _parse_cookie(value: str, source: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    name = parts[0].partition("=")[0]
    attrs = {part.partition("=")[0].casefold(): part.partition("=")[2].casefold() for part in parts[1:]}
    return {
        "source_id": source,
        "name": name,
        "secure": "secure" in attrs,
        "httponly": "httponly" in attrs,
        "samesite": attrs.get("samesite", ""),
        "partitioned": "partitioned" in attrs,
        "attributes": sorted(attrs),
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> Any:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title(), "Set-Cookie"):
            value = get(container, key) if container_name == "source" else container.get(key)
            if field_value(value):
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return value
    return ""
