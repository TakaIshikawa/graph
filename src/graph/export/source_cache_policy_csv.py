"""CSV export for source cache policy metadata."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = ["cache_policy", "source_count", "min_ttl_seconds", "max_ttl_seconds", "etag_count", "last_modified_count"]
_MAX_AGE_RE = re.compile(r"\bmax-age\s*=\s*(\d+)\b", re.IGNORECASE)


def export_source_cache_policy_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write grouped source cache-policy rows."""
    source_list = list(sources)
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_count": 0, "ttls": [], "etag_count": 0, "last_modified_count": 0})
    for source in source_list:
        policy = _policy(source)
        bucket = groups[policy]
        bucket["source_count"] += 1
        ttl = _ttl(source)
        if ttl is not None:
            bucket["ttls"].append(ttl)
        bucket["etag_count"] += int(bool(_value(source, "etag")))
        bucket["last_modified_count"] += int(bool(_value(source, "last_modified")))

    rows = []
    for policy in sorted(groups, key=sort_key):
        bucket = groups[policy]
        ttls = bucket["ttls"]
        rows.append(
            {
                "cache_policy": policy,
                "source_count": bucket["source_count"],
                "min_ttl_seconds": min(ttls) if ttls else "",
                "max_ttl_seconds": max(ttls) if ttls else "",
                "etag_count": bucket["etag_count"],
                "last_modified_count": bucket["last_modified_count"],
            }
        )
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _policy(source: Mapping[str, Any] | object) -> str:
    explicit = _value(source, "cache_policy")
    if explicit:
        return _normalize_policy(explicit)
    cache_control = _value(source, "cache_control")
    lower = cache_control.casefold()
    if _truthy(_value(source, "no_store")) or "no-store" in lower:
        return "no-store"
    if _truthy(_value(source, "no_cache")) or "no-cache" in lower:
        return "no-cache"
    if _truthy(_value(source, "immutable")) or "immutable" in lower:
        return "immutable"
    if _ttl(source) is not None:
        return "ttl"
    if "must-revalidate" in lower or "proxy-revalidate" in lower or "revalidate" in lower:
        return "revalidate"
    return "unknown"


def _normalize_policy(value: str) -> str:
    text = value.strip().casefold().replace("_", "-").replace(" ", "-")
    if text in {"no-store", "nostore"}:
        return "no-store"
    if text in {"no-cache", "nocache"}:
        return "no-cache"
    if text in {"immutable", "ttl", "revalidate"}:
        return text
    return text or "unknown"


def _ttl(source: Mapping[str, Any] | object) -> int | None:
    for key in ("ttl_seconds", "ttl", "max_age", "max-age"):
        parsed = _parse_seconds(_value(source, key))
        if parsed is not None:
            return parsed
    match = _MAX_AGE_RE.search(_value(source, "cache_control"))
    return int(match.group(1)) if match else None


def _parse_seconds(value: str) -> int | None:
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        return int(text)
    match = _MAX_AGE_RE.search(text)
    return int(match.group(1)) if match else None


def _value(source: Mapping[str, Any] | object, key: str) -> str:
    aliases = {key, key.replace("_", "-"), key.replace("_", " ").title(), key.upper()}
    data = metadata(source)
    for alias in aliases:
        text = field_value(get(source, alias))
        if text:
            return text
    for alias in aliases:
        text = field_value(data.get(alias))
        if text:
            return text
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for header, value in container.items():
                if str(header).casefold().replace("-", "_") == key.casefold().replace("-", "_"):
                    text = field_value(value)
                    if text:
                        return text
    return ""


def _truthy(value: str) -> bool:
    return value.casefold() in {"1", "true", "yes", "y", "on"}
