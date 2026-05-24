"""Store-level source crawl depth summaries."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

DEFAULT_URL_KEYS = ("normalized_url", "url", "source_url", "canonical_url", "external_url")
DEFAULT_PARENT_URL_KEYS = ("parent_url", "referrer_url", "from_url", "referring_url")


def summarize_source_crawl_depth(
    store: Any,
    *,
    url_keys: Sequence[str] = DEFAULT_URL_KEYS,
    parent_url_keys: Sequence[str] = DEFAULT_PARENT_URL_KEYS,
) -> dict[str, Any]:
    """Estimate source URL crawl depth from parent or referrer metadata."""

    source_url_keys = _validate_names(url_keys, "url_keys")
    source_parent_keys = _validate_names(parent_url_keys, "parent_url_keys")
    conn = _connection_from_store(store)
    rows = conn.execute(
        """
        SELECT id, source_project, source_id, source_entity_type, title, metadata
        FROM knowledge_units
        ORDER BY source_project, source_id, source_entity_type, id
        """
    ).fetchall()

    sources = [_source_from_row(row, source_url_keys, source_parent_keys) for row in rows]
    by_url = {source["url"]: source for source in sources if source["url"]}

    for source in sources:
        depth, status = _resolve_depth(source, by_url)
        source["depth"] = depth
        source["status"] = status

    host_counts: Counter[tuple[str, str]] = Counter()
    for source in sources:
        host = source["host"] or ""
        depth_key = str(source["depth"]) if source["depth"] is not None else source["status"]
        host_counts[(host, depth_key)] += 1

    host_depth_counts = [
        {"host": host, "depth": depth, "count": count}
        for (host, depth), count in sorted(host_counts.items(), key=lambda item: item[0])
    ]
    return {"sources": sources, "host_depth_counts": host_depth_counts}


def _connection_from_store(store: Any) -> sqlite3.Connection:
    conn = getattr(store, "conn", store)
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("store must be a Store or sqlite3.Connection")
    conn.row_factory = sqlite3.Row
    return conn


def _validate_names(names: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ValueError(f"{label} must be a sequence of non-empty strings")
    normalized = tuple(name for name in names if isinstance(name, str) and name.strip())
    if len(normalized) != len(names):
        raise ValueError(f"{label} must be a sequence of non-empty strings")
    return normalized


def _source_from_row(
    row: sqlite3.Row,
    url_keys: tuple[str, ...],
    parent_url_keys: tuple[str, ...],
) -> dict[str, Any]:
    metadata = _load_metadata(row["metadata"])
    url = _first_url(metadata, url_keys)
    parent_url = _first_url(metadata, parent_url_keys)
    normalized_url = _normalize_url(url)
    normalized_parent_url = _normalize_url(parent_url)
    return {
        "unit_id": str(row["id"]),
        "source_project": str(row["source_project"]),
        "source_id": str(row["source_id"]),
        "source_entity_type": str(row["source_entity_type"]),
        "url": normalized_url,
        "parent_url": normalized_parent_url,
        "host": _host(normalized_url),
        "depth": None,
        "status": "unresolved",
    }


def _resolve_depth(
    source: dict[str, Any],
    by_url: dict[str, dict[str, Any]],
) -> tuple[int | None, str]:
    if not source["url"]:
        return None, "missing_url"
    if not source["parent_url"]:
        return 0, "root"

    seen: set[str] = set()
    depth = 1
    parent_url = source["parent_url"]
    while parent_url:
        if parent_url in seen or parent_url == source["url"]:
            return None, "cycle"
        seen.add(parent_url)
        parent = by_url.get(parent_url)
        if parent is None:
            return None, "unresolved_parent"
        if not parent["parent_url"]:
            return depth, "resolved"
        parent_url = parent["parent_url"]
        depth += 1
    return depth, "resolved"


def _load_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _first_url(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _metadata_value(metadata, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _metadata_value(metadata: dict[str, Any], path: str) -> Any:
    current: Any = metadata
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _normalize_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return value.strip()
    netloc = parsed.hostname.lower() if parsed.hostname else parsed.netloc.lower()
    if parsed.port and not (
        (parsed.scheme == "http" and parsed.port == 80)
        or (parsed.scheme == "https" and parsed.port == 443)
    ):
        netloc = f"{netloc}:{parsed.port}"
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def _host(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    return parsed.hostname.lower() if parsed.hostname else None
