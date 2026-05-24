"""Summarize source fetch and ingestion error metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit


_ERROR_LIST_KEYS = ("errors", "fetch_errors", "ingestion_errors", "import_errors")
_ERROR_KIND_KEYS = ("error_kind", "kind", "type", "error_type", "exception")
_STATUS_KEYS = ("status_code", "status", "http_status", "response_status")
_TIMESTAMP_KEYS = ("last_seen_at", "seen_at", "occurred_at", "error_at", "failed_at", "fetched_at")
_URL_KEYS = ("url", "source_url", "canonical_url", "fetch_url", "request_url")


def source_fetch_error_summary(units: Iterable[Any]) -> list[dict[str, Any]]:
    """Return stable grouped counts for source fetch or ingestion failures.

    The helper is intentionally read-only and accepts KnowledgeUnit-like objects
    or mappings. Optional metadata fields may be absent; missing grouping values
    are represented as ``None``.
    """

    groups: dict[tuple[str | None, str | None, str | None, str | None], dict[str, Any]] = {}
    for unit in units:
        metadata = _metadata(unit)
        source_id = _get(unit, "source_id")
        source_project = _get(unit, "source_project")
        unit_timestamp = _string(_get(unit, "updated_at") or _get(unit, "ingested_at"))
        entries = _error_entries(metadata)
        for entry in entries:
            status = _first(entry, _STATUS_KEYS)
            status_class = _status_class(status)
            error_kind = _string(_first(entry, _ERROR_KIND_KEYS) or _first(metadata, _ERROR_KIND_KEYS))
            host = _host(_string(_first(entry, _URL_KEYS) or _first(metadata, _URL_KEYS)))
            last_seen_at = _string(_first(entry, _TIMESTAMP_KEYS) or _first(metadata, _TIMESTAMP_KEYS)) or unit_timestamp
            key = (_string(source_id), host, status_class, error_kind)
            if key not in groups:
                groups[key] = {
                    "source_id": _string(source_id),
                    "source_project": _string(source_project),
                    "host": host,
                    "status_class": status_class,
                    "error_kind": error_kind,
                    "count": 0,
                    "last_seen_at": None,
                }
            groups[key]["count"] += 1
            if last_seen_at and (
                groups[key]["last_seen_at"] is None or last_seen_at > groups[key]["last_seen_at"]
            ):
                groups[key]["last_seen_at"] = last_seen_at

    return sorted(
        groups.values(),
        key=lambda row: (
            row["source_id"] or "",
            row["host"] or "",
            row["status_class"] or "",
            row["error_kind"] or "",
        ),
    )


def _error_entries(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    for key in _ERROR_LIST_KEYS:
        value = metadata.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, Mapping))
        elif isinstance(value, Mapping):
            entries.append(value)
    if entries:
        return entries
    if any(metadata.get(key) not in (None, "") for key in (*_ERROR_KIND_KEYS, *_STATUS_KEYS, "error")):
        return [metadata]
    return []


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _status_class(value: Any) -> str | None:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    if status <= 0:
        return None
    return f"{status // 100}xx"


def _host(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url if "://" in url else f"https://{url}")
    return parsed.hostname.lower() if parsed.hostname else None


def _string(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
