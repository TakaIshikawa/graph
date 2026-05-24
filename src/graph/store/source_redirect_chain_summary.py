"""Summarize redirect chains encoded in source metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit


_URL_KEYS = ("url", "source_url", "canonical_url", "fetch_url")
_TARGET_KEYS = ("redirect_target", "redirect_url", "redirected_to", "final_url")


def source_redirect_chain_summary(units: Iterable[Any]) -> list[dict[str, Any]]:
    records = [_record(unit) for unit in units]
    by_url = {record["url"]: record for record in records if record["url"]}
    rows = []
    for record in records:
        start = record["url"]
        current = record
        seen = {record["source_id"]}
        hosts = {_host(start)}
        hop_count = 0
        loop = False
        final_url = start
        while current["target"]:
            target = current["target"]
            target_record = by_url.get(target)
            hop_count += 1
            hosts.add(_host(target))
            final_url = target
            if target_record is None:
                break
            if target_record["source_id"] in seen:
                loop = True
                final_url = target_record["url"]
                break
            seen.add(target_record["source_id"])
            current = target_record
        rows.append(
            {
                "source_id": record["source_id"],
                "source_project": record["source_project"],
                "start_url": start,
                "final_url": final_url,
                "hop_count": hop_count,
                "loop": loop,
                "cross_host": len({host for host in hosts if host}) > 1,
            }
        )
    return sorted(rows, key=lambda row: row["source_id"] or "")


def _record(unit: Any) -> dict[str, Any]:
    metadata = _metadata(unit)
    return {
        "source_id": _string(_get(unit, "source_id")),
        "source_project": _string(_get(unit, "source_project")),
        "url": _normalize_url(_first(metadata, _URL_KEYS)),
        "target": _normalize_url(_first(metadata, _TARGET_KEYS)),
    }


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_url(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    return text if "://" in text else f"https://{text}"


def _host(url: str | None) -> str | None:
    if not url:
        return None
    return urlsplit(url).hostname.lower() if urlsplit(url).hostname else None


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
