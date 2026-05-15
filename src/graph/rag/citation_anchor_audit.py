"""Audit citation anchors attached to retrieved RAG/search results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()
_ID_KEYS = ("id", "unit_id", "source_id")
_ANCHOR_KEYS = ("citation", "citation_url", "url", "source_url", "canonical_url", "title", "source_title")
_URL_KEYS = ("citation_url", "url", "source_url", "canonical_url")
_TITLE_KEYS = ("title", "source_title")


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            value = unit_metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, _ID_KEYS) or f"result-{index + 1}"


def _normalized_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").casefold()
    if not host:
        return None
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def _issue(result_id: str, issue_type: str, message: str) -> dict[str, str]:
    return {"result_id": result_id, "type": issue_type, "message": message}


def audit_citation_anchors(results: Iterable[Any]) -> dict[str, Any]:
    """Inspect citation fields and report weak, missing, or duplicated anchors."""
    rows = list(results)
    issues: list[dict[str, str]] = []
    anchored_count = 0
    weak_ids: set[str] = set()
    anchors_by_url: dict[str, list[str]] = defaultdict(list)

    for index, result in enumerate(rows):
        result_id = _result_id(result, index)
        title = _first(result, _TITLE_KEYS)
        url = _first(result, _URL_KEYS)
        citation = _first(result, ("citation",))
        normalized_url = _normalized_url(url)
        if normalized_url is not None:
            anchors_by_url[normalized_url].append(result_id)

        anchor_values = [value for value in (title, url, citation) if value]
        if not anchor_values:
            issues.append(_issue(result_id, "missing", "citation anchor is missing"))
            weak_ids.add(result_id)
            continue
        anchored_count += 1

        if citation and len(anchor_values) >= 2:
            continue
        if title and not url and not citation:
            issues.append(_issue(result_id, "title-only", "citation anchor only has a title"))
            weak_ids.add(result_id)
        elif url and not title and not citation:
            issues.append(_issue(result_id, "url-only", "citation anchor only has a URL"))
            weak_ids.add(result_id)
        elif len(set(anchor_values)) < len(anchor_values):
            issues.append(_issue(result_id, "ambiguous", "citation anchor repeats equivalent values"))
            weak_ids.add(result_id)

    duplicate_anchors = {
        anchor: sorted(ids)
        for anchor, ids in sorted(anchors_by_url.items())
        if len(set(ids)) > 1
    }
    for anchor, ids in duplicate_anchors.items():
        for result_id in ids:
            issues.append(
                _issue(result_id, "duplicate", f"citation URL is duplicated as {anchor}")
            )
            weak_ids.add(result_id)

    return {
        "total_results": len(rows),
        "anchored_count": anchored_count,
        "weak_anchor_count": len(weak_ids),
        "duplicate_anchors": duplicate_anchors,
        "issues": sorted(issues, key=lambda item: (item["result_id"], item["type"])),
    }
