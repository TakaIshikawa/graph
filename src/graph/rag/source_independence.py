"""Estimate whether retrieved RAG results are independent sources."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str):
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            if (text := _string(value)):
                return text
    return None


def _id(result: Any, index: int) -> str:
    return _first(result, ("id", "unit_id", "source_id")) or f"result-{index + 1}"


def _domain(raw: str | None) -> str | None:
    if not raw:
        return None
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = parsed.hostname or parsed.netloc
    return host.casefold().removeprefix("www.") if host else None


def _tokens(text: str | None) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9]{4,}", text or "")}


def _add(groups: dict[str, set[str]], reason: str, key: str | None, result_id: str) -> None:
    if key:
        groups[f"{reason}:{key}"].add(result_id)


def analyze_source_independence(results: Iterable[Any]) -> dict[str, Any]:
    """Group likely dependent retrieved results and score source independence."""
    rows: list[dict[str, Any]] = []
    groups: dict[str, set[str]] = defaultdict(set)
    for index, result in enumerate(results):
        result_id = _id(result, index)
        canonical = _first(result, ("canonical_url", "url"))
        row = {
            "result_id": result_id,
            "domain": _domain(_first(result, ("domain", "url", "canonical_url"))),
            "canonical_url": canonical.casefold() if canonical else None,
            "source_id": _first(result, ("source_id",)),
            "author": _first(result, ("author", "byline")),
            "title": _first(result, ("title",)),
            "tokens": _tokens(_first(result, ("content", "text", "snippet"))),
        }
        rows.append(row)
        _add(groups, "domain", row["domain"], result_id)
        _add(groups, "canonical_url", row["canonical_url"], result_id)
        _add(groups, "source_id", row["source_id"], result_id)
        if row["author"] and row["title"]:
            _add(groups, "author_title", f"{row['author'].casefold()}|{row['title'].casefold()}", result_id)

    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            union = left["tokens"] | right["tokens"]
            overlap = len(left["tokens"] & right["tokens"]) / max(len(union), 1)
            if union and overlap >= 0.6:
                key = f"fingerprint:{left['result_id']}:{right['result_id']}"
                groups[key].update([left["result_id"], right["result_id"]])

    output_groups = [
        {"reason": key.split(":", 1)[0], "key": key.split(":", 1)[1], "result_ids": sorted(ids)}
        for key, ids in groups.items()
        if len(ids) > 1
    ]
    output_groups.sort(key=lambda row: (row["reason"], row["key"]))
    dependent_ids = sorted({result_id for group in output_groups for result_id in group["result_ids"]})
    result_count = len(rows)
    score = round((result_count - len(dependent_ids)) / max(result_count, 1), 6)
    return {
        "groups": output_groups,
        "independence_score": score,
        "dependent_result_ids": dependent_ids,
        "summary": {"result_count": result_count, "dependent_result_count": len(dependent_ids), "group_count": len(output_groups)},
    }
