"""Summarize checksum metadata drift against current unit content."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_CHECKSUM_KEYS = ("sha256", "content_sha256", "md5", "content_md5", "checksum", "content_hash")


def summarize_unit_content_checksum_drift(units: Iterable[Any]) -> dict[str, Any]:
    counts = {"matching": 0, "missing": 0, "stale": 0, "algorithm_unknown": 0, "missing_content": 0}
    stale_examples = []
    total = 0
    for index, unit in enumerate(units):
        total += 1
        content = _content(unit)
        checksum = _checksum(unit)
        if content is None:
            counts["missing_content"] += 1
            continue
        if checksum is None:
            counts["missing"] += 1
            continue
        algorithm, expected = checksum
        if algorithm not in {"sha256", "md5"}:
            counts["algorithm_unknown"] += 1
            continue
        actual = hashlib.new(algorithm, content.encode("utf-8")).hexdigest()
        if actual == expected.lower():
            counts["matching"] += 1
        else:
            counts["stale"] += 1
            stale_examples.append({"unit_id": unit_id(unit) or str(index), "algorithm": algorithm, "expected": expected})
    stale_examples.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": total, "counts": counts, "stale_examples": stale_examples}


def _content(unit: Any) -> str | None:
    value = get(unit, "content")
    if value is None:
        value = metadata(unit).get("content")
    return None if value is None else str(value)


def _checksum(unit: Any) -> tuple[str, str] | None:
    meta = metadata(unit)
    for key in _CHECKSUM_KEYS:
        value = field_value(get(unit, key)) or field_value(meta.get(key))
        if not value:
            continue
        if key in {"sha256", "content_sha256"}:
            return ("sha256", value)
        if key in {"md5", "content_md5"}:
            return ("md5", value)
        if ":" in value:
            algorithm, digest = value.split(":", 1)
            return (algorithm.casefold(), digest)
        if len(value) == 64:
            return ("sha256", value)
        if len(value) == 32:
            return ("md5", value)
        return ("unknown", value)
    return None
