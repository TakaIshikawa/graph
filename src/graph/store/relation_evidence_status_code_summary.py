"""Summarize HTTP status codes found in relation evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_STATUS_KEYS = ("status_code", "status", "http_status", "response_status")
_EVIDENCE_KEYS = ("evidence", "evidence_items", "supporting_evidence")
_ID_KEYS = ("id", "relation_id", "edge_id")
_FALLBACK_KEYS = ("source", "source_id", "target", "target_id", "type", "relation_type", "predicate")


def summarize_relation_evidence_status_codes(relations: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = with_codes = invalid_count = 0
    status_counts: Counter[int] = Counter()
    class_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for relation in relations:
        total += 1
        found_valid = False
        for value in _status_values(relation):
            code = _status_code(value)
            if code is None:
                invalid_count += 1
                if len(samples) < limit:
                    samples.append({"relation_id": _relation_id(relation), "status_code": field_value(value), "valid": False})
                continue
            found_valid = True
            status_counts[code] += 1
            class_counts[f"{code // 100}xx"] += 1
            if len(samples) < limit:
                samples.append({"relation_id": _relation_id(relation), "status_code": code, "valid": True})
        if found_valid:
            with_codes += 1

    return {
        "total_relations": total,
        "relations_with_status_codes": with_codes,
        "status_counts": {str(key): status_counts[key] for key in sorted(status_counts)},
        "status_class_counts": {key: class_counts[key] for key in sorted(class_counts, key=sort_key)},
        "invalid_count": invalid_count,
        "samples": samples,
    }


def _status_values(relation: Any) -> list[Any]:
    values: list[Any] = []
    meta = metadata(relation)
    for source in (relation, meta):
        for key in _STATUS_KEYS:
            raw = get(source, key) if source is relation else source.get(key)
            if raw not in (None, ""):
                values.append(raw)
    for evidence in _evidence_values(relation):
        if isinstance(evidence, Mapping):
            for key in _STATUS_KEYS:
                raw = evidence.get(key)
                if raw not in (None, ""):
                    values.append(raw)
    return values


def _evidence_values(relation: Any) -> list[Any]:
    values: list[Any] = []
    meta = metadata(relation)
    for source in (relation, meta):
        for key in _EVIDENCE_KEYS:
            raw = get(source, key) if source is relation else source.get(key)
            if raw not in (None, ""):
                values.extend(raw if isinstance(raw, list | tuple | set) else [raw])
    return values


def _status_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 100 <= value <= 599 else None
    text = field_value(value)
    if text.isdecimal():
        code = int(text)
        return code if 100 <= code <= 599 else None
    return None


def _relation_id(relation: Any) -> str:
    meta = metadata(relation)
    for key in _ID_KEYS:
        value = field_value(get(relation, key)) or field_value(meta.get(key))
        if value:
            return value
    parts = [field_value(get(relation, key)) or field_value(meta.get(key)) for key in _FALLBACK_KEYS]
    return "|".join(part for part in parts if part)
