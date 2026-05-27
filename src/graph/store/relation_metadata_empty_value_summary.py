"""Summarize empty top-level relation metadata values."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def summarize_relation_metadata_empty_values(relations: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    relation_count = 0
    empty_value_count = 0
    relations_with_empty: set[str] = set()
    counts_by_key: Counter[str] = Counter()
    counts_by_empty_kind: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for index, relation in enumerate(relations):
        relation_count += 1
        relation_id = _relation_id(relation, index)
        found = False
        for key, value in _metadata(relation).items():
            kind = _empty_kind(value)
            if kind is None:
                continue
            found = True
            empty_value_count += 1
            key_text = str(key)
            counts_by_key[key_text] += 1
            counts_by_empty_kind[kind] += 1
            if len(samples) < sample_limit:
                samples.append({"relation_id": relation_id, "key": key_text, "empty_kind": kind})
        if found:
            relations_with_empty.add(relation_id)

    return {
        "relation_count": relation_count,
        "relations_with_empty_metadata_count": len(relations_with_empty),
        "empty_value_count": empty_value_count,
        "counts_by_key": [{"key": key, "count": counts_by_key[key]} for key in sorted(counts_by_key, key=lambda key: (-counts_by_key[key], key.casefold(), key))],
        "counts_by_empty_kind": [
            {"empty_kind": kind, "count": counts_by_empty_kind[kind]}
            for kind in sorted(counts_by_empty_kind, key=lambda kind: (-counts_by_empty_kind[kind], kind))
        ],
        "samples": samples,
    }


def _metadata(relation: Mapping[str, Any] | object) -> Mapping[str, Any]:
    value = relation.get("metadata") if isinstance(relation, Mapping) else getattr(relation, "metadata", None)
    return value if isinstance(value, Mapping) else {}


def _relation_id(relation: Mapping[str, Any] | object, index: int) -> str:
    if isinstance(relation, Mapping):
        value = relation.get("id")
    else:
        value = getattr(relation, "id", None)
    text = "" if value is None else str(value).strip()
    return text or f"relation:{index + 1}"


def _empty_kind(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, str) and value.strip() == "":
        return "blank_string"
    if isinstance(value, list) and not value:
        return "empty_list"
    if isinstance(value, Mapping) and not value:
        return "empty_dict"
    return None
