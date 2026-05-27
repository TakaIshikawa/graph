"""Relation metadata completeness summary."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

DEFAULT_REQUIRED_KEYS = ("confidence", "evidence", "created_at", "source")


def summarize_relation_metadata_completeness(
    relations: Iterable[Mapping[str, Any] | object], required_keys: Sequence[str] | None = None
) -> dict[str, Any]:
    keys = tuple(required_keys or DEFAULT_REQUIRED_KEYS)
    grouped: dict[tuple[str, str], list[Mapping[str, Any] | object]] = defaultdict(list)
    for relation in relations:
        grouped[(_relation_type(relation), _source(relation))].append(relation)

    rows: list[dict[str, Any]] = []
    for (relation_type, source), group in sorted(grouped.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))):
        missing_counts: Counter[str] = Counter()
        complete_count = 0
        missing_metadata_count = 0
        for relation in group:
            metadata = _metadata(relation)
            if not metadata:
                missing_metadata_count += 1
            missing = [key for key in keys if not _present(_field(relation, metadata, key))]
            if missing:
                missing_counts.update(missing)
            else:
                complete_count += 1
        rows.append(
            {
                "relation_type": relation_type,
                "source": source,
                "relation_count": len(group),
                "complete_count": complete_count,
                "incomplete_count": len(group) - complete_count,
                "missing_metadata_count": missing_metadata_count,
                "missing_key_counts": [{"key": key, "count": missing_counts[key]} for key in sorted(missing_counts, key=_sort_key)],
            }
        )
    return {"rows": rows, "relation_summaries": rows, "total_relations": sum(row["relation_count"] for row in rows)}


def _field(relation: Mapping[str, Any] | object, metadata: Mapping[str, Any], key: str) -> object:
    value = _get(relation, key)
    return value if _present(value) else metadata.get(key)


def _present(value: object) -> bool:
    if isinstance(value, (list, tuple, set, Mapping)):
        return bool(value)
    return bool(_text(value))


def _relation_type(relation: Mapping[str, Any] | object) -> str:
    return _text(_get(relation, "relation_type")) or _text(_get(relation, "type")) or "unknown"


def _source(relation: Mapping[str, Any] | object) -> str:
    metadata = _metadata(relation)
    return _text(_get(relation, "source")) or _text(_get(relation, "source_project")) or _text(metadata.get("source")) or _text(metadata.get("source_project")) or "unknown"


def _metadata(value: Mapping[str, Any] | object) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
