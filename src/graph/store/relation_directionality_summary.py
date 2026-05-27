"""Directionality summary for graph relations."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def summarize_relation_directionality(relations: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any] | object]] = defaultdict(list)
    for relation in relations:
        grouped[(_relation_type(relation), _source(relation))].append(relation)

    rows: list[dict[str, Any]] = []
    for (relation_type, source), group in sorted(grouped.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))):
        endpoint_pairs = [_endpoint_pair(relation) for relation in group]
        directed_count = sum(1 for relation in group if _is_directed(relation))
        missing_endpoint_count = sum(1 for source_id, target_id in endpoint_pairs if not source_id or not target_id)
        self_loop_count = sum(1 for source_id, target_id in endpoint_pairs if source_id and source_id == target_id)
        rows.append(
            {
                "relation_type": relation_type,
                "source": source,
                "relation_count": len(group),
                "directed_count": directed_count,
                "undirected_count": len(group) - directed_count,
                "self_loop_count": self_loop_count,
                "missing_endpoint_count": missing_endpoint_count,
                "reciprocal_candidate_count": _reciprocal_candidate_count(endpoint_pairs),
            }
        )
    return {"rows": rows, "relation_summaries": rows, "total_relations": sum(row["relation_count"] for row in rows)}


def _reciprocal_candidate_count(endpoint_pairs: list[tuple[str, str]]) -> int:
    counts = Counter(pair for pair in endpoint_pairs if pair[0] and pair[1] and pair[0] != pair[1])
    total = 0
    for (source_id, target_id), count in counts.items():
        if _sort_key(source_id) < _sort_key(target_id):
            total += min(count, counts.get((target_id, source_id), 0))
    return total


def _endpoint_pair(relation: Mapping[str, Any] | object) -> tuple[str, str]:
    return (_endpoint(relation, ("source_id", "from_unit_id", "from_id", "source")), _endpoint(relation, ("target_id", "to_unit_id", "to_id", "target")))


def _endpoint(relation: Mapping[str, Any] | object, keys: tuple[str, ...]) -> str:
    metadata = _metadata(relation)
    for key in keys:
        text = _text(_get(relation, key)) or _text(metadata.get(key))
        if text:
            return text
    return ""


def _is_directed(relation: Mapping[str, Any] | object) -> bool:
    metadata = _metadata(relation)
    for key in ("directed", "is_directed"):
        value = _get(relation, key)
        if value is not None:
            return _bool(value)
        if key in metadata:
            return _bool(metadata.get(key))
    return True


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    text = _text(value).casefold()
    if text in {"false", "0", "no", "n", "off", "undirected"}:
        return False
    return True


def _relation_type(relation: Mapping[str, Any] | object) -> str:
    metadata = _metadata(relation)
    return _text(_get(relation, "relation_type")) or _text(_get(relation, "type")) or _text(metadata.get("relation_type")) or _text(metadata.get("type")) or "unknown"


def _source(relation: Mapping[str, Any] | object) -> str:
    metadata = _metadata(relation)
    return _text(_get(relation, "source_project")) or _text(metadata.get("source_project")) or _text(metadata.get("source")) or _text(_get(relation, "source")) or "unknown"


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
