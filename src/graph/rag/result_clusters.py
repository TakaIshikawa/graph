"""Cluster RAG/search result dictionaries by shared tags and source."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_MISSING = object()
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class _ResultItem:
    index: int
    result_id: str
    title: str | None
    source: str | None
    tag_keys: frozenset[str]
    tag_labels: dict[str, str]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parents[right_root] = left_root


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    label = " ".join(str(value).strip().split())
    return label or None


def _unit_value(unit: Any, key: str) -> Any:
    if isinstance(unit, Mapping):
        return unit.get(key, _MISSING)
    return getattr(unit, key, _MISSING)


def _result_value(result: Mapping[str, Any], key: str) -> Any:
    value = result.get(key, _MISSING)
    if value is not _MISSING and value is not None:
        return value

    unit = result.get("unit", _MISSING)
    if unit is _MISSING or unit is None:
        return value
    nested_value = _unit_value(unit, key)
    if nested_value is not _MISSING:
        return nested_value
    return value


def _tag_key(label: str) -> str:
    return " ".join(label.casefold().split())


def _display_key(label: str) -> tuple[str, str]:
    return (label.casefold(), label)


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.casefold()).strip("-")
    return slug or "cluster"


def _tags(raw_tags: Any) -> dict[str, str]:
    if not isinstance(raw_tags, Iterable) or isinstance(raw_tags, str | bytes):
        return {}

    labels_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    for raw_tag in raw_tags:
        label = _string_value(raw_tag)
        if label is None:
            continue
        labels_by_key[_tag_key(label)][label] += 1

    return {
        key: sorted(labels, key=lambda label: (-labels[label], _display_key(label)))[0]
        for key, labels in labels_by_key.items()
        if key
    }


def _result_id(result: Mapping[str, Any], index: int) -> str:
    for key in ("id", "unit_id"):
        value = _string_value(result.get(key, _MISSING))
        if value is not None:
            return value

    value = _string_value(_result_value(result, "id"))
    if value is not None:
        return value

    return f"result-{index + 1}"


def _item(result: Mapping[str, Any], index: int) -> _ResultItem:
    tag_labels = _tags(_result_value(result, "tags"))
    return _ResultItem(
        index=index,
        result_id=_result_id(result, index),
        title=_string_value(_result_value(result, "title")),
        source=_string_value(_result_value(result, "source_project")),
        tag_keys=frozenset(tag_labels),
        tag_labels=tag_labels,
    )


def _should_link(left: _ResultItem, right: _ResultItem, min_shared_tags: int) -> bool:
    if left.tag_keys and right.tag_keys:
        return len(left.tag_keys & right.tag_keys) >= min_shared_tags
    return left.source is not None and left.source == right.source


def _component_items(
    items: list[_ResultItem],
    *,
    max_clusters: int,
    min_shared_tags: int,
) -> list[list[_ResultItem]]:
    components_by_root: dict[int, list[_ResultItem]] = defaultdict(list)
    disjoint_set = _DisjointSet(len(items))

    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            right = items[right_index]
            if _should_link(left, right, min_shared_tags):
                disjoint_set.union(left_index, right_index)

    for index, item in enumerate(items):
        components_by_root[disjoint_set.find(index)].append(item)

    components = list(components_by_root.values())
    components.sort(key=_component_sort_key)
    if len(components) <= max_clusters:
        return components

    kept = components[: max_clusters - 1]
    overflow = [item for component in components[max_clusters - 1 :] for item in component]
    overflow.sort(key=lambda item: (item.result_id, item.index))
    return kept + [overflow]


def _component_sort_key(component: list[_ResultItem]) -> tuple[int, str, str]:
    label = _cluster_label(component)
    first_id = min(item.result_id for item in component)
    return (-len(component), first_id, label.casefold())


def _cluster_label(component: list[_ResultItem]) -> str:
    tag_counts: Counter[str] = Counter()
    tag_labels: dict[str, Counter[str]] = defaultdict(Counter)
    tag_first_seen: dict[str, tuple[str, int]] = {}
    source_counts: Counter[str] = Counter()

    for item in component:
        if item.source is not None:
            source_counts[item.source] += 1
        for key, label in item.tag_labels.items():
            tag_counts[key] += 1
            tag_labels[key][label] += 1
            tag_first_seen[key] = min(
                tag_first_seen.get(key, (item.result_id, item.index)),
                (item.result_id, item.index),
            )

    if tag_counts:
        prominent_tags = sorted(
            tag_counts,
            key=lambda key: (
                -tag_counts[key],
                tag_first_seen[key],
                _display_key(_tag_label(key, tag_labels[key])),
            ),
        )[:2]
        return " + ".join(_tag_label(key, tag_labels[key]) for key in prominent_tags)

    if source_counts:
        return sorted(source_counts, key=lambda source: (-source_counts[source], source))[0]

    return "Unlabeled results"


def _tag_label(key: str, labels: Counter[str]) -> str:
    if not labels:
        return key
    return sorted(labels, key=lambda label: (-labels[label], _display_key(label)))[0]


def _cluster_tags(component: list[_ResultItem]) -> list[str]:
    tag_counts: Counter[str] = Counter()
    tag_labels: dict[str, Counter[str]] = defaultdict(Counter)
    tag_first_seen: dict[str, tuple[str, int]] = {}
    for item in component:
        for key, label in item.tag_labels.items():
            tag_counts[key] += 1
            tag_labels[key][label] += 1
            tag_first_seen[key] = min(
                tag_first_seen.get(key, (item.result_id, item.index)),
                (item.result_id, item.index),
            )

    return [
        _tag_label(key, tag_labels[key])
        for key in sorted(
            tag_counts,
            key=lambda key: (
                -tag_counts[key],
                tag_first_seen[key],
                _display_key(_tag_label(key, tag_labels[key])),
            ),
        )
    ]


def _cluster_sources(component: list[_ResultItem]) -> list[str]:
    return sorted({item.source for item in component if item.source is not None})


def _representative_title(component: list[_ResultItem]) -> str | None:
    titled_items = [item for item in component if item.title is not None]
    if not titled_items:
        return None
    return sorted(titled_items, key=lambda item: (item.result_id, item.index))[0].title


def _cluster_dict(component: list[_ResultItem], index: int) -> dict[str, Any]:
    label = _cluster_label(component)
    result_ids = sorted(item.result_id for item in component)
    return {
        "id": f"cluster-{index + 1}-{_slug(label)}",
        "label": label,
        "size": len(component),
        "tags": _cluster_tags(component),
        "sources": _cluster_sources(component),
        "result_ids": result_ids,
        "representative_title": _representative_title(component),
    }


def cluster_results_by_overlap(
    results: Iterable[Mapping[str, Any]],
    *,
    max_clusters: int = 8,
    min_shared_tags: int = 1,
) -> list[dict[str, Any]]:
    """Return deterministic lightweight clusters for RAG/search results."""
    max_clusters_value = _validate_positive_int(max_clusters, "max_clusters")
    min_shared_tags_value = _validate_positive_int(min_shared_tags, "min_shared_tags")
    result_items = [
        _item(result, index)
        for index, result in enumerate(results)
        if isinstance(result, Mapping)
    ]
    if not result_items:
        return []

    components = _component_items(
        result_items,
        max_clusters=max_clusters_value,
        min_shared_tags=min_shared_tags_value,
    )
    return [_cluster_dict(component, index) for index, component in enumerate(components)]
