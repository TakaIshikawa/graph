"""Build deterministic hierarchies from slash-delimited unit tags."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.types.models import KnowledgeUnit


def _validate_min_count(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("min_count must be a positive integer")
    return value


def _tag_path(raw_tag: str) -> list[str]:
    return [part.strip() for part in raw_tag.strip().split("/") if part.strip()]


def _node(
    tag: str,
    parent: str | None,
    depth: int,
    unit_ids: set[str],
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tag": tag,
        "parent": parent,
        "depth": depth,
        "count": len(unit_ids),
        "unit_ids": sorted(unit_ids),
        "children": children,
    }


def build_tag_hierarchy(
    units: Iterable[KnowledgeUnit],
    *,
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """Return nested tag nodes derived from slash-delimited unit tags.

    Counts are unit counts, not raw tag occurrences. Repeated tags on the same
    unit only count once, and nested tags contribute to each path prefix.
    """
    min_count_value = _validate_min_count(min_count)
    unit_ids_by_tag: dict[str, set[str]] = defaultdict(set)
    children_by_parent: dict[str | None, set[str]] = defaultdict(set)
    depth_by_tag: dict[str, int] = {}

    for unit in units:
        unit_paths: set[str] = set()
        for raw_tag in unit.tags:
            if not isinstance(raw_tag, str):
                continue
            path = _tag_path(raw_tag)
            for index in range(len(path)):
                unit_paths.add("/".join(path[: index + 1]))

        for tag in unit_paths:
            parts = tag.split("/")
            parent = "/".join(parts[:-1]) if len(parts) > 1 else None
            unit_ids_by_tag[tag].add(unit.id)
            children_by_parent[parent].add(tag)
            depth_by_tag[tag] = len(parts) - 1

    kept_tags = {
        tag for tag, unit_ids in unit_ids_by_tag.items() if len(unit_ids) >= min_count_value
    }

    def build_children(parent: str | None) -> list[dict[str, Any]]:
        child_tags = [tag for tag in children_by_parent[parent] if tag in kept_tags]
        child_tags.sort(key=lambda tag: (-len(unit_ids_by_tag[tag]), tag))
        return [
            _node(
                tag=tag,
                parent=parent,
                depth=depth_by_tag[tag],
                unit_ids=unit_ids_by_tag[tag],
                children=build_children(tag),
            )
            for tag in child_tags
        ]

    return build_children(None)
