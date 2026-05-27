"""Summarize nested depth of unit metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import metadata, sort_key, unit_id


def summarize_unit_yaml_nested_depth(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    buckets: Counter[str] = Counter()
    examples = []
    max_depth = 0
    for unit in units:
        total_units += 1
        depth, paths = _depth(metadata(unit))
        max_depth = max(max_depth, depth)
        buckets[str(depth)] += 1
        if len(examples) < sample_limit:
            examples.append({"unit_id": unit_id(unit), "depth": depth, "key_path": paths[0] if paths else ""})
    examples.sort(key=lambda row: (-row["depth"], sort_key(row["unit_id"])))
    return {"total_units": total_units, "max_depth": max_depth, "depth_buckets": dict(sorted(buckets.items(), key=lambda item: int(item[0]))), "deepest_examples": examples[:sample_limit]}


def _depth(value: Any, path: str = "") -> tuple[int, list[str]]:
    if isinstance(value, Mapping):
        if not value:
            return (0, [path] if path else [])
        best_depth = -1
        best_paths: list[str] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            child_depth, child_paths = _depth(child, child_path)
            depth = 1 + child_depth
            if depth > best_depth:
                best_depth = depth
                best_paths = child_paths or [child_path]
            elif depth == best_depth:
                best_paths.extend(child_paths or [child_path])
        return best_depth, sorted(best_paths, key=sort_key)
    if isinstance(value, list | tuple):
        if not value:
            return (0, [path] if path else [])
        best_depth = -1
        best_paths: list[str] = []
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            child_depth, child_paths = _depth(child, child_path)
            depth = 1 + child_depth
            if depth > best_depth:
                best_depth = depth
                best_paths = child_paths or [child_path]
            elif depth == best_depth:
                best_paths.extend(child_paths or [child_path])
        return best_depth, sorted(best_paths, key=sort_key)
    return (0, [path] if path else [])
