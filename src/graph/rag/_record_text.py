"""Small record access helpers for RAG analyzers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def first(item: Any, keys: tuple[str, ...]) -> Any:
    for container in (item, value(item, "metadata"), value(item, "unit"), value(value(item, "unit"), "metadata")):
        if container is None:
            continue
        for key in keys:
            found = value(container, key)
            if found not in (None, ""):
                return found
    return None


def record_id(item: Any, index: int, prefix: str = "result") -> str:
    return str(first(item, ("result_id", "context_id", "id", "unit_id", "source_id")) or f"{prefix}-{index + 1}")


def text_blob(item: Any) -> str:
    parts: list[str] = []
    for key in ("title", "snippet", "content", "text"):
        found = first(item, (key,))
        if found not in (None, ""):
            parts.append(str(found))
    meta = value(item, "metadata")
    if isinstance(meta, Mapping):
        parts.extend(str(v) for v in meta.values() if v not in (None, ""))
    return " ".join(parts)
