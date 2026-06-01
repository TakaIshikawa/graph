"""Analyze RAG results for duplicate normalized titles."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import first, record_id


def analyze_result_duplicate_title_signals(results: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    display: dict[str, str] = {}
    for index, item in enumerate(results):
        title = first(item, ("title",))
        norm = _normalize_title(title)
        if not norm:
            continue
        display.setdefault(norm, str(title).strip())
        groups[norm].append({"result_id": record_id(item, index), "title": str(title).strip()})
    duplicate_keys = sorted(key for key, members in groups.items() if len(members) > 1)
    return {
        "duplicate_group_count": len(duplicate_keys),
        "duplicate_titles": [display[key] for key in duplicate_keys],
        "duplicate_results": [{"title": display[key], "members": groups[key]} for key in duplicate_keys],
    }


def _normalize_title(title: Any) -> str:
    text = " ".join(str(title or "").lower().split())
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())
