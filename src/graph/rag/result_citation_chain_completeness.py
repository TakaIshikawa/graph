"""Analyze citation-chain completeness in RAG results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, string, value


def analyze_result_citation_chain_completeness(results: Iterable[Any]) -> dict[str, Any]:
    rows = list(results or [])
    ids: set[str] = set()
    parents: dict[str, str | None] = {}
    missing = 0
    for index, result in enumerate(rows):
        cid = _citation_id(result, index)
        if cid is None:
            missing += 1
            continue
        ids.add(cid)
        parents[cid] = _parent_id(result)

    roots = sorted(cid for cid, parent in parents.items() if not parent)
    linked = sorted(cid for cid, parent in parents.items() if parent and parent in ids)
    orphan = sorted(cid for cid, parent in parents.items() if parent and parent not in ids)
    return {
        "total_results": len(rows),
        "roots": roots,
        "linked_children": linked,
        "orphan_children": len(orphan),
        "missing_citation_id_count": missing,
        "orphan_ids": orphan,
    }


def _citation_id(result: Any, index: int) -> str | None:
    for key in ("citation_id", "source_id", "url", "title"):
        text = string(value(result, key))
        if text:
            return text
    fallback = result_id(result, index)
    return None if fallback.startswith("result-") else fallback


def _parent_id(result: Any) -> str | None:
    for key in ("parent_citation_id", "parent_id", "cited_by"):
        text = string(value(result, key))
        if text:
            return text
    return None
