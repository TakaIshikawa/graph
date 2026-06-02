"""Measure citation density across RAG retrieval results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import has_evidence, result_id, string, value

_CITATION_KEYS = ("citations", "citation", "citation_id", "source_url", "url", "references", "source")


def analyze_result_citation_density(results: Iterable[Any]) -> dict[str, Any]:
    rows = list(results or [])
    sparse_results = []
    cited_count = 0
    for index, result in enumerate(rows):
        if _has_citation(result):
            cited_count += 1
            continue
        sparse_results.append(
            {
                "id": result_id(result, index),
                "title": string(value(result, "title")) or string(value(result, "name")),
            }
        )

    result_count = len(rows)
    uncited_count = result_count - cited_count
    return {
        "result_count": result_count,
        "cited_result_count": cited_count,
        "uncited_result_count": uncited_count,
        "citation_density": 0.0 if result_count == 0 else round(cited_count / result_count, 4),
        "sparse_results": sparse_results,
    }


def _has_citation(result: Any) -> bool:
    return any(has_evidence(value(result, key)) for key in _CITATION_KEYS)
