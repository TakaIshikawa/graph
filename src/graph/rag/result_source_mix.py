"""Summarize source diversity across retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import any_present, domain_for, result_id, rounded_ratio, string, value

_PROVENANCE_KEYS = (
    "url",
    "source_url",
    "source",
    "source_id",
    "source_project",
    "domain",
    "source_domain",
    "author",
    "published_at",
    "updated_at",
    "created_at",
)


def _bucket(result: Any, keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        text = string(value(result, key))
        if text is not None:
            return text
    return default


def _summary(counter: Counter[str], total: int) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": count, "percentage": rounded_ratio(count, total)}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def analyze_result_source_mix(results: Iterable[Any]) -> dict[str, Any]:
    """Return source-project, domain, entity-type, and provenance summaries."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    total = len(rows)
    projects: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    entity_types: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    result_rows: list[dict[str, Any]] = []

    for index, result in enumerate(rows):
        project = _bucket(result, ("source_project", "project", "source_name", "source"), "unknown_project")
        domain = domain_for(result) or "unknown_domain"
        entity_type = _bucket(result, ("entity_type", "type", "unit_type", "content_type"), "unknown_entity_type")
        has_provenance = any_present(result, _PROVENANCE_KEYS)
        provenance_bucket = "with_provenance" if has_provenance else "missing_provenance"

        projects[project] += 1
        domains[domain] += 1
        entity_types[entity_type] += 1
        provenance[provenance_bucket] += 1
        result_rows.append(
            {
                "result_id": result_id(result, index),
                "source_project": project,
                "source_domain": domain,
                "entity_type": entity_type,
                "provenance": provenance_bucket,
            }
        )

    return {
        "total_results": total,
        "source_project": _summary(projects, total),
        "source_domain": _summary(domains, total),
        "entity_type": _summary(entity_types, total),
        "provenance": _summary(provenance, total),
        "results": result_rows,
    }
