"""Analyze marginal novelty and saturation across ranked RAG evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, domain_for, iter_strings, metadata, result_id, rounded_ratio, tokens, value


def analyze_evidence_saturation(results: Iterable[Any], query: Any = None) -> dict[str, Any]:
    """Return marginal novelty rows for ranked results."""
    query_terms = tokens(query, min_length=3)
    seen_terms: set[str] = set()
    seen_sources: set[str] = set()
    seen_entities: set[str] = set()
    rows = []
    plateau_index = None

    for index, result in enumerate(results):
        result_terms = tokens(content_text(result), min_length=3)
        term_basis = query_terms or result_terms
        new_terms = sorted((result_terms & term_basis) - seen_terms)
        source = domain_for(result) or (iter_strings(value(result, "source"))[:1] or ["unknown"])[0].casefold()
        source_gain = 0 if source in seen_sources else 1
        entities = _entities(result)
        new_entities = sorted(entities - seen_entities)
        gain = len(new_terms) + source_gain + len(new_entities)
        if plateau_index is None and index > 0 and gain == 0:
            plateau_index = index
        rows.append(
            {
                "result_id": result_id(result, index),
                "rank": index,
                "source_bucket": source,
                "new_terms": new_terms,
                "new_entities": new_entities,
                "source_gain": source_gain,
                "marginal_gain": gain,
            }
        )
        seen_terms.update(result_terms & term_basis)
        seen_sources.add(source)
        seen_entities.update(entities)

    total = len(rows)
    saturated = sum(1 for row in rows if row["marginal_gain"] == 0)
    warnings = ["no_results"] if total == 0 else []
    if plateau_index is not None:
        warnings.append("evidence_plateau")
    return {
        "result_count": total,
        "saturation_score": rounded_ratio(saturated, total),
        "first_plateau_index": plateau_index,
        "coverage_counts": {
            "terms": len(seen_terms),
            "sources": len(seen_sources),
            "entities": len(seen_entities),
        },
        "marginal_gains": rows,
        "warnings": warnings,
    }


def _entities(result: Any) -> set[str]:
    values = []
    for key in ("entities", "entity", "tags", "keywords"):
        values.extend(iter_strings(value(result, key)))
    for key, item in metadata(result).items():
        if key in {"entities", "tags", "keywords"}:
            values.extend(iter_strings(item))
    return {item.casefold() for item in values if item}
