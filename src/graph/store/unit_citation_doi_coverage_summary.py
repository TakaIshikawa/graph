"""Summarize DOI coverage for unit citation metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_CITATION_KEYS = ("citations", "references", "citation_records")
_DOI_KEYS = ("doi", "DOI", "Doi")
_UNIT_ID_KEYS = ("id", "unit_id")


def summarize_unit_citation_doi_coverage(units: Iterable[Any]) -> dict[str, Any]:
    """Aggregate citation DOI coverage across units."""

    total_units = units_with_citations = total_citations = doi_citations = 0
    units_missing_dois: list[str] = []

    for unit in units:
        total_units += 1
        citations = _citations(unit)
        if citations:
            units_with_citations += 1
        missing_for_unit = False
        for citation in citations:
            total_citations += 1
            if _doi(citation):
                doi_citations += 1
            else:
                missing_for_unit = True
        if missing_for_unit:
            units_missing_dois.append(_unit_id(unit))

    missing_doi_citations = total_citations - doi_citations
    return {
        "total_units": total_units,
        "units_with_citations": units_with_citations,
        "total_citations": total_citations,
        "doi_citations": doi_citations,
        "missing_doi_citations": missing_doi_citations,
        "coverage_ratio": doi_citations / total_citations if total_citations else 0.0,
        "units_missing_dois": sorted(units_missing_dois),
    }


def _citations(unit: Any) -> list[Mapping[str, Any]]:
    metadata = _metadata(unit)
    for key in _CITATION_KEYS:
        value = _get(unit, key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        value = metadata.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _doi(citation: Mapping[str, Any]) -> str | None:
    for key in _DOI_KEYS:
        value = citation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key, value in citation.items():
        if key.lower() == "doi" and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(item: Any) -> str:
    for key in _UNIT_ID_KEYS:
        value = _get(item, key)
        if value not in (None, ""):
            return str(value)
    return ""


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)
