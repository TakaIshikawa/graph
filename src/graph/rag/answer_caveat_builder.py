"""Build concise answer caveats from RAG retrieval diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_ATTRIBUTION_FIELDS = {"source", "url", "title", "author", "attribution"}
_DATE_FIELDS = {"date", "published_date", "published_at", "created_at", "updated_at"}


def build_answer_caveats(
    query: str,
    results: Iterable[Any],
    diagnostics: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    *,
    min_results: int = 2,
    stale_age_days: int | float = 365,
    min_coverage_score: int | float = 0.5,
) -> list[str]:
    """Return ordered caveats for weak retrieval evidence.

    The builder only consumes the supplied results and optional diagnostic
    dictionaries. It does not run retrieval analysis itself, so callers can pass
    precomputed output from any diagnostic helper with compatible field names.
    """
    result_count = len(list(results))
    rows = _diagnostic_rows(diagnostics)
    caveats: list[str] = []

    if result_count == 0:
        caveats.append("No retrieved results were available, so the answer may be incomplete.")
    elif result_count < min_results:
        plural = "result" if result_count == 1 else "results"
        caveats.append(f"Only {result_count} retrieved {plural} supported the answer.")

    if _has_missing_attribution(rows):
        caveats.append("Some retrieved evidence is missing source attribution.")
    if _has_stale_evidence(rows, float(stale_age_days)):
        caveats.append("Some retrieved evidence may be stale.")
    if _has_conflicting_evidence(rows):
        caveats.append("Retrieved evidence contains possible conflicts.")
    if _has_weak_coverage(rows, float(min_coverage_score)):
        caveats.append("Retrieved evidence has weak coverage of the query.")
    if _has_missing_dates(rows):
        caveats.append("Some retrieved evidence is missing usable dates.")

    return _dedupe(caveats)


def _diagnostic_rows(diagnostics: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if diagnostics is None:
        return []
    rows: list[Mapping[str, Any]] = []
    _collect_mappings(diagnostics, rows)
    return rows


def _collect_mappings(value: Any, rows: list[Mapping[str, Any]]) -> None:
    if isinstance(value, Mapping):
        rows.append(value)
        for nested in value.values():
            _collect_mappings(nested, rows)
        return
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for nested in value:
            _collect_mappings(nested, rows)


def _has_missing_attribution(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        if _truthy(row.get("missing_attribution")):
            return True
        if _positive_any(row, ("missing_source", "missing_url", "missing_title", "missing_author")):
            return True
        missing_fields = _string_set(row.get("missing_fields"))
        if missing_fields & _ATTRIBUTION_FIELDS:
            return True
    return False


def _has_stale_evidence(rows: list[Mapping[str, Any]], stale_age_days: float) -> bool:
    for row in rows:
        if _truthy(row.get("stale")) or _truthy(row.get("has_stale_evidence")):
            return True
        age_days = _number(row.get("age_days"))
        if age_days is not None and age_days >= stale_age_days:
            return True
        freshness_score = _number(row.get("freshness_score"))
        if freshness_score is not None and freshness_score <= 0.2 and not _has_missing_date_reason(row):
            return True
    return False


def _has_conflicting_evidence(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        if _truthy(row.get("conflicting_evidence")) or _truthy(row.get("has_conflicts")):
            return True
        if _positive_any(row, ("conflict_count", "contradiction_count", "tension_count")):
            return True
        flags = row.get("flags")
        if isinstance(flags, Iterable) and not isinstance(flags, str | bytes | Mapping) and any(flags):
            return True
    return False


def _has_weak_coverage(rows: list[Mapping[str, Any]], min_coverage_score: float) -> bool:
    for row in rows:
        if _truthy(row.get("weak_query_coverage")):
            return True
        coverage_score = _number(row.get("coverage_score"))
        if coverage_score is not None and coverage_score < min_coverage_score:
            return True
        missing_terms = row.get("missing_terms")
        matched_terms = row.get("matched_terms")
        if _non_empty_iterable(missing_terms) and not _non_empty_iterable(matched_terms):
            return True
    return False


def _has_missing_dates(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        if _truthy(row.get("missing_dates")) or _truthy(row.get("missing_date")):
            return True
        if _has_missing_date_reason(row):
            return True
        missing_fields = _string_set(row.get("missing_fields"))
        if missing_fields & _DATE_FIELDS:
            return True
        if _positive_any(row, ("missing_date_metadata", "missing_published_date", "missing_dates_count")):
            return True
    return False


def _has_missing_date_reason(row: Mapping[str, Any]) -> bool:
    reason = row.get("reason")
    return isinstance(reason, str) and "missing date" in reason.casefold()


def _positive_any(row: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any((number is not None and number > 0) for number in (_number(row.get(key)) for key in keys))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "stale", "missing", "weak"}
    return bool(value)


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().casefold()} if value.strip() else set()
    if isinstance(value, Iterable) and not isinstance(value, bytes | Mapping):
        return {
            str(item).strip().casefold()
            for item in value
            if str(item).strip()
        }
    return set()


def _non_empty_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping) and any(value)


def _dedupe(caveats: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for caveat in caveats:
        if caveat in seen:
            continue
        seen.add(caveat)
        unique.append(caveat)
    return unique
