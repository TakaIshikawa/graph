"""Build compact answer readiness briefs from RAG/search results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.citation_coverage import analyze_citation_coverage
from graph.rag.context_gaps import detect_context_gaps
from graph.rag.date_coverage import analyze_result_date_coverage
from graph.rag.evidence_packets import build_evidence_packets


def _ratio(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _readiness_label(score: float) -> str:
    if score >= 0.8:
        return "ready"
    if score >= 0.5:
        return "partial"
    return "blocked"


def _supporting_sources(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for packet in packets:
        source = packet.get("source") or packet.get("source_project") or "unknown"
        row = grouped.setdefault(
            str(source),
            {
                "source": str(source),
                "count": 0,
                "unit_ids": [],
                "citation_count": 0,
                "average_evidence_strength": 0.0,
                "_strength_total": 0.0,
            },
        )
        row["count"] += 1
        row["unit_ids"].append(packet["id"])
        row["citation_count"] += 1 if packet.get("citation_fields") else 0
        row["_strength_total"] += float(packet.get("evidence_strength") or 0.0)

    rows = []
    for row in grouped.values():
        row["unit_ids"] = sorted(row["unit_ids"])
        row["average_evidence_strength"] = round(row["_strength_total"] / row["count"], 6)
        del row["_strength_total"]
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["count"],
            -row["citation_count"],
            row["source"],
        )
    )
    return rows


def _blocking_gaps(
    *,
    result_count: int,
    citation_coverage: Mapping[str, Any],
    date_coverage: Mapping[str, Any],
    context_gaps: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    if result_count == 0:
        gaps.append(
            {
                "type": "empty_results",
                "severity": "error",
                "message": "No retrieved results are available for answer generation.",
            }
        )
    if citation_coverage.get("with_citation_count", 0) == 0 and result_count:
        gaps.append(
            {
                "type": "missing_citations",
                "severity": "error",
                "message": "No retrieved results include citation metadata.",
            }
        )
    if date_coverage.get("dated_results", 0) == 0 and result_count:
        gaps.append(
            {
                "type": "missing_dates",
                "severity": "warning",
                "message": "No retrieved results include parseable date metadata.",
            }
        )
    for gap in context_gaps.get("gaps", []):
        if gap.get("severity") in {"error", "warning"}:
            gaps.append(dict(gap))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for gap in gaps:
        key = (str(gap.get("type")), str(gap.get("message")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gap)
    return deduped


def _recommended_actions(blocking_gaps: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for gap in blocking_gaps:
        gap_type = gap.get("type")
        if gap_type == "empty_results":
            actions.append("Run a broader retrieval before drafting the answer.")
        elif gap_type == "missing_citations":
            actions.append("Retrieve or enrich results with URLs, identifiers, or references.")
        elif gap_type == "missing_dates":
            actions.append("Add dated sources before making time-sensitive claims.")
        elif gap_type == "source_diversity":
            actions.append("Add evidence from additional source projects.")
        elif isinstance(gap_type, str) and gap_type.startswith("missing_required_"):
            actions.append("Retrieve context for the missing required facets.")
    if not actions:
        actions.append("Proceed with answer generation and cite the strongest packets.")
    return sorted(set(actions))


def build_answer_readiness_brief(
    query: str,
    results: Iterable[Any],
    required_facets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact readiness brief for downstream answer generation."""
    result_list = list(results)
    packets = build_evidence_packets(result_list, query=query, limit=None)
    citation_coverage = analyze_citation_coverage(result_list)
    date_coverage = analyze_result_date_coverage(result_list)
    context_gaps = detect_context_gaps(
        result_list,
        required_facets=required_facets,
        min_sources=1,
    )
    blocking_gaps = _blocking_gaps(
        result_count=len(result_list),
        citation_coverage=citation_coverage,
        date_coverage=date_coverage,
        context_gaps=context_gaps,
    )
    average_strength = (
        sum(float(packet["evidence_strength"]) for packet in packets) / len(packets)
        if packets
        else 0.0
    )
    readiness_score = round(
        min(
            1.0,
            average_strength * 0.45
            + _ratio(citation_coverage.get("citation_coverage_ratio")) * 0.25
            + _ratio(date_coverage.get("coverage_ratio")) * 0.2
            + (0.1 if not blocking_gaps else 0.0),
        ),
        6,
    )

    return {
        "query": query,
        "result_count": len(result_list),
        "readiness_score": readiness_score,
        "readiness_label": _readiness_label(readiness_score),
        "blocking_gaps": blocking_gaps,
        "coverage": {
            "citation": citation_coverage,
            "date": date_coverage,
            "context": context_gaps,
        },
        "supporting_sources": _supporting_sources(packets),
        "date_coverage": date_coverage,
        "citation_coverage": citation_coverage,
        "evidence_packets": packets,
        "recommended_next_actions": _recommended_actions(blocking_gaps),
    }
