"""Build prioritized answer verification checks from retrieved RAG results."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import (
    any_present,
    content_text,
    domain_for,
    ordered_terms,
    result_date,
    result_id,
    source_id,
    tokens,
)

_CITATION_KEYS = (
    "url",
    "source_url",
    "canonical_url",
    "external_url",
    "link",
    "permalink",
    "uri",
    "doi",
    "pmid",
    "arxiv_id",
    "isbn",
    "citation",
    "citations",
)
_PROVENANCE_KEYS = (
    "source",
    "source_id",
    "source_project",
    "source_name",
    "author",
    "created_at",
    "updated_at",
    "published_at",
)
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{1,2}(?:-\d{1,2})?)?\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:%| percent)?\b", re.IGNORECASE)


def _empty_plan(query: Any, max_checks: int) -> dict[str, Any]:
    return {
        "query": "" if query is None else str(query),
        "checks": [],
        "counts": {
            "result_count": 0,
            "check_count": 0,
            "with_citations": 0,
            "with_dates": 0,
            "with_provenance": 0,
            "source_count": 0,
        },
        "summary": f"No verification checks generated; 0 usable results and max_checks={max_checks}.",
    }


def _add_check(
    checks: list[dict[str, Any]],
    *,
    code: str,
    priority: int,
    reason: str,
    target_result_ids: list[str],
    suggested_action: str,
) -> None:
    checks.append(
        {
            "id": f"check-{len(checks) + 1:02d}-{code}",
            "priority": priority,
            "reason": reason,
            "target_result_ids": target_result_ids,
            "suggested_action": suggested_action,
        }
    )


def build_answer_verification_plan(
    query: Any,
    results: Iterable[Any],
    *,
    max_checks: int = 8,
) -> dict[str, Any]:
    """Return prioritized checks for verifying an answer grounded in results."""
    if not isinstance(max_checks, int) or isinstance(max_checks, bool) or max_checks < 1:
        max_checks = 8

    try:
        rows = list(results or [])
    except TypeError:
        return _empty_plan(query, max_checks)
    if not rows:
        return _empty_plan(query, max_checks)

    ids = [result_id(result, index) for index, result in enumerate(rows)]
    cited = [rid for rid, result in zip(ids, rows, strict=True) if any_present(result, _CITATION_KEYS)]
    dated = [rid for rid, result in zip(ids, rows, strict=True) if result_date(result) is not None]
    provenance = [rid for rid, result in zip(ids, rows, strict=True) if any_present(result, _PROVENANCE_KEYS)]
    sources = [source_id(result) or domain_for(result) or "unknown" for result in rows]
    source_counts = Counter(sources)

    checks: list[dict[str, Any]] = []
    missing_citations = [rid for rid in ids if rid not in cited]
    if missing_citations:
        _add_check(
            checks,
            code="citations",
            priority=100,
            reason=f"{len(missing_citations)} result(s) lack citation or URL metadata.",
            target_result_ids=missing_citations,
            suggested_action="Confirm answer citations before using these results as evidence.",
        )

    date_hits = {
        rid: sorted(set(_DATE_RE.findall(content_text(result))))
        for rid, result in zip(ids, rows, strict=True)
    }
    date_targets = [rid for rid, hits in date_hits.items() if hits]
    missing_date_metadata = [rid for rid in date_targets if rid not in dated]
    if date_targets:
        _add_check(
            checks,
            code="dates",
            priority=90 if missing_date_metadata else 65,
            reason="Date-like claims appear in retrieved content; verify chronology against result metadata.",
            target_result_ids=date_targets,
            suggested_action="Check every date claim against published, updated, or source timestamp fields.",
        )

    number_targets = [
        rid
        for rid, result in zip(ids, rows, strict=True)
        if _NUMBER_RE.search(content_text(result))
    ]
    if number_targets:
        _add_check(
            checks,
            code="facts",
            priority=85,
            reason="Numeric or count-like facts were found in candidate evidence.",
            target_result_ids=number_targets,
            suggested_action="Recalculate or corroborate numeric claims before finalizing the answer.",
        )

    missing_provenance = [rid for rid in ids if rid not in provenance]
    if missing_provenance:
        _add_check(
            checks,
            code="provenance",
            priority=80,
            reason=f"{len(missing_provenance)} result(s) have weak source provenance metadata.",
            target_result_ids=missing_provenance,
            suggested_action="Prefer results with source, author, project, or timestamp provenance.",
        )

    if len(source_counts) <= 1 and len(rows) > 1:
        _add_check(
            checks,
            code="agreement",
            priority=75,
            reason="All evidence comes from one source, so cross-source agreement is untested.",
            target_result_ids=ids,
            suggested_action="Find an independent source before treating the answer as corroborated.",
        )
    else:
        repeated_sources = [source for source, count in sorted(source_counts.items()) if count > 1]
        if repeated_sources:
            targets = [
                rid
                for rid, source in zip(ids, sources, strict=True)
                if source in repeated_sources
            ]
            _add_check(
                checks,
                code="agreement",
                priority=60,
                reason="Some sources appear more than once; compare them with independent evidence.",
                target_result_ids=targets,
                suggested_action="Check whether repeated-source evidence agrees with distinct sources.",
            )

    query_terms = set(ordered_terms(query))
    weak_targets = [
        rid
        for rid, result in zip(ids, rows, strict=True)
        if query_terms and not (tokens(content_text(result)) & query_terms)
    ]
    if weak_targets:
        _add_check(
            checks,
            code="query-fit",
            priority=55,
            reason="Some results do not contain normalized query terms.",
            target_result_ids=weak_targets,
            suggested_action="Confirm these results are relevant before relying on them.",
        )

    checks.sort(key=lambda item: (-int(item["priority"]), item["id"]))
    checks = checks[:max_checks]
    for index, check in enumerate(checks, start=1):
        check["id"] = f"check-{index:02d}-{check['id'].split('-', 2)[2]}"

    counts = {
        "result_count": len(rows),
        "check_count": len(checks),
        "with_citations": len(cited),
        "with_dates": len(dated),
        "with_provenance": len(provenance),
        "source_count": len(source_counts),
    }
    return {
        "query": "" if query is None else str(query),
        "checks": checks,
        "counts": counts,
        "summary": (
            f"Generated {len(checks)} verification check(s) across {len(rows)} result(s); "
            f"{len(cited)} cited, {len(dated)} dated, {len(source_counts)} source(s)."
        ),
    }
