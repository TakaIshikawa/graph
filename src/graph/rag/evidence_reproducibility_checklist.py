"""Build reproducibility checklists from evidence snippets."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id

_CHECKS: dict[str, tuple[str, re.Pattern[str]]] = {
    "data_availability": (
        "Confirm that underlying data can be inspected or requested.",
        re.compile(r"\b(?:data availability|dataset|data set|repository|supplemental data|available upon request)\b", re.I),
    ),
    "methods_detail": (
        "Confirm that methods are detailed enough to repeat the analysis.",
        re.compile(r"\b(?:methods?|methodology|protocol|procedure|materials and methods|statistical analysis)\b", re.I),
    ),
    "code_availability": (
        "Confirm that code, scripts, or notebooks are available.",
        re.compile(r"\b(?:code availability|github|gitlab|script|notebook|replication code|source code)\b", re.I),
    ),
    "sample_size": (
        "Confirm the sample size or observation count.",
        re.compile(r"\b(?:n\s*=\s*\d+|sample size|participants?|subjects?|respondents?|observations?)\b", re.I),
    ),
    "preregistration": (
        "Confirm preregistration or a registered protocol when relevant.",
        re.compile(r"\b(?:preregistered|pre-registered|preregistration|registered protocol|clinicaltrials\.gov|osf registration)\b", re.I),
    ),
}


def build_evidence_reproducibility_checklist(evidence: Iterable[Any]) -> dict[str, Any]:
    """Return reproducibility checklist rows for evidence-like records."""
    records = list(evidence or [])
    items = [_item(record, index) for index, record in enumerate(records)]
    totals = {
        status: sum(1 for item in items for check in item["checks"] if check["status"] == status)
        for status in ("present", "missing")
    }
    totals["unknown"] = 0 if records else len(_CHECKS)
    return {
        "evidence_count": len(records),
        "items": items,
        "summary": totals,
        "warnings": [] if records else ["no_evidence"],
    }


def _item(record: Any, index: int) -> dict[str, Any]:
    text = _record_text(record)
    checks = []
    for name, (hint, pattern) in _CHECKS.items():
        matched = pattern.search(text)
        checks.append(
            {
                "name": name,
                "status": "present" if matched else "missing",
                "matched_text": matched.group(0) if matched else "",
                "recommendation": "" if matched else hint,
            }
        )
    return {
        "evidence_id": result_id(record, index),
        "checks": checks,
        "reproducibility_score": round(
            sum(1 for check in checks if check["status"] == "present") / len(checks),
            2,
        ),
    }


def _record_text(record: Any) -> str:
    return " ".join([content_text(record), " ".join(iter_strings(metadata(record)))])
