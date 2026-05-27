"""Audit whether answer citation labels match supplied evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, string, value

_CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]")
_LABEL_KEYS = ("citation_label", "label", "citation", "id", "source_id", "result_id")


def audit_answer_citation_consistency(answer: Any, evidence: Iterable[Any] | Any) -> dict[str, Any]:
    """Return citation labels that are valid, missing, and unused."""
    cited_labels = _citation_labels(answer)
    evidence_ids, allowed_labels = _evidence_labels(evidence)
    missing_labels = [label for label in cited_labels if label not in allowed_labels]
    unused_evidence_ids = [evidence_id for evidence_id in evidence_ids if evidence_id not in cited_labels]
    valid_count = len(cited_labels) - len(missing_labels)

    findings = []
    for label in missing_labels:
        findings.append({"type": "missing_citation_label", "label": label})
    for evidence_id in unused_evidence_ids:
        findings.append({"type": "unused_evidence", "evidence_id": evidence_id})

    return {
        "cited_labels": cited_labels,
        "missing_labels": missing_labels,
        "unused_evidence_ids": unused_evidence_ids,
        "consistency_ratio": 0.0 if not cited_labels else round(valid_count / len(cited_labels), 4),
        "findings": findings,
    }


def _citation_labels(answer: Any) -> list[str]:
    seen: set[str] = set()
    labels = []
    for label in _CITATION_RE.findall(string(answer) or ""):
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _evidence_labels(evidence: Iterable[Any] | Any) -> tuple[list[str], set[str]]:
    if evidence is None:
        items = []
    elif isinstance(evidence, str):
        items = [evidence]
    else:
        try:
            items = list(evidence)
        except TypeError:
            items = [evidence]

    evidence_ids = []
    allowed: set[str] = set()
    for index, item in enumerate(items):
        primary = item if isinstance(item, str) else result_id(item, index)
        primary_text = string(primary)
        if primary_text is not None:
            evidence_ids.append(primary_text)
            allowed.add(primary_text)
        for key in _LABEL_KEYS:
            label = string(value(item, key))
            if label is not None:
                allowed.add(label)
    return evidence_ids, allowed
