"""Detect sampling-bias limitations in evidence records."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id

_BIAS_TYPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("convenience_sample", re.compile(r"\bconvenience sample\b", re.I)),
    ("self_selection", re.compile(r"\bself[- ]selected\b|\bvolunteers?\b", re.I)),
    ("online_survey", re.compile(r"\bonline survey\b|\bweb survey\b", re.I)),
    ("small_sample", re.compile(r"\bsmall sample\b|\bn\s*[=:]\s*(?:[1-9]|[1-4]\d)\b", re.I)),
    ("single_site", re.compile(r"\bsingle[- ]site\b|\bone site\b", re.I)),
    ("attrition", re.compile(r"\battrition\b|\bdropout rate\b", re.I)),
    ("nonresponse", re.compile(r"\bnonresponse\b|\bnon-response\b", re.I)),
    ("selection_bias", re.compile(r"\bselection bias\b", re.I)),
    ("not_representative", re.compile(r"\bnot representative\b|\bnon[- ]representative\b", re.I)),
)


def detect_evidence_sampling_bias(evidence: Iterable[Any]) -> dict[str, Any]:
    """Return sampling-bias warnings and evidence ids by bias type."""
    biased: list[dict[str, Any]] = []
    ids_by_type: dict[str, list[str]] = {label: [] for label, _ in _BIAS_TYPES}
    for index, record in enumerate(evidence or []):
        evidence_id = result_id(record, index)
        text = _record_text(record)
        labels = [label for label, pattern in _BIAS_TYPES if pattern.search(text)]
        if not labels:
            continue
        biased.append({"evidence_id": evidence_id, "bias_types": labels})
        for label in labels:
            ids_by_type[label].append(evidence_id)
    ids_by_type = {label: ids for label, ids in ids_by_type.items() if ids}
    return {
        "biased_evidence": biased,
        "bias_types": list(ids_by_type),
        "evidence_ids_by_bias_type": ids_by_type,
        "warnings": ["sampling_bias_cues_detected"] if biased else [],
        "confidence": 0.8 if biased else 0.0,
    }


def _record_text(record: Any) -> str:
    return " ".join([content_text(record), " ".join(iter_strings(metadata(record)))])
