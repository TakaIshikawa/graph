"""Detect reproducibility evidence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REPRO_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("reproducible", re.compile(r"\breproducib(?:le|ility)\b", re.I)),
    ("replication", re.compile(r"\breplicat(?:e|ed|ion)\b", re.I)),
    ("preregistered", re.compile(r"\bpre[- ]?registered\b|\bpreregistration\b", re.I)),
    ("open_data", re.compile(r"\bopen data\b", re.I)),
)
_ARTIFACTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code", re.compile(r"\bcode available\b|\bsource code\b|\bgithub\b", re.I)),
    ("dataset", re.compile(r"\bdataset available\b|\bdata available\b|\braw data\b|\bopen data\b", re.I)),
    ("notebook", re.compile(r"\bnotebook\b|\bjupyter\b|\bcolab\b", re.I)),
    ("protocol", re.compile(r"\bprotocol\b|\bstudy protocol\b", re.I)),
)
_METHOD_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("methods_appendix", re.compile(r"\bmethods appendix\b|\bsupplementary methods\b", re.I)),
    ("random_seed", re.compile(r"\brandom seed\b|\bseed\b", re.I)),
    ("transparent_methods", re.compile(r"\btransparent methods\b|\bmethods detail\b", re.I)),
)


def detect_query_reproducibility_requirement(query: str) -> dict[str, Any]:
    """Return reproducibility cues and evidence artifact requirements."""
    normalized = _normalize_query(query)
    cues = [label for label, pattern in _REPRO_CUES if pattern.search(normalized)]
    artifacts = [label for label, pattern in _ARTIFACTS if pattern.search(normalized)]
    methods = [label for label, pattern in _METHOD_CUES if pattern.search(normalized)]
    requires = bool(cues or artifacts or methods)
    recommendations = []
    if artifacts:
        recommendations.append("prioritize_sources_with_required_reproducibility_artifacts")
    if cues or methods:
        recommendations.append("retrieve_methods_and_replication_materials_before_summarizing")
    return {
        "requires_reproducibility_evidence": requires,
        "reproducibility_cues": cues,
        "artifact_requirements": artifacts,
        "method_transparency_cues": methods,
        "recommendations": recommendations,
        "confidence": 0.85 if artifacts else (0.7 if cues or methods else 0.0),
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())
