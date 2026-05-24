"""Audit uncertainty quantification in RAG answers."""

from __future__ import annotations

import re
from typing import Any

_RANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|[A-Za-z]+)?\s*(?:-|to|through)\s*\d+(?:\.\d+)?\s*(?:%|[A-Za-z]+)?\b", re.I)
_CI_RE = re.compile(r"\b(?:confidence interval|CI)\b[^.;:\n]*\b\d+(?:\.\d+)?\s*(?:%|to|-)", re.I)
_PROBABILITY_RE = re.compile(r"\b(?:probability|chance|likelihood|confidence)\s+(?:of\s+)?\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?%\s+(?:chance|probability|confidence)\b", re.I)
_QUALITATIVE_RE = re.compile(r"\b(?:uncertain|roughly|approximately|about|around|likely|unlikely|low confidence|medium confidence|high confidence)\b", re.I)
_PREDICTIVE_QUERY_RE = re.compile(r"\b(?:estimate|forecast|predict|projection|likely|chance|risk|uncertain)\b", re.I)


def audit_answer_uncertainty_quantification(query: str, answer: str) -> dict[str, Any]:
    """Return uncertainty markers and whether predictive answers quantify uncertainty."""
    normalized_query = " ".join(str(query or "").split())
    normalized_answer = " ".join(str(answer or "").split())
    markers = {
        "numeric_ranges": _matches(_RANGE_RE, normalized_answer),
        "confidence_intervals": _matches(_CI_RE, normalized_answer),
        "probabilities": _matches(_PROBABILITY_RE, normalized_answer),
        "qualitative_uncertainty": _matches(_QUALITATIVE_RE, normalized_answer),
    }
    has_quantification = bool(markers["numeric_ranges"] or markers["confidence_intervals"] or markers["probabilities"])
    has_uncertainty = has_quantification or bool(markers["qualitative_uncertainty"])
    predictive = bool(_PREDICTIVE_QUERY_RE.search(normalized_query))
    return {
        "query_needs_uncertainty": predictive,
        "has_uncertainty_marker": has_uncertainty,
        "has_quantified_uncertainty": has_quantification,
        "markers": markers,
        "missing_quantification": predictive and not has_quantification,
    }


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return [match.group(0).strip() for match in pattern.finditer(text)]
