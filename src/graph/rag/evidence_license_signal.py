"""Classify license signals in RAG evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from graph.rag._analysis_utils import content_text, metadata, result_id

_LICENSES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("CC-BY", "open", re.compile(r"\bcc[-\s]?by\b|creative\s+commons\s+attribution", re.I)),
    ("MIT", "permissive", re.compile(r"\bmit\s+license\b", re.I)),
    ("Apache-2.0", "permissive", re.compile(r"\bapache(?:\s+license)?\s*2(?:\.0)?\b", re.I)),
    ("GPL", "copyleft", re.compile(r"\bgpl\b|gnu\s+general\s+public\s+license", re.I)),
    ("public domain", "open", re.compile(r"\bpublic\s+domain\b|cc0\b", re.I)),
    ("all rights reserved", "restrictive", re.compile(r"\ball\s+rights\s+reserved\b", re.I)),
    ("copyright", "restrictive", re.compile(r"\bcopyright\b", re.I)),
    ("terms of service", "restrictive", re.compile(r"\bterms\s+of\s+service\b|\btos\b", re.I)),
)


def analyze_evidence_license_signal(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    counts: Counter[str] = Counter()
    for index, result in enumerate(results):
        text = f"{content_text(result)} {' '.join(str(value) for value in metadata(result).values())}"
        license_name, classification = _classify(text)
        counts[classification] += 1
        rows.append({"id": result_id(result, index), "index": index, "license": license_name, "classification": classification})
    return {"result_count": len(results), "license_counts": dict(sorted(counts.items())), "results": rows}


def _classify(text: str) -> tuple[str, str]:
    for name, classification, pattern in _LICENSES:
        if pattern.search(text):
            return name, classification
    return "unknown", "unknown"
