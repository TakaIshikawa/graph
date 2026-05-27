"""Analyze stakeholder coverage in RAG results."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, metadata, result_id

_DEFAULT = ("user", "customer", "employee", "admin", "regulator", "vendor", "patient", "student", "investor")


def analyze_result_stakeholder_coverage(results: Iterable[Any], stakeholders: Iterable[str] | None = None) -> dict[str, Any]:
    expected = [s.casefold() for s in (stakeholders or _DEFAULT)]
    covered = set()
    per_result = []
    for index, result in enumerate(results):
        haystack = (content_text(result) + " " + " ".join(str(v) for v in metadata(result).values())).casefold()
        matches = [stakeholder for stakeholder in expected if re.search(rf"\b{re.escape(stakeholder)}s?\b", haystack)]
        covered.update(matches)
        per_result.append({"id": result_id(result, index), "matches": matches})
    return {"covered": sorted(covered), "missing": [s for s in expected if s not in covered], "per_result_matches": per_result}
