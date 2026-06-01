"""Audit answers for prerequisites present in evidence but absent from the answer."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string

_PREREQ_RE = re.compile(
    r"\b(?:prerequisite|dependency|requirement|permission|setup step)s?(?:\s+required)?\s*(?:is|are|:|-)?\s*([^.;\n]+)",
    re.I,
)


def audit_answer_missing_prerequisites(answer: str, evidence: Iterable[Any]) -> list[dict[str, Any]]:
    answer_text = str(answer or "").casefold()
    rows = []
    seen: set[str] = set()
    for index, item in enumerate(evidence or []):
        for phrase in _phrases(_item_text(item)):
            key = phrase.casefold()
            if key in seen or key in answer_text:
                continue
            seen.add(key)
            rows.append({"prerequisite_phrase": phrase, "source_id": result_id(item, index), "severity": "medium"})
    return sorted(rows, key=lambda row: (row["prerequisite_phrase"].casefold(), row["source_id"]))


def _phrases(text: str) -> list[str]:
    return [match.group(1).strip(" .,:;") for match in _PREREQ_RE.finditer(text) if match.group(1).strip(" .,:;")]


def _item_text(item: Any) -> str:
    return string(item) if isinstance(item, str) else content_text(item)
