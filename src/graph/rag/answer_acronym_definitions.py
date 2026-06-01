"""Audit acronym definitions in generated answers."""

from __future__ import annotations

import re
from typing import Any

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:s)?\b")


def audit_answer_acronym_definitions(answer: str) -> list[dict[str, Any]]:
    text = str(answer or "")
    rows = []
    seen: set[str] = set()
    for match in _ACRONYM_RE.finditer(text):
        acronym = match.group(0).removesuffix("s")
        if acronym in seen:
            continue
        seen.add(acronym)
        defined = _defined(text, acronym, match.start())
        rows.append({"acronym": acronym, "defined_on_first_use": defined, "severity": "none" if defined else "medium"})
    return sorted(rows, key=lambda row: row["acronym"])


def _defined(text: str, acronym: str, start: int) -> bool:
    prefix = text[max(0, start - 120) : start]
    suffix = text[start + len(acronym) : start + len(acronym) + 120]
    long_name_before = re.search(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,8})\s*\(\s*$", prefix)
    expansion_after = re.match(r"\s*\(\s*[A-Z][A-Za-z]+(?:\s+[A-Za-z]+){1,8}\s*\)", suffix)
    return bool(long_name_before or expansion_after)
