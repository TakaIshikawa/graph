"""Audit whether numeric answer claims appear in retrieved evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_RANGE_RE = re.compile(r"(?<!\w)([$€£]?\s?\d+(?:,\d{3})*(?:\.\d+)?%?)\s*(?:-|to|through)\s*([$€£]?\s?\d+(?:,\d{3})*(?:\.\d+)?%?)(?!\w)", re.I)
_NUMBER_RE = re.compile(r"[$€£]?\s?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent)?", re.I)


def audit_evidence_numeric_support(answer: str, results: Iterable[Any]) -> dict[str, Any]:
    """Extract answer numbers and check deterministic evidence appearances."""
    answer_text = str(answer or "")
    evidence = [{"result_id": result_id(result, index), "text": content_text(result)} for index, result in enumerate(results or [])]
    numbers = _answer_numbers(answer_text)
    supported = []
    unsupported = []
    matches = []
    for number in numbers:
        variants = _variants(number)
        hit_ids = [row["result_id"] for row in evidence if any(variant in _canonical(row["text"]) for variant in variants)]
        number_row = {**number, "matched_result_ids": hit_ids}
        if hit_ids:
            supported.append(number_row)
            for result_id_ in hit_ids:
                matches.append({"matched_text": number["matched_text"], "result_id": result_id_})
        else:
            number_row["reasons"] = ["missing_numeric_evidence"]
            unsupported.append(number_row)

    reason_counts = Counter(reason for row in unsupported for reason in row["reasons"])
    warnings = []
    if not answer_text.strip():
        warnings.append("no_answer")
    if not evidence:
        warnings.append("no_results")
    if unsupported:
        warnings.append("unsupported_numbers")
    return {
        "answer_numbers": numbers,
        "supported_numbers": supported,
        "unsupported_numbers": unsupported,
        "result_matches": matches,
        "reason_counts": dict(sorted(reason_counts.items())),
        "warnings": warnings,
    }


def _answer_numbers(text: str) -> list[dict[str, Any]]:
    rows = []
    occupied: list[tuple[int, int]] = []
    for match in _RANGE_RE.finditer(text):
        if not _compatible_range(match.group(1).strip(), match.group(2).strip()):
            continue
        occupied.append(match.span())
        rows.append({"matched_text": match.group(0), "number_type": "range", "values": [match.group(1).strip(), match.group(2).strip()]})
    for match in _NUMBER_RE.finditer(text):
        if any(match.start() < end and start < match.end() for start, end in occupied):
            continue
        matched = " ".join(match.group(0).split())
        rows.append({"matched_text": matched, "number_type": _number_type(matched), "values": [matched]})
    return rows


def _number_type(text: str) -> str:
    if text.startswith(("$", "€", "£")):
        return "currency"
    if "%" in text or "percent" in text.casefold():
        return "percentage"
    if "." in text:
        return "decimal"
    return "integer"


def _compatible_range(left: str, right: str) -> bool:
    left_type = _number_type(left)
    right_type = _number_type(right)
    if left_type == right_type:
        return True
    return {left_type, right_type} == {"currency", "integer"}


def _variants(number: dict[str, Any]) -> set[str]:
    variants = {_canonical(number["matched_text"])}
    for value in number["values"]:
        canonical = _canonical(value)
        variants.add(canonical)
        if canonical.endswith("%"):
            variants.add(canonical[:-1] + " percent")
        if canonical.endswith(" percent"):
            variants.add(canonical.removesuffix(" percent") + "%")
    return {variant for variant in variants if variant}


def _canonical(text: str) -> str:
    return " ".join(str(text).casefold().replace(",", "").split())
