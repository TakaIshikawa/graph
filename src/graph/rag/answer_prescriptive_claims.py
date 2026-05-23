"""Audit prescriptive answer claims against nearby citations and evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import MISSING, iter_strings, string, tokens, value

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_INLINE_CITATION_RE = re.compile(r"(\[\d+\]|\[[^\]]+\]\([^)]+\)|\([A-Za-z][^)]*,\s*\d{4}\))")
_PRESCRIPTIVE_RE = re.compile(r"\b(should|must|avoid|use|switch|recommend(?:s|ed|ing|ation)?|recommended)\b", re.I)
_SUPPORT_RE = re.compile(
    r"\b(should|must|avoid|use|switch|recommend(?:s|ed|ing|ation)?|recommended|best practice|guideline|advis(?:e|es|ed|ory)|required|prefer)\b",
    re.I,
)


def audit_answer_prescriptive_claims(
    answer: str,
    evidence_spans: Iterable[Any] = (),
    *,
    citation_spans: Iterable[Any] | None = None,
    proximity_chars: int = 180,
    min_overlap: float = 0.35,
) -> dict[str, Any]:
    """Return prescriptive answer claims whose nearby support is weak or missing.

    Evidence support is intentionally conservative: a claim is supported when a
    nearby citation or evidence span exists and at least one linked/nearby
    evidence span repeats recommendation language with meaningful term overlap.
    """
    if not isinstance(proximity_chars, int) or isinstance(proximity_chars, bool) or proximity_chars < 0:
        raise ValueError("proximity_chars must be a non-negative integer")
    if not isinstance(min_overlap, int | float) or isinstance(min_overlap, bool) or min_overlap < 0:
        raise ValueError("min_overlap must be a non-negative number")

    text = str(answer or "")
    evidence = _normalize_evidence(evidence_spans)
    citations = _normalize_citations(citation_spans, text)
    claims = []

    for sentence_index, start, end, sentence in _sentences(text):
        cues = sorted({match.group(1).casefold() for match in _PRESCRIPTIVE_RE.finditer(sentence)})
        if not cues:
            continue
        nearby_citations = _nearby(citations, start, end, proximity_chars)
        nearby_evidence = _matching_evidence(evidence, start, end, nearby_citations, proximity_chars)
        support = _support_status(sentence, nearby_citations, nearby_evidence, min_overlap)
        claims.append(
            {
                "sentence_index": sentence_index,
                "start": start,
                "end": end,
                "claim": sentence,
                "prescriptive_cues": cues,
                "support_status": support["status"],
                "reasons": support["reasons"],
                "citation_ids": [row["id"] for row in nearby_citations],
                "evidence_ids": [row["id"] for row in support["supporting_evidence"]],
                "support_scores": support["scores"],
            }
        )

    flagged = [claim for claim in claims if claim["support_status"] != "supported"]
    reason_counts = Counter(reason for claim in flagged for reason in claim["reasons"])
    warnings = []
    if not text.strip():
        warnings.append("no_answer")
    if flagged:
        warnings.append("unsupported_or_weak_prescriptive_claims")

    return {
        "claim_count": len(claims),
        "supported_claim_count": sum(1 for claim in claims if claim["support_status"] == "supported"),
        "weak_claim_count": sum(1 for claim in claims if claim["support_status"] == "weak"),
        "unsupported_claim_count": sum(1 for claim in claims if claim["support_status"] == "unsupported"),
        "claims": claims,
        "flagged_claims": flagged,
        "reason_counts": dict(sorted(reason_counts.items())),
        "warnings": warnings,
    }


def _sentences(text: str) -> list[tuple[int, int, int, str]]:
    rows = []
    for index, match in enumerate(_SENTENCE_RE.finditer(text)):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        rows.append((index, match.start() + leading, match.start() + trailing, stripped))
    return rows


def _normalize_citations(spans: Iterable[Any] | None, answer: str) -> list[dict[str, Any]]:
    if spans is None:
        return [
            {"id": match.group(0), "start": match.start(), "end": match.end(), "text": match.group(0), "refs": {match.group(0)}}
            for match in _INLINE_CITATION_RE.finditer(answer)
        ]
    rows = []
    for index, span in enumerate(spans or ()):
        start = _integer(value(span, "start"))
        end = _integer(value(span, "end"))
        text = _span_text(span)
        if start is None or end is None:
            found_at = answer.find(text) if text else -1
            if found_at >= 0:
                start = found_at
                end = found_at + len(text)
        if start is None or end is None:
            continue
        cid = _span_id(span, index, "citation")
        rows.append({"id": cid, "start": start, "end": end, "text": text, "refs": _refs(span, cid, text)})
    return rows


def _normalize_evidence(spans: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for index, span in enumerate(spans or ()):
        text = _span_text(span)
        if not text:
            continue
        eid = _span_id(span, index, "evidence")
        start = _integer(value(span, "start"))
        end = _integer(value(span, "end"))
        rows.append({"id": eid, "start": start, "end": end, "text": text, "refs": _refs(span, eid, text), "tokens": tokens(text)})
    return rows


def _span_text(span: Any) -> str:
    if isinstance(span, str):
        return " ".join(span.split())
    for key in ("text", "span", "quote", "content", "snippet"):
        text = string(value(span, key))
        if text is not None:
            return text
    return string(span) or ""


def _span_id(span: Any, index: int, prefix: str) -> str:
    for key in ("id", "span_id", "citation_id", "evidence_id", "result_id", "source_id"):
        text = string(value(span, key))
        if text is not None:
            return text
    return f"{prefix}-{index + 1}"


def _refs(span: Any, own_id: str, text: str) -> set[str]:
    refs = {own_id}
    if text:
        refs.add(text)
    for key in ("id", "span_id", "citation_id", "citation_ids", "evidence_id", "result_id", "source_id", "source_ids"):
        raw = value(span, key)
        if raw is MISSING:
            continue
        refs.update(item for item in iter_strings(raw) if item)
    return refs


def _nearby(rows: list[dict[str, Any]], start: int, end: int, distance: int) -> list[dict[str, Any]]:
    return [row for row in rows if _distance(start, end, row.get("start"), row.get("end")) <= distance]


def _matching_evidence(
    evidence: list[dict[str, Any]],
    start: int,
    end: int,
    citations: list[dict[str, Any]],
    distance: int,
) -> list[dict[str, Any]]:
    citation_refs = set().union(*(row["refs"] for row in citations)) if citations else set()
    matches = []
    for row in evidence:
        linked = bool(citation_refs & row["refs"])
        colocated = row["start"] is not None and row["end"] is not None and _distance(start, end, row["start"], row["end"]) <= distance
        if linked or colocated or (not citations and row["start"] is None and row["end"] is None):
            matches.append(row)
    return matches


def _support_status(
    claim: str,
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    min_overlap: float,
) -> dict[str, Any]:
    claim_tokens = tokens(_INLINE_CITATION_RE.sub("", claim))
    supporting = []
    scores = []
    for row in evidence:
        overlap = (claim_tokens & row["tokens"]) if claim_tokens else set()
        score = len(overlap) / len(claim_tokens) if claim_tokens else 0.0
        scores.append({"evidence_id": row["id"], "overlap_score": round(score, 4), "explicit_support": bool(_SUPPORT_RE.search(row["text"]))})
        if score >= min_overlap and _SUPPORT_RE.search(row["text"]):
            supporting.append(row)

    if supporting:
        return {"status": "supported", "reasons": [], "supporting_evidence": supporting, "scores": scores}
    if citations or evidence:
        reasons = []
        if citations and not evidence:
            reasons.append("nearby_citation_without_matching_evidence")
        if evidence and not any(_SUPPORT_RE.search(row["text"]) for row in evidence):
            reasons.append("evidence_not_prescriptive")
        if evidence and not any(score["overlap_score"] >= min_overlap for score in scores):
            reasons.append("low_evidence_overlap")
        return {"status": "weak", "reasons": reasons or ["weak_prescriptive_support"], "supporting_evidence": [], "scores": scores}
    return {"status": "unsupported", "reasons": ["no_nearby_citation_or_evidence"], "supporting_evidence": [], "scores": []}


def _distance(start: int, end: int, other_start: Any, other_end: Any) -> int:
    if other_start is None or other_end is None:
        return 10**9
    if other_end < start:
        return start - other_end
    if end < other_start:
        return other_start - end
    return 0


def _integer(value_: Any) -> int | None:
    if isinstance(value_, bool) or value_ is MISSING or value_ is None:
        return None
    try:
        return int(value_)
    except (TypeError, ValueError):
        return None
