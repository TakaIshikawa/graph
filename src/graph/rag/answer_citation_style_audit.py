"""Audit citation styles used in an answer."""

from __future__ import annotations

import re
from typing import Any

_STYLE_PATTERNS = {
    "numeric_bracket": re.compile(r"\[(?:\d+(?:\s*,\s*\d+)*)]"),
    "markdown_link": re.compile(r"\[[^\]\n]+]\(https?://[^)\s]+[^)]*\)"),
    "author_year": re.compile(r"\([A-Z][A-Za-z-]+(?:\s+et al\.)?,\s*(?:19|20)\d{2}\)"),
    "footnote": re.compile(r"\[\^[^\]\n]+]"),
    "bare_url": re.compile(r"https?://[^\s)]+"),
}
_SENTENCE_RE = re.compile(r"[^.!?\n][^.!?\n]*(?:[.!?]|$)")


def audit_answer_citation_style(answer: str) -> dict[str, Any]:
    text = str(answer or "")
    counts = {style: len(pattern.findall(text)) for style, pattern in _STYLE_PATTERNS.items()}
    material = [style for style, count in counts.items() if count > 0]
    dominant = sorted(material, key=lambda style: (-counts[style], style))[0] if material else "none"
    uncited = sum(1 for sentence in _sentences(text) if _factual(sentence) and not _has_citation(sentence))
    cited_total = sum(counts.values())
    total_claims = cited_total + uncited
    score = round(cited_total / total_claims, 4) if total_claims else 1.0
    return {"style_counts": counts, "dominant_style": dominant, "mixed_style_warning": len(material) > 1, "uncited_sentence_count": uncited, "citation_style_score": score}


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip() and not match.group(0).strip().startswith("#")]


def _has_citation(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in _STYLE_PATTERNS.values())


def _factual(sentence: str) -> bool:
    lowered = sentence.casefold()
    return len(sentence.split()) >= 5 and not lowered.startswith(("note:", "maybe", "perhaps"))
