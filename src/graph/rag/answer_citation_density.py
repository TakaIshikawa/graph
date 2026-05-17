"""Estimate citation density in draft RAG answers."""

from __future__ import annotations

import re
from typing import Any

_BRACKET_CITATION_RE = re.compile(r"\[(?:\d+(?:\s*,\s*\d+)*|\d+\s*-\s*\d+)\]")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


def estimate_answer_citation_density(
    answer: str,
    *,
    min_citations_per_paragraph: int = 1,
    max_citations_per_paragraph: int = 4,
) -> dict[str, Any]:
    """Estimate paragraph-level citation density for a draft answer."""
    if min_citations_per_paragraph < 0 or max_citations_per_paragraph < min_citations_per_paragraph:
        raise ValueError("citation thresholds must be non-negative and ordered")

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", answer or "") if paragraph.strip()]
    counts = [_citation_count(paragraph) for paragraph in paragraphs]
    uncited = [index for index, count in enumerate(counts) if count < min_citations_per_paragraph]
    over_cited = [index for index, count in enumerate(counts) if count > max_citations_per_paragraph]
    acceptable = len(paragraphs) - len(set(uncited).union(over_cited))
    return {
        "paragraph_count": len(paragraphs),
        "citation_count": sum(counts),
        "uncited_paragraph_indexes": uncited,
        "over_cited_paragraph_indexes": over_cited,
        "density_score": round(acceptable / len(paragraphs), 3) if paragraphs else 0.0,
    }


def _citation_count(paragraph: str) -> int:
    return len(_BRACKET_CITATION_RE.findall(paragraph)) + len(_MARKDOWN_LINK_RE.findall(paragraph))
