"""Audit whether answer source attribution is balanced across evidence sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from graph.rag._record_text import first


def audit_answer_source_attribution_balance(answer: str, evidence: Iterable[Any] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    sources = sorted({source for item in evidence or () if (source := _source_name(item))})
    answer_text = str(answer or "").lower()
    mentions = {source: len(re.findall(rf"\b{re.escape(source.lower())}\b", answer_text)) for source in sources}
    mentions = {source: count for source, count in mentions.items() if count}
    dominant = max(mentions.items(), key=lambda item: (item[1], item[0]))[0] if mentions else None
    dominant_count = mentions.get(dominant, 0) if dominant else 0
    total_mentions = sum(mentions.values())
    return {
        "evidence_source_count": len(sources),
        "answer_source_mentions": mentions,
        "dominant_answer_source": dominant,
        "single_source_overattribution": len(sources) > 1 and dominant_count >= 2 and dominant_count == total_mentions,
        "samples": sources[:sample_limit],
    }


def _source_name(item: Any) -> str | None:
    for key in ("domain", "url", "source", "source_type", "author", "title"):
        value = first(item, (key,))
        if not value:
            continue
        text = str(value).strip()
        if key == "url" or "://" in text:
            parsed = urlparse(text if "://" in text else f"//{text}")
            text = parsed.netloc or parsed.path.split("/")[0]
        text = text.lower().removeprefix("www.")
        return text
    return None
