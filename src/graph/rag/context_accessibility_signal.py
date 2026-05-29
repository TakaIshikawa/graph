"""Analyze accessibility signals in retrieved context items."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from graph.rag._analysis_utils import content_text, metadata, result_id, value

_POSITIVE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("alt_text", re.compile(r"\balt\s+text\b|alternative\s+text|aria-label", re.I)),
    ("captions", re.compile(r"\bcaption(?:s|ed)?\b|subtitles?", re.I)),
    ("transcript", re.compile(r"\btranscript\b", re.I)),
    ("headings", re.compile(r"\bheadings?\b|<h[1-6]\b", re.I)),
    ("table_headers", re.compile(r"\btable\s+headers?\b|<th\b", re.I)),
)


def analyze_context_accessibility_signals(context_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    missing = 0
    for index, item in enumerate(context_items):
        text = f"{content_text(item)} {' '.join(str(v) for v in metadata(item).values())}"
        signals = [name for name, pattern in _POSITIVE if pattern.search(text)]
        content_type = str(value(item, "content_type") or value(item, "format") or "").casefold()
        is_media_only = content_type in {"image", "audio"} or bool(re.search(r"\b(image|audio)[-_ ]only\b", text, re.I))
        missing_accessibility = is_media_only and not any(signal in signals for signal in ("alt_text", "captions", "transcript"))
        if missing_accessibility:
            missing += 1
        rows.append({"id": result_id(item, index), "index": index, "signals": signals, "missing_accessibility": missing_accessibility})
    return {"item_count": len(context_items), "missing_accessibility_count": missing, "items": rows}
