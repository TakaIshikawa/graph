"""Analyze content-format diversity in RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.rag._analysis_utils import result_id, string, value

_FORMATS = ("article", "PDF", "video", "audio", "code", "dataset", "forum", "docs", "unknown")


def analyze_result_format_diversity(results: Iterable[Any]) -> dict[str, Any]:
    items = list(results or [])
    rows = [{"result_id": result_id(item, index), "format": _format(item)} for index, item in enumerate(items)]
    counts = Counter(row["format"] for row in rows)
    dominant = max(_FORMATS, key=lambda label: (counts[label], -_FORMATS.index(label))) if items else "unknown"
    diversity = round((len(counts) - 1) / (len(_FORMATS) - 1), 4) if items else 0.0
    return {
        "total_results": len(items),
        "format_counts": dict(sorted(counts.items())),
        "diversity_score": diversity,
        "dominant_format": dominant,
        "samples": sorted(rows, key=lambda row: (row["format"], row["result_id"]))[:5],
    }


def _format(result: Any) -> str:
    for key in ("content_type", "mime_type", "format", "source_type", "type"):
        label = _label(string(value(result, key)) or "")
        if label != "unknown":
            return label
    url = string(value(result, "url")) or ""
    title = string(value(result, "title")) or ""
    return _label(f"{url} {title}")


def _label(text: str) -> str:
    value_ = text.casefold()
    suffix = Path(urlparse(value_.split()[0] if value_.split() else "").path).suffix
    if "pdf" in value_ or suffix == ".pdf":
        return "PDF"
    if "video" in value_ or "youtube.com" in value_ or "vimeo.com" in value_:
        return "video"
    if "audio" in value_ or suffix in {".mp3", ".wav", ".m4a"}:
        return "audio"
    if "github.com" in value_ or "code" in value_ or suffix in {".py", ".js", ".ts"}:
        return "code"
    if "dataset" in value_ or "csv" in value_ or "json" in value_ or "kaggle.com" in value_:
        return "dataset"
    if "forum" in value_ or "reddit.com" in value_ or "stackoverflow.com" in value_:
        return "forum"
    if "docs" in value_ or "documentation" in value_:
        return "docs"
    if "article" in value_ or "text/html" in value_ or value_.startswith("http"):
        return "article"
    return "unknown"
