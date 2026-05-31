"""Analyze source type coverage for retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, domain_for, result_id, string, value

_TYPES = ("docs", "academic", "news", "forum", "code", "dataset", "video", "audio", "unknown")
_CUES = {
    "academic": ("doi", "pubmed", "arxiv", "journal", "paper", "study"),
    "news": ("news", "reuters", "apnews", "nytimes", "article"),
    "forum": ("forum", "reddit", "stackoverflow", "discussion"),
    "code": ("github", "gitlab", "source code", "repository"),
    "dataset": ("dataset", "data catalog", "kaggle", "csv"),
    "video": ("youtube", "vimeo", "video"),
    "audio": ("podcast", "audio"),
    "docs": ("docs", "documentation", "manual", "guide"),
}


def analyze_result_source_type_coverage(results: Iterable[Any], expected_types: Iterable[str] | None = None) -> dict[str, Any]:
    rows = list(results or [])
    counts: Counter[str] = Counter({kind: 0 for kind in _TYPES})
    examples: dict[str, list[dict[str, str]]] = {kind: [] for kind in _TYPES}
    for index, result in enumerate(rows):
        kind = _classify(result)
        rid = result_id(result, index)
        counts[kind] += 1
        if len(examples[kind]) < 3 and rid not in {sample["id"] for sample in examples[kind]}:
            examples[kind].append({"id": rid, "title": string(value(result, "title")) or ""})
    covered = [kind for kind in _TYPES if counts[kind] > 0 and kind != "unknown"]
    expected = sorted({_normalize_type(kind) for kind in (expected_types or [])})
    return {
        "total_results": len(rows),
        "type_counts": dict(counts),
        "covered_types": covered,
        "missing_expected_types": [kind for kind in expected if counts[kind] == 0],
        "diversity_score": round(len(covered) / len([kind for kind in _TYPES if kind != "unknown"]), 4),
        "examples": {kind: examples[kind] for kind in _TYPES if examples[kind]},
    }


def _classify(result: Any) -> str:
    explicit = _normalize_type(string(value(result, "source_type")) or string(value(result, "type")) or "")
    if explicit != "unknown":
        return explicit
    text = " ".join(part for part in [domain_for(result) or "", string(value(result, "url")) or "", content_text(result)] if part).casefold()
    for kind, cues in _CUES.items():
        if any(cue in text for cue in cues):
            return kind
    return "unknown"


def _normalize_type(value_: str) -> str:
    text = value_.casefold().strip()
    aliases = {"documentation": "docs", "doc": "docs", "paper": "academic", "research": "academic", "software": "code", "data": "dataset"}
    return aliases.get(text, text if text in _TYPES else "unknown")
