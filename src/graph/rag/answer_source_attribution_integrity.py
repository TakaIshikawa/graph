"""Audit answer citations against retrieved source records."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, string, value

_BRACKET_RE = re.compile(r"\[([A-Za-z][\w:-]*)\]")
_SOURCE_LABEL_RE = re.compile(r"\bsource(?:s)?\s*:\s*([^.;\n\]]+)", re.I)
_URL_RE = re.compile(r"https?://[^\s)\]]+")


def audit_answer_source_attribution_integrity(answer: str, retrieved_results: Iterable[Any]) -> dict[str, Any]:
    """Flag answer citations or source labels that do not match retrieved results."""
    results = list(retrieved_results or [])
    by_id = {result_id(result, index): result for index, result in enumerate(results)}
    title_to_urls: dict[str, set[str]] = defaultdict(set)
    title_to_ids: dict[str, set[str]] = defaultdict(set)
    for index, result in enumerate(results):
        title = _norm(value(result, "title"))
        url = string(value(result, "url") or value(result, "source_url") or value(result, "canonical_url"))
        if title:
            title_to_ids[title].add(result_id(result, index))
            if url:
                title_to_urls[title].add(url.rstrip(".,)"))

    issues = []
    answer_text = str(answer or "")
    for citation_id in sorted(set(_BRACKET_RE.findall(answer_text))):
        if citation_id not in by_id:
            issues.append({"type": "unknown_citation_id", "label": citation_id, "severity": "high"})

    known_labels = {label.casefold() for label in by_id}
    known_labels.update(title_to_ids)
    for result in results:
        for key in ("name", "source", "source_name"):
            label = _norm(value(result, key))
            if label:
                known_labels.add(label)
    for label in sorted({_clean_label(match.group(1)) for match in _SOURCE_LABEL_RE.finditer(answer_text)}):
        if label and label.casefold() not in known_labels:
            issues.append({"type": "unknown_source_label", "label": label, "severity": "high"})

    for title, ids in sorted(title_to_ids.items()):
        if len(ids) > 1:
            issues.append({"type": "duplicate_conflicting_label", "label": title, "result_ids": sorted(ids), "severity": "medium"})

    for title, urls in sorted(title_to_urls.items()):
        if not title or title not in _norm(answer_text):
            continue
        mentioned_urls = {url.rstrip(".,)") for url in _URL_RE.findall(answer_text)}
        wrong_urls = sorted(mentioned_urls - urls)
        if mentioned_urls and wrong_urls:
            issues.append({"type": "title_url_mismatch", "label": title, "expected_urls": sorted(urls), "observed_urls": wrong_urls, "severity": "high"})

    return {"issues": issues, "retrieved_result_count": len(results)}


def _norm(value_: object) -> str:
    return (string(value_) or "").casefold()


def _clean_label(value_: object) -> str:
    return (string(value_) or "").strip(" []()")
