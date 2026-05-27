"""Audit whether RAG results include dataset-bearing evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id, string, value

_DATASET_CUES: dict[str, tuple[re.Pattern[str], ...]] = {
    "dataset": (re.compile(r"\bdata(?:set|sets)?\b", re.I),),
    "data_repository": (
        re.compile(r"\b(?:data\s+repository|data\s+archive|dataverse|osf|repository)\b", re.I),
    ),
    "supplementary_data": (
        re.compile(r"\b(?:supplementary|supplemental|supporting)\s+(?:data|information|materials?)\b", re.I),
    ),
    "benchmark": (re.compile(r"\bbenchmark(?:\s+dataset|\s+suite|\s+corpus)?\b", re.I),),
    "registry": (re.compile(r"\b(?:registry|registered\s+dataset)\b", re.I),),
    "zenodo": (re.compile(r"\bzenodo\b", re.I),),
    "figshare": (re.compile(r"\bfigshare\b", re.I),),
    "dryad": (re.compile(r"\bdryad\b", re.I),),
    "kaggle": (re.compile(r"\bkaggle\b", re.I),),
    "huggingface_dataset": (
        re.compile(r"\b(?:hugging\s*face|hf)\s+dataset\b", re.I),
        re.compile(r"\bhuggingface\.co/datasets\b", re.I),
    ),
    "github_release": (
        re.compile(r"\bgithub\s+release\b", re.I),
        re.compile(r"\bgithub\.com/[^/\s]+/[^/\s]+/releases\b", re.I),
    ),
}
_TEXT_KEYS = ("title", "snippet", "content", "text", "summary", "url", "source_url", "canonical_url", "link", "uri")


def audit_result_dataset_coverage(results: Iterable[Any]) -> dict[str, Any]:
    """Return dataset-bearing result coverage for a retrieval result list."""
    result_list = list(results or [])
    dataset_sources = []
    missing_dataset_result_ids = []

    for index, result in enumerate(result_list):
        current_id = result_id(result, index)
        cues = _dataset_cues(_result_text(result))
        if cues:
            dataset_sources.append({"result_id": current_id, "dataset_cues": cues})
        else:
            missing_dataset_result_ids.append(current_id)

    dataset_result_count = len(dataset_sources)
    total = len(result_list)
    return {
        "has_dataset_coverage": bool(dataset_sources),
        "dataset_result_count": dataset_result_count,
        "coverage_ratio": 0.0 if not total else round(dataset_result_count / total, 4),
        "dataset_sources": dataset_sources,
        "missing_dataset_result_ids": missing_dataset_result_ids,
    }


def _result_text(result: Any) -> str:
    parts = [content_text(result)]
    for key in _TEXT_KEYS:
        text = string(value(result, key))
        if text:
            parts.append(text)
    meta = metadata(result)
    parts.extend(iter_strings(meta))
    parts.extend(str(key) for key in meta)
    if isinstance(result, Mapping):
        parts.extend(iter_strings({key: item for key, item in result.items() if key not in {"metadata"}}))
    return " ".join(part for part in parts if part)


def _dataset_cues(text: str) -> list[str]:
    normalized = " ".join(str(text or "").split())
    return [cue for cue, patterns in _DATASET_CUES.items() if any(pattern.search(normalized) for pattern in patterns)]
