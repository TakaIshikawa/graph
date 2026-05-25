"""Plan dataset artifact retrieval for RAG result sets."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id, string, value

_ARTIFACT_CUES: dict[str, tuple[re.Pattern[str], ...]] = {
    "raw_data": (
        re.compile(r"\b(?:raw\s+data|raw\s+dataset|source\s+data|microdata)\b", re.I),
    ),
    "csv": (
        re.compile(r"\b(?:csv|comma[-\s]separated|spreadsheet)\b", re.I),
    ),
    "data_dictionary": (
        re.compile(r"\b(?:data\s+dictionary|variable\s+dictionary|field\s+definitions?)\b", re.I),
    ),
    "codebook": (
        re.compile(r"\bcodebook\b", re.I),
    ),
    "schema": (
        re.compile(r"\b(?:schema|table\s+definition|column\s+definitions?)\b", re.I),
    ),
    "repository": (
        re.compile(r"\b(?:repository|repo|github|gitlab|archive|dataverse|osf)\b", re.I),
    ),
    "notebook": (
        re.compile(r"\b(?:notebook|jupyter|colab|r\s+markdown|rmd)\b", re.I),
    ),
    "supplement": (
        re.compile(r"\b(?:supplement|supplementary|supporting\s+information)\b", re.I),
    ),
    "appendix": (
        re.compile(r"\bappendix\b", re.I),
    ),
    "replication_package": (
        re.compile(r"\b(?:replication\s+package|replication\s+materials?|reproducibility\s+package)\b", re.I),
    ),
}
_EMPIRICAL_RE = re.compile(
    r"\b(?:dataset|empirical|statistical|regression|survey|sample|observations?|variables?|measurements?|study\s+data)\b",
    re.I,
)
_PRESENT_KEYS = (
    "artifact",
    "artifacts",
    "attachments",
    "files",
    "links",
    "resources",
    "supplements",
    "repository",
    "source_url",
    "url",
)


def plan_result_dataset_artifacts(query: str, results: Iterable[Any] = ()) -> dict[str, Any]:
    """Return required, present, and missing dataset artifacts for RAG retrieval."""
    query_text = _normalize_query(query)
    result_rows = [_result_artifacts(result, index) for index, result in enumerate(results or [])]
    present = sorted({artifact for row in result_rows for artifact in row["artifacts"]})
    required = _required_artifacts(query_text, result_rows)
    missing = [artifact for artifact in required if artifact not in present]
    return {
        "required_artifacts": required,
        "present_artifacts": present,
        "missing_artifacts": missing,
        "result_artifacts": result_rows,
        "retrieval_hints": [_hint(artifact) for artifact in missing],
        "warnings": [f"missing_{artifact}" for artifact in missing],
    }


def _required_artifacts(query_text: str, result_rows: list[dict[str, Any]]) -> list[str]:
    required = set(_artifacts_in_text(query_text))
    combined_result_text = " ".join(row["evidence_text"] for row in result_rows)
    if _EMPIRICAL_RE.search(query_text) or _EMPIRICAL_RE.search(combined_result_text):
        required.update({"raw_data", "data_dictionary"})
    return sorted(required)


def _result_artifacts(result: Any, index: int) -> dict[str, Any]:
    meta = metadata(result)
    candidate_texts = [content_text(result), *iter_strings(meta)]
    for key in _PRESENT_KEYS:
        found = value(result, key)
        if isinstance(found, str):
            candidate_texts.append(found)
        elif isinstance(found, Mapping):
            candidate_texts.extend(iter_strings(found))
        else:
            candidate_texts.extend(iter_strings(found))
    evidence_text = " ".join(text for text in candidate_texts if text)
    return {
        "result_id": result_id(result, index),
        "artifacts": _artifacts_in_text(evidence_text),
        "evidence_text": evidence_text,
    }


def _artifacts_in_text(text: str) -> list[str]:
    normalized = " ".join(str(text or "").split())
    return [
        artifact
        for artifact, patterns in _ARTIFACT_CUES.items()
        if any(pattern.search(normalized) for pattern in patterns)
    ]


def _hint(artifact: str) -> str:
    return f"retrieve_{artifact}_from_dataset_repository_or_primary_source"


def _normalize_query(query: str) -> str:
    text = string(query)
    if text is None:
        raise ValueError("query must be a non-empty string")
    return text
