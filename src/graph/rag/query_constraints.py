"""Extract explicit retrieval constraints from natural-language RAG queries."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import MISSING, string, value

_TOKEN_RE = re.compile(r"""[^\s"'`]+:(?:"[^"]*"|'[^']*'|`[^`]*`|[^\s]+)|[+-](?:"[^"]*"|'[^']*'|`[^`]*`|[^\s]+)|"[^"]*"|'[^']*'|`[^`]*`|\S+""")
_FILTER_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*):(?P<value>.+)$")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{4}|today|yesterday|last\s+\d+\s+(?:day|days|week|weeks|month|months|year|years))\b", re.I)
_RELATIVE_RE = re.compile(r"\b(?:after|before|since|until|from|through|between)\s+(\d{4}(?:-\d{2}-\d{2})?)", re.I)


def extract_query_constraints(query: Any) -> dict[str, Any]:
    """Return deterministic explicit retrieval constraints found in ``query``."""
    text = _query_text(query)
    quoted_phrases: list[str] = []
    required_terms: list[str] = []
    excluded_terms: list[str] = []
    site_filters: list[str] = []
    domain_filters: list[str] = []
    filetypes: list[str] = []
    content_types: list[str] = []
    date_constraints: list[str] = []

    for token in _TOKEN_RE.findall(text):
        cleaned = _clean(token)
        if not cleaned:
            continue
        if _is_quoted(token):
            _append(quoted_phrases, cleaned)
            continue
        if token.startswith("+") and len(token) > 1:
            _append(required_terms, _clean(token[1:]).casefold())
            continue
        if token.startswith("-") and len(token) > 1 and not _looks_negative_number(token):
            _append(excluded_terms, _clean(token[1:]).casefold())
            continue

        match = _FILTER_RE.match(token)
        if not match:
            continue
        key = match.group("key").casefold()
        value_text = _clean(match.group("value"))
        if not value_text:
            continue
        if key == "site":
            _append(site_filters, value_text.casefold())
        elif key in {"domain", "host"}:
            _append(domain_filters, value_text.casefold())
        elif key in {"filetype", "ext"}:
            _append(filetypes, value_text.casefold().lstrip("."))
        elif key in {"type", "contenttype", "content_type"}:
            _append(content_types, value_text.casefold())
        elif key in {"after", "before", "since", "until", "date", "year"}:
            _append(date_constraints, f"{key}:{value_text.casefold()}")

    for match in _DATE_RE.finditer(text):
        _append(date_constraints, " ".join(match.group(0).casefold().split()))
    for match in _RELATIVE_RE.finditer(text):
        _append(date_constraints, " ".join(match.group(0).casefold().split()))

    return {
        "query": text,
        "quoted_phrases": quoted_phrases,
        "required_terms": required_terms,
        "excluded_terms": excluded_terms,
        "site_filters": site_filters,
        "domain_filters": domain_filters,
        "filetypes": filetypes,
        "content_types": content_types,
        "date_constraints": date_constraints,
        "has_constraints": any(
            (
                quoted_phrases,
                required_terms,
                excluded_terms,
                site_filters,
                domain_filters,
                filetypes,
                content_types,
                date_constraints,
            )
        ),
    }


def _query_text(query: Any) -> str:
    for key in ("query", "text", "content", "question"):
        item = value(query, key)
        if item is not MISSING:
            text = string(item)
            if text is not None:
                return text
    return string(query) or ""


def _clean(value_: str) -> str:
    text = " ".join(value_.strip().split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        text = " ".join(text[1:-1].strip().split())
    return text


def _is_quoted(token: str) -> bool:
    stripped = token.strip()
    return len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"', "`"}


def _looks_negative_number(token: str) -> bool:
    return bool(re.match(r"^-\d", token))


def _append(values: list[str], value_: str) -> None:
    if value_ and value_ not in values:
        values.append(value_)
