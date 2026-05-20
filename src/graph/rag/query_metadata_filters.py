"""Extract lightweight metadata filter hints from RAG queries."""

from __future__ import annotations

import re
from typing import Any

_FILTER_KEYS = ("source", "tag", "author", "project", "account")
_TOKEN_RE = re.compile(r"""[^\s"'`]+:(?:"[^"]*"|'[^']*'|`[^`]*`|[^\s]+)|"[^"]*"|'[^']*'|`[^`]*`|\S+""")
_FILTER_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*):(?P<value>.*)$")


def extract_query_metadata_filter_hints(query: str | None) -> dict[str, Any]:
    """Return deterministic metadata filter hints and leftover query terms.

    Recognized filters are lexical hints in ``key:value`` form for ``source``,
    ``tag``, ``author``, ``project``, and ``account``. Quoted values are
    unwrapped and preserved, while malformed or unsupported filter-looking
    tokens are reported under ``ignored_tokens`` instead of raising.
    """
    filters: dict[str, list[str]] = {key: [] for key in _FILTER_KEYS}
    terms: list[str] = []
    quoted_hints: list[str] = []
    ignored_tokens: list[str] = []

    text = "" if query is None else str(query)
    for token in _tokens(text):
        filter_match = _FILTER_RE.match(token)
        if filter_match:
            key = filter_match.group("key").casefold()
            value = _clean_value(filter_match.group("value"))
            if key in filters and value:
                _append_unique(filters[key], value)
            else:
                ignored_tokens.append(token)
            continue

        if ":" in token:
            ignored_tokens.append(token)
            continue

        value = _clean_value(token)
        if not value:
            continue
        if _is_quoted(token):
            _append_unique(quoted_hints, value)
        else:
            _append_unique(terms, value.casefold())

    return {
        "filters": filters,
        "quoted_hints": quoted_hints,
        "terms": terms,
        "ignored_tokens": ignored_tokens,
    }


def _tokens(query: str) -> list[str]:
    return _TOKEN_RE.findall(query)


def _clean_value(value: str) -> str:
    text = " ".join(value.strip().split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        text = " ".join(text[1:-1].strip().split())
    return text


def _is_quoted(token: str) -> bool:
    stripped = token.strip()
    return len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"', "`"}


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
