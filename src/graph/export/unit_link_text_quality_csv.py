"""CSV export for weak Markdown link text in unit content."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "link_text", "target", "issue", "line_number"]
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_WEAK_LABELS = {"here", "this", "link", "click here"}


def export_units_to_link_text_quality_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(
        key=lambda row: (
            sort_key(row["unit_id"]),
            int(row["line_number"]),
            sort_key(row["link_text"]),
            sort_key(row["target"]),
            sort_key(row["issue"]),
        )
    )
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    links = _links(str(get(unit, "content") or ""))
    rows: list[dict[str, str | int]] = []
    for link in links:
        issue = _link_text_issue(link["link_text"])
        if issue:
            rows.append(_row(uid, link, issue))

    label_targets: dict[str, set[str]] = defaultdict(set)
    for link in links:
        label = _normalized_label(link["link_text"])
        if label:
            label_targets[label].add(link["target"])
    repeated_labels = {label for label, targets in label_targets.items() if len(targets) > 1}
    rows.extend(_row(uid, link, "repeated_text_different_target") for link in links if _normalized_label(link["link_text"]) in repeated_labels)
    return rows


def _links(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for match in _MD_LINK_RE.finditer(line):
            rows.append({"link_text": field_value(match.group(1)), "target": field_value(match.group(2)), "line_number": line_number})
    return rows


def _row(unit: str, link: dict[str, str | int], issue: str) -> dict[str, str | int]:
    return {
        "unit_id": unit,
        "link_text": link["link_text"],
        "target": link["target"],
        "issue": issue,
        "line_number": link["line_number"],
    }


def _link_text_issue(text: str) -> str:
    if not text:
        return "empty_text"
    if _is_bare_url(text):
        return "bare_url_text"
    if _normalized_label(text) in _WEAK_LABELS:
        return "weak_label"
    return ""


def _normalized_label(text: str) -> str:
    return re.sub(r"\s+", " ", field_value(text)).casefold()


def _is_bare_url(text: str) -> bool:
    parsed = urlparse(text if not text.casefold().startswith("www.") else f"https://{text}")
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
