"""Markdown export helpers for tag co-occurrence reports."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.rag import build_tag_cooccurrence_matrix
from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_tag_cooccurrence_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_count: int = 1,
    limit: int | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report of tag co-occurrence pairs."""
    matrix = build_tag_cooccurrence_matrix(units, min_count=min_count, limit=limit)
    rows = sorted(
        matrix["pairs"],
        key=lambda pair: (
            -pair["count"],
            _sort_key(pair["source"]),
            _sort_key(pair["target"]),
        ),
    )
    stats = matrix["stats"]
    text = _render_report(rows, stats=stats)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "rows_exported": len(rows),
        "units_scanned": stats["unit_count"],
        "min_count": stats["min_count"],
        "limit": stats["limit"],
        "bytes_written": output_path.stat().st_size,
    }


def _render_report(rows: list[dict[str, Any]], *, stats: dict[str, Any]) -> str:
    lines = [
        "# Tag Co-occurrence",
        "",
        "## Summary",
        "",
        "| Option | Value |",
        "| --- | ---: |",
        f"| Units scanned | {stats['unit_count']} |",
        f"| Tags found | {stats['tag_count']} |",
        f"| Pairs reported | {len(rows)} |",
        f"| Min count | {stats['min_count']} |",
    ]
    if stats["limit"] is not None:
        lines.append(f"| Limit | {stats['limit']} |")
    lines.extend(
        [
            "",
            "## Tag Pairs",
            "",
            "| Tag A | Tag B | Count |",
            "| --- | --- | ---: |",
        ]
    )
    if rows:
        for row in rows:
            lines.append(
                "| "
                f"{_markdown_cell(row['source'])} | "
                f"{_markdown_cell(row['target'])} | "
                f"{row['count']} |"
            )
    else:
        lines.append("| _None_ | _None_ | 0 |")
    return "\n".join(lines).rstrip() + "\n"


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
