"""Markdown export for edge relation/source summaries."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_relation_summary_markdown(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    top_examples: int = 3,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown summary grouped by relation and source."""
    if (
        not isinstance(top_examples, int)
        or isinstance(top_examples, bool)
        or top_examples < 0
    ):
        raise ValueError("top_examples must be a non-negative integer")

    edge_list = sorted(list(edges), key=_edge_sort_key)
    rows = _summary_rows(edge_list, top_examples=top_examples)
    text = _render_report(rows, edges_scanned=len(edge_list), top_examples=top_examples)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "edges_scanned": len(edge_list),
        "groups_exported": len(rows),
        "top_examples": top_examples,
        "bytes_written": output_path.stat().st_size,
    }


def _summary_rows(edges: list[KnowledgeEdge], *, top_examples: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[KnowledgeEdge]] = {}
    for edge in edges:
        grouped.setdefault((_relation(edge), _source(edge)), []).append(edge)

    rows = []
    for (relation, source), group_edges in grouped.items():
        total_weight = sum(edge.weight for edge in group_edges)
        edge_count = len(group_edges)
        rows.append(
            {
                "relation": relation,
                "source": source,
                "edge_count": edge_count,
                "total_weight": total_weight,
                "average_weight": total_weight / edge_count if edge_count else 0.0,
                "examples": _examples_text(group_edges, limit=top_examples),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["edge_count"], _sort_key(row["relation"]), _sort_key(row["source"])),
    )


def _render_report(rows: list[dict[str, Any]], *, edges_scanned: int, top_examples: int) -> str:
    lines = [
        "# Edge Relation Summary",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Edges scanned | {edges_scanned} |",
        f"| Groups reported | {len(rows)} |",
        f"| Top examples | {top_examples} |",
        "",
        "## Relations",
        "",
        "| Relation | Source | Edges | Total weight | Average weight | Examples |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    if rows:
        for row in rows:
            lines.append(
                "| "
                f"{_markdown_cell(row['relation'])} | "
                f"{_markdown_cell(row['source'])} | "
                f"{row['edge_count']} | "
                f"{_weight_text(row['total_weight'])} | "
                f"{_weight_text(row['average_weight'])} | "
                f"{_markdown_cell(row['examples'])} |"
            )
    else:
        lines.append("| _None_ | _None_ | 0 | 0.00 | 0.00 | _None_ |")
    return "\n".join(lines).rstrip() + "\n"


def _examples_text(edges: list[KnowledgeEdge], *, limit: int) -> str:
    if limit == 0:
        return "_None_"
    examples = [
        f"{_inline_text(edge.from_unit_id) or '_None_'}->{_inline_text(edge.to_unit_id) or '_None_'}"
        for edge in sorted(edges, key=_edge_sort_key)[:limit]
    ]
    return "; ".join(examples) if examples else "_None_"


def _relation(edge: KnowledgeEdge) -> str:
    return _inline_text(getattr(edge.relation, "value", edge.relation)) or "Unknown"


def _source(edge: KnowledgeEdge) -> str:
    return _inline_text(getattr(edge.source, "value", edge.source)) or "Unknown"


def _edge_sort_key(edge: KnowledgeEdge) -> tuple[str, str, str, str, str]:
    return (_relation(edge), _source(edge), _inline_text(edge.from_unit_id), _inline_text(edge.to_unit_id), _inline_text(edge.id))


def _weight_text(value: float) -> str:
    return f"{value:.2f}"


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
