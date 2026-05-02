"""Markdown relation evidence export helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_evidence_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    *,
    max_examples_per_relation: int | None = 5,
) -> str:
    """Render a deterministic Markdown report summarizing edge evidence by relation."""
    if max_examples_per_relation is not None and (
        not isinstance(max_examples_per_relation, int)
        or isinstance(max_examples_per_relation, bool)
        or max_examples_per_relation < 0
    ):
        raise ValueError("max_examples_per_relation must be a non-negative integer or None")

    exported_units = sorted(units, key=_unit_sort_key)
    units_by_id = {unit.id: unit for unit in exported_units}
    edge_list = list(edges)
    grouped: dict[str, list[KnowledgeEdge]] = defaultdict(list)
    skipped = Counter()

    for edge in edge_list:
        has_from = edge.from_unit_id in units_by_id
        has_to = edge.to_unit_id in units_by_id
        if not has_from and not has_to:
            skipped["missing_both"] += 1
            continue
        if not has_from:
            skipped["missing_from"] += 1
            continue
        if not has_to:
            skipped["missing_to"] += 1
            continue
        grouped[_field_value(edge.relation)].append(edge)

    valid_edge_count = sum(len(relation_edges) for relation_edges in grouped.values())
    skipped_edge_count = sum(skipped.values())
    lines = [
        "# Relation Evidence",
        "",
        f"- Units: {len(exported_units)}",
        f"- Edges: {len(edge_list)}",
        f"- Valid edges: {valid_edge_count}",
        f"- Skipped edges: {skipped_edge_count}",
        f"  - Missing from endpoint: {skipped['missing_from']}",
        f"  - Missing to endpoint: {skipped['missing_to']}",
        f"  - Missing both endpoints: {skipped['missing_both']}",
        "",
    ]

    if not grouped:
        lines.extend(["_No valid relation evidence._", ""])
        return "\n".join(lines).rstrip() + "\n"

    for relation in sorted(grouped, key=_sort_text):
        relation_edges = sorted(
            grouped[relation],
            key=lambda edge: _edge_sort_key(edge, units_by_id),
        )
        source_counts = Counter(_field_value(edge.source) for edge in relation_edges)
        lines.extend(
            [
                f"## `{_code_text(relation)}`",
                "",
                f"- Edges: {len(relation_edges)}",
                f"- Average weight: {_number_text(_average_weight(relation_edges))}",
                f"- Sources: {_source_counts_text(source_counts)}",
                "",
                "### Examples",
                "",
            ]
        )

        if max_examples_per_relation == 0:
            lines.append("_No examples requested._")
        else:
            for edge in relation_edges[:max_examples_per_relation]:
                from_unit = units_by_id[edge.from_unit_id]
                to_unit = units_by_id[edge.to_unit_id]
                lines.append(
                    f"- {_unit_ref(from_unit)} -> {_unit_ref(to_unit)} "
                    f"(edge: `{_code_text(edge.id)}`, weight: {_number_text(edge.weight)}, "
                    f"source: `{_code_text(_field_value(edge.source))}`)"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _average_weight(edges: list[KnowledgeEdge]) -> float:
    return sum(float(edge.weight) for edge in edges) / len(edges)


def _source_counts_text(source_counts: Counter[str]) -> str:
    if not source_counts:
        return "_None._"
    return ", ".join(
        f"`{_code_text(source)}` {count}"
        for source, count in sorted(
            source_counts.items(),
            key=lambda item: _sort_text(item[0]),
        )
    )


def _unit_ref(unit: KnowledgeUnit) -> str:
    return f"{_markdown_text(_unit_title(unit))} (`{_code_text(unit.id)}`)"


def _unit_title(unit: KnowledgeUnit) -> str:
    for value in (
        unit.title,
        unit.metadata.get("title"),
        unit.metadata.get("label"),
        unit.metadata.get("name"),
        unit.id,
        unit.source_id,
    ):
        text = _inline_text(value)
        if text:
            return text
    return "Untitled"


def _edge_sort_key(
    edge: KnowledgeEdge,
    units_by_id: dict[str, KnowledgeUnit],
) -> tuple[str, str, str, str, str, str, float, str]:
    from_unit = units_by_id[edge.from_unit_id]
    to_unit = units_by_id[edge.to_unit_id]
    return (
        *_unit_sort_key(from_unit),
        *_unit_sort_key(to_unit),
        -float(edge.weight),
        _inline_text(edge.id),
    )


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    title = _unit_title(unit)
    return (title.casefold(), title, _inline_text(unit.id))


def _sort_text(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _number_text(value: object) -> str:
    number = round(float(value), 6)
    return f"{number:g}"


def _markdown_text(value: object) -> str:
    return (
        _inline_text(value)
        .replace("\\", r"\\")
        .replace("*", r"\*")
        .replace("_", r"\_")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("`", r"\`")
    )


def _code_text(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("`", r"\`")


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()
