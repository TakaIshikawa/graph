"""Markdown concept map export helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_concept_map_markdown(
    units: Iterable[KnowledgeUnit],
    relationships: Iterable[KnowledgeEdge],
    *,
    group_by: str = "tag",
    max_links_per_unit: int = 5,
    title: str = "Concept Map",
) -> str:
    """Return a deterministic Markdown concept map grouped by tag or source."""
    if group_by not in {"tag", "source"}:
        raise ValueError("group_by must be 'tag' or 'source'")
    if (
        not isinstance(max_links_per_unit, int)
        or isinstance(max_links_per_unit, bool)
        or max_links_per_unit < 0
    ):
        raise ValueError("max_links_per_unit must be a non-negative integer")

    ordered_units = sorted(
        list(units),
        key=lambda unit: (_unit_label(unit).casefold(), _inline_text(unit.id)),
    )
    unit_by_id = {unit.id: unit for unit in ordered_units}
    links_by_unit = _links_by_unit(relationships, unit_by_id)
    groups = _groups(ordered_units, group_by=group_by)

    lines = [f"# {_heading_text(title) or 'Concept Map'}", ""]
    if not ordered_units:
        lines.append("_No units available._")
        return "\n".join(lines).rstrip() + "\n"

    for group_name, group_units in groups:
        lines.extend([f"## {_heading_text(group_name)}", ""])
        for unit in group_units:
            lines.append(f"- {_inline_markdown(_unit_label(unit))} (`{_code_text(unit.id)}`)")
            unit_links = links_by_unit.get(unit.id, [])[:max_links_per_unit]
            if unit_links:
                for link in unit_links:
                    lines.append(
                        "  - "
                        f"{_inline_markdown(link.relation)} {link.direction} "
                        f"{_inline_markdown(link.label)} (`{_code_text(link.unit_id)}`) "
                        f"- weight {link.weight:g}"
                    )
            else:
                lines.append("  - _No linked concepts._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


class _ConceptLink:
    def __init__(
        self,
        *,
        unit_id: str,
        label: str,
        relation: str,
        direction: str,
        weight: float,
        edge_id: str,
    ) -> None:
        self.unit_id = unit_id
        self.label = label
        self.relation = relation
        self.direction = direction
        self.weight = weight
        self.edge_id = edge_id

    @property
    def sort_key(self) -> tuple[float, str, str, str, str, str]:
        return (
            -self.weight,
            self.relation.casefold(),
            self.label.casefold(),
            self.unit_id,
            self.direction,
            self.edge_id,
        )


def _groups(
    units: list[KnowledgeUnit],
    *,
    group_by: str,
) -> list[tuple[str, list[KnowledgeUnit]]]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        for group_name in _unit_groups(unit, group_by=group_by):
            grouped[group_name].append(unit)

    return [
        (group_name, sorted(group_units, key=lambda unit: (_unit_label(unit).casefold(), unit.id)))
        for group_name, group_units in sorted(
            grouped.items(), key=lambda item: (item[0].casefold(), item[0])
        )
    ]


def _unit_groups(unit: KnowledgeUnit, *, group_by: str) -> list[str]:
    if group_by == "tag":
        tags = sorted(
            {_inline_text(tag) for tag in unit.tags if _inline_text(tag)},
            key=lambda tag: (tag.casefold(), tag),
        )
        return tags or ["Untagged"]

    source = _inline_text(_field_value(unit.source_project))
    return [source or "Unknown Source"]


def _links_by_unit(
    relationships: Iterable[KnowledgeEdge],
    unit_by_id: dict[str, KnowledgeUnit],
) -> dict[str, list[_ConceptLink]]:
    links: dict[str, list[_ConceptLink]] = defaultdict(list)
    for edge in relationships:
        from_unit = unit_by_id.get(edge.from_unit_id)
        to_unit = unit_by_id.get(edge.to_unit_id)
        if from_unit is None or to_unit is None:
            continue

        relation = _inline_text(_field_value(edge.relation)) or "related"
        edge_id = _inline_text(edge.id)
        links[from_unit.id].append(
            _ConceptLink(
                unit_id=to_unit.id,
                label=_unit_label(to_unit),
                relation=relation,
                direction="->",
                weight=edge.weight,
                edge_id=edge_id,
            )
        )
        links[to_unit.id].append(
            _ConceptLink(
                unit_id=from_unit.id,
                label=_unit_label(from_unit),
                relation=relation,
                direction="<-",
                weight=edge.weight,
                edge_id=edge_id,
            )
        )

    return {
        unit_id: sorted(unit_links, key=lambda link: link.sort_key)
        for unit_id, unit_links in links.items()
    }


def _unit_label(unit: KnowledgeUnit) -> str:
    for value in (
        unit.metadata.get("label"),
        unit.title,
        unit.metadata.get("title"),
        unit.metadata.get("name"),
        unit.id,
        unit.source_id,
    ):
        text = _inline_text(value)
        if text:
            return text
    return "Untitled"


def _field_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _heading_text(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("#", r"\#")


def _inline_markdown(value: object) -> str:
    return (
        _inline_text(value)
        .replace("\\", r"\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def _code_text(value: object) -> str:
    return _inline_text(value).replace("`", r"\`")
