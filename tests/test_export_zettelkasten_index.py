from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_zettelkasten_index_markdown
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(unit_id: str, title: str, tags: list[str]):
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"{title} content with a long enough preview body",
        content_type=ContentType.INSIGHT,
        tags=tags,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def edge(edge_id: str, from_id: str, to_id: str):
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_id,
        to_unit_id=to_id,
        relation=EdgeRelation.REFERENCES,
        created_at=UNIT_TIME,
    )


def test_export_zettelkasten_index_markdown_groups_and_links_units():
    text = export_zettelkasten_index_markdown(
        [
            unit("unit-b", "Beta [Note]", ["research"]),
            unit("unit-a", "Alpha", ["research", "inbox"]),
            unit("unit-c", "Gamma", []),
        ],
        [edge("edge-a", "unit-a", "unit-b")],
    )

    assert "## By Tag" in text
    assert "### inbox" in text
    assert "### research" in text
    assert "- `unit-b` Beta \\[Note\\]" in text
    assert "Backlinks: `unit-a` Alpha" in text
    assert "Outgoing: `unit-b` Beta \\[Note\\]" in text
    assert "## Orphans" in text
    assert "- `unit-c` Gamma" in text


def test_export_zettelkasten_index_markdown_is_deterministic_for_shuffled_input():
    units = [unit("unit-b", "Beta", ["tag"]), unit("unit-a", "Alpha", ["tag"])]
    edges = [edge("edge-a", "unit-a", "unit-b")]

    assert export_zettelkasten_index_markdown(units, edges) == export_zettelkasten_index_markdown(
        reversed(units),
        reversed(edges),
    )
