from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_graph_adjacency_markdown
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
        confidence=0.8,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=1.0,
        source=EdgeSource.INFERRED,
        metadata={},
        created_at=EDGE_TIME,
    )


def test_export_graph_adjacency_markdown_groups_outgoing_edges_and_backlinks():
    markdown = export_graph_adjacency_markdown(
        [
            unit("unit-c", "Gamma"),
            unit("unit-a", "Alpha"),
            unit("unit-b", "Beta"),
        ],
        [
            edge("edge-c", "unit-c", "unit-a", EdgeRelation.CHALLENGES),
            edge("edge-b", "unit-a", "unit-c", EdgeRelation.REFERENCES),
            edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        ],
    )

    assert markdown == (
        "# Graph Adjacency List\n"
        "\n"
        '<a id="unit-unit-a"></a>\n'
        "\n"
        "## Alpha\n"
        "\n"
        "- ID: `unit-a`\n"
        "\n"
        "### Outgoing\n"
        "\n"
        "- `builds_on`: [Beta](#unit-unit-b)\n"
        "- `references`: [Gamma](#unit-unit-c)\n"
        "\n"
        "### Backlinks\n"
        "\n"
        "- `challenges`: [Gamma](#unit-unit-c)\n"
        "\n"
        '<a id="unit-unit-b"></a>\n'
        "\n"
        "## Beta\n"
        "\n"
        "- ID: `unit-b`\n"
        "\n"
        "### Outgoing\n"
        "\n"
        "_None._\n"
        "\n"
        "### Backlinks\n"
        "\n"
        "- `builds_on`: [Alpha](#unit-unit-a)\n"
        "\n"
        '<a id="unit-unit-c"></a>\n'
        "\n"
        "## Gamma\n"
        "\n"
        "- ID: `unit-c`\n"
        "\n"
        "### Outgoing\n"
        "\n"
        "- `challenges`: [Alpha](#unit-unit-a)\n"
        "\n"
        "### Backlinks\n"
        "\n"
        "- `references`: [Alpha](#unit-unit-a)\n"
    )


def test_export_graph_adjacency_markdown_can_omit_backlinks():
    markdown = export_graph_adjacency_markdown(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON)],
        include_backlinks=False,
    )

    assert "### Backlinks" not in markdown
    assert "- `builds_on`: [Beta](#unit-unit-b)" in markdown


def test_export_graph_adjacency_markdown_skips_missing_edge_endpoints():
    markdown = export_graph_adjacency_markdown(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [
            edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
            edge("edge-b", "unit-a", "missing", EdgeRelation.REFERENCES),
            edge("edge-c", "missing", "unit-b", EdgeRelation.CHALLENGES),
        ],
    )

    assert "missing" not in markdown
    assert "- `builds_on`: [Beta](#unit-unit-b)" in markdown
    assert "`references`" not in markdown
    assert "`challenges`" not in markdown


def test_export_graph_adjacency_markdown_is_deterministic_and_escapes_markdown():
    units = [
        unit("unit-b", "Beta [B]"),
        unit("unit-a", "Alpha (A)"),
    ]
    edges = [
        edge("edge-b", "unit-a", "unit-b", EdgeRelation.REFERENCES),
        edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
    ]

    first = export_graph_adjacency_markdown(units, edges)
    second = export_graph_adjacency_markdown(reversed(units), reversed(edges))

    assert first == second
    assert first.index("## Alpha (A)") < first.index("## Beta [B]")
    assert "- `builds_on`: [Beta \\[B\\]](#unit-unit-b)" in first
    assert "- `references`: [Beta \\[B\\]](#unit-unit-b)" in first
