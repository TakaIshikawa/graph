from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_edge_adjacency_markdown
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(unit_id: str, title: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
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


def test_export_edge_adjacency_markdown_groups_outgoing_edges_and_backlinks():
    markdown = export_edge_adjacency_markdown(
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
        "# Edge Adjacency\n"
        "\n"
        "## Alpha (`unit-a`)\n"
        "\n"
        "### Outgoing\n"
        "\n"
        "#### `builds_on`\n"
        "\n"
        "- Beta (`unit-b`)\n"
        "\n"
        "#### `references`\n"
        "\n"
        "- Gamma (`unit-c`)\n"
        "\n"
        "### Backlinks\n"
        "\n"
        "#### `challenges`\n"
        "\n"
        "- Gamma (`unit-c`)\n"
        "\n"
        "## Beta (`unit-b`)\n"
        "\n"
        "### Outgoing\n"
        "\n"
        "_None._\n"
        "\n"
        "### Backlinks\n"
        "\n"
        "#### `builds_on`\n"
        "\n"
        "- Alpha (`unit-a`)\n"
        "\n"
        "## Gamma (`unit-c`)\n"
        "\n"
        "### Outgoing\n"
        "\n"
        "#### `challenges`\n"
        "\n"
        "- Alpha (`unit-a`)\n"
        "\n"
        "### Backlinks\n"
        "\n"
        "#### `references`\n"
        "\n"
        "- Alpha (`unit-a`)\n"
    )


def test_export_edge_adjacency_markdown_can_omit_backlinks():
    markdown = export_edge_adjacency_markdown(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON)],
        include_backlinks=False,
    )

    assert "### Backlinks" not in markdown
    assert "#### `builds_on`\n\n- Beta (`unit-b`)" in markdown


def test_export_edge_adjacency_markdown_skips_missing_endpoint_units():
    markdown = export_edge_adjacency_markdown(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [
            edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
            edge("edge-b", "unit-a", "missing-to", EdgeRelation.REFERENCES),
            edge("edge-c", "missing-from", "unit-b", EdgeRelation.CHALLENGES),
        ],
    )

    assert "missing" not in markdown
    assert "#### `builds_on`\n\n- Beta (`unit-b`)" in markdown
    assert "`references`" not in markdown
    assert "`challenges`" not in markdown


def test_export_edge_adjacency_markdown_deduplicates_edge_rows_and_is_deterministic():
    units = [
        unit("unit-b", "Beta [B]"),
        unit("unit-a", "Alpha *A*"),
        unit("unit-c", "", metadata={"title": "Gamma"}),
    ]
    edges = [
        edge("edge-c", "unit-a", "unit-c", EdgeRelation.REFERENCES),
        edge("edge-b", "unit-a", "unit-b", EdgeRelation.REFERENCES),
        edge("edge-a", "unit-a", "unit-b", EdgeRelation.REFERENCES),
    ]

    first = export_edge_adjacency_markdown(units, edges)
    second = export_edge_adjacency_markdown(reversed(units), reversed(edges))

    assert first == second
    assert first.index("## Alpha \\*A\\* (`unit-a`)") < first.index("## Beta \\[B\\] (`unit-b`)")
    assert "## Gamma (`unit-c`)" in first
    assert first.count("- Beta \\[B\\] (`unit-b`)") == 1
    assert "- Gamma (`unit-c`)" in first


def test_export_edge_adjacency_markdown_writes_file_output(tmp_path):
    path = tmp_path / "nested" / "edge-adjacency.md"

    markdown = export_edge_adjacency_markdown(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON)],
        path,
    )

    assert path.read_text(encoding="utf-8") == markdown


def test_export_edge_adjacency_markdown_is_importable_from_graph_export():
    from graph.export import export_edge_adjacency_markdown as imported

    assert imported is export_edge_adjacency_markdown
