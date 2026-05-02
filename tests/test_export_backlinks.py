from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_unit_backlinks_markdown
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


def test_export_unit_backlinks_markdown_groups_incoming_and_outgoing_links(tmp_path):
    path = tmp_path / "backlinks.md"

    stats = export_unit_backlinks_markdown(
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
        path,
    )

    text = path.read_text(encoding="utf-8")

    assert text == (
        "# Unit Backlinks\n"
        "\n"
        "## Alpha (`unit-a`)\n"
        "\n"
        "### Incoming\n"
        "\n"
        "#### `challenges`\n"
        "\n"
        "- Gamma (`unit-c`)\n"
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
        "## Beta (`unit-b`)\n"
        "\n"
        "### Incoming\n"
        "\n"
        "#### `builds_on`\n"
        "\n"
        "- Alpha (`unit-a`)\n"
        "\n"
        "### Outgoing\n"
        "\n"
        "_None._\n"
        "\n"
        "## Gamma (`unit-c`)\n"
        "\n"
        "### Incoming\n"
        "\n"
        "#### `references`\n"
        "\n"
        "- Alpha (`unit-a`)\n"
        "\n"
        "### Outgoing\n"
        "\n"
        "#### `challenges`\n"
        "\n"
        "- Alpha (`unit-a`)\n"
    )
    assert stats == {
        "path": str(path),
        "units_scanned": 3,
        "units_exported": 3,
        "edges_scanned": 3,
        "bytes_written": len(text.encode("utf-8")),
    }


def test_export_unit_backlinks_markdown_can_omit_orphans(tmp_path):
    path = tmp_path / "backlinks.md"

    stats = export_unit_backlinks_markdown(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta"), unit("unit-c", "Gamma")],
        [edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON)],
        path,
        include_orphans=False,
    )

    text = path.read_text(encoding="utf-8")

    assert "## Alpha (`unit-a`)" in text
    assert "## Beta (`unit-b`)" in text
    assert "## Gamma (`unit-c`)" not in text
    assert stats["units_scanned"] == 3
    assert stats["units_exported"] == 2


def test_export_unit_backlinks_markdown_skips_unknown_edge_endpoints(tmp_path):
    path = tmp_path / "backlinks.md"

    stats = export_unit_backlinks_markdown(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [
            edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
            edge("edge-b", "unit-a", "missing-to", EdgeRelation.REFERENCES),
            edge("edge-c", "missing-from", "unit-b", EdgeRelation.CHALLENGES),
        ],
        path,
    )

    text = path.read_text(encoding="utf-8")

    assert "missing" not in text
    assert "`references`" not in text
    assert "`challenges`" not in text
    assert "#### `builds_on`\n\n- Beta (`unit-b`)" in text
    assert stats["edges_scanned"] == 3


def test_export_unit_backlinks_markdown_is_deterministic_and_escapes_titles(tmp_path):
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
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"

    export_unit_backlinks_markdown(units, edges, first_path)
    export_unit_backlinks_markdown(reversed(units), reversed(edges), second_path)
    first = first_path.read_text(encoding="utf-8")
    second = second_path.read_text(encoding="utf-8")

    assert first == second
    assert first.index("## Alpha \\*A\\* (`unit-a`)") < first.index(
        "## Beta \\[B\\] (`unit-b`)"
    )
    assert "## Gamma (`unit-c`)" in first
    assert first.count("- Beta \\[B\\] (`unit-b`)") == 1
    assert "- Gamma (`unit-c`)" in first


def test_export_unit_backlinks_markdown_importable_from_graph_export():
    from graph.export import export_unit_backlinks_markdown as imported

    assert imported is export_unit_backlinks_markdown
