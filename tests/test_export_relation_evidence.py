from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_relation_evidence_markdown
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
    *,
    weight: float = 1.0,
    source: EdgeSource = EdgeSource.INFERRED,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
        source=source,
        metadata={},
        created_at=EDGE_TIME,
    )


def test_export_relation_evidence_markdown_groups_relation_summaries_and_examples():
    markdown = export_relation_evidence_markdown(
        [
            unit("unit-c", "Gamma"),
            unit("unit-a", "Alpha"),
            unit("unit-b", "Beta"),
        ],
        [
            edge(
                "edge-c",
                "unit-c",
                "unit-a",
                EdgeRelation.CHALLENGES,
                weight=0.5,
                source=EdgeSource.SOURCE,
            ),
            edge(
                "edge-b",
                "unit-a",
                "unit-c",
                EdgeRelation.BUILDS_ON,
                weight=1.0,
                source=EdgeSource.MANUAL,
            ),
            edge(
                "edge-a",
                "unit-a",
                "unit-b",
                EdgeRelation.BUILDS_ON,
                weight=0.5,
                source=EdgeSource.INFERRED,
            ),
        ],
    )

    assert markdown == (
        "# Relation Evidence\n"
        "\n"
        "- Units: 3\n"
        "- Edges: 3\n"
        "- Valid edges: 3\n"
        "- Skipped edges: 0\n"
        "  - Missing from endpoint: 0\n"
        "  - Missing to endpoint: 0\n"
        "  - Missing both endpoints: 0\n"
        "\n"
        "## `builds_on`\n"
        "\n"
        "- Edges: 2\n"
        "- Average weight: 0.75\n"
        "- Sources: `inferred` 1, `manual` 1\n"
        "\n"
        "### Examples\n"
        "\n"
        "- Alpha (`unit-a`) -> Beta (`unit-b`) "
        "(edge: `edge-a`, weight: 0.5, source: `inferred`)\n"
        "- Alpha (`unit-a`) -> Gamma (`unit-c`) "
        "(edge: `edge-b`, weight: 1, source: `manual`)\n"
        "\n"
        "## `challenges`\n"
        "\n"
        "- Edges: 1\n"
        "- Average weight: 0.5\n"
        "- Sources: `source` 1\n"
        "\n"
        "### Examples\n"
        "\n"
        "- Gamma (`unit-c`) -> Alpha (`unit-a`) "
        "(edge: `edge-c`, weight: 0.5, source: `source`)\n"
    )


def test_export_relation_evidence_markdown_reports_and_omits_missing_endpoint_edges():
    markdown = export_relation_evidence_markdown(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [
            edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
            edge("edge-b", "unit-a", "missing-to", EdgeRelation.REFERENCES),
            edge("edge-c", "missing-from", "unit-b", EdgeRelation.CHALLENGES),
            edge("edge-d", "missing-from", "missing-to", EdgeRelation.RELATES_TO),
        ],
    )

    assert "- Edges: 4" in markdown
    assert "- Valid edges: 1" in markdown
    assert "- Skipped edges: 3" in markdown
    assert "  - Missing from endpoint: 1" in markdown
    assert "  - Missing to endpoint: 1" in markdown
    assert "  - Missing both endpoints: 1" in markdown
    assert "missing" not in markdown
    assert "## `builds_on`" in markdown
    assert "## `references`" not in markdown
    assert "## `challenges`" not in markdown
    assert "## `relates_to`" not in markdown


def test_export_relation_evidence_markdown_limits_examples_and_is_deterministic():
    units = [
        unit("unit-b", "Beta [B]"),
        unit("unit-a", "Alpha *A*"),
        unit("unit-c", "Gamma"),
    ]
    edges = [
        edge("edge-c", "unit-a", "unit-c", EdgeRelation.REFERENCES, weight=0.3),
        edge("edge-a", "unit-a", "unit-b", EdgeRelation.REFERENCES, weight=0.9),
        edge("edge-b", "unit-b", "unit-c", EdgeRelation.REFERENCES, weight=0.7),
    ]

    first = export_relation_evidence_markdown(units, edges, max_examples_per_relation=2)
    second = export_relation_evidence_markdown(
        reversed(units),
        reversed(edges),
        max_examples_per_relation=2,
    )

    assert first == second
    assert "- Alpha \\*A\\* (`unit-a`) -> Beta \\[B\\] (`unit-b`)" in first
    assert "- Alpha \\*A\\* (`unit-a`) -> Gamma (`unit-c`)" in first
    assert "edge-b" not in first


def test_export_relation_evidence_markdown_handles_no_valid_edges():
    markdown = export_relation_evidence_markdown(
        [unit("unit-a", "Alpha")],
        [edge("edge-a", "missing", "unit-a", EdgeRelation.BUILDS_ON)],
    )

    assert markdown == (
        "# Relation Evidence\n"
        "\n"
        "- Units: 1\n"
        "- Edges: 1\n"
        "- Valid edges: 0\n"
        "- Skipped edges: 1\n"
        "  - Missing from endpoint: 1\n"
        "  - Missing to endpoint: 0\n"
        "  - Missing both endpoints: 0\n"
        "\n"
        "_No valid relation evidence._\n"
    )


def test_export_relation_evidence_markdown_is_importable_from_graph_export():
    from graph.export import export_relation_evidence_markdown as imported

    assert imported is export_relation_evidence_markdown


@pytest.mark.parametrize("max_examples_per_relation", [-1, 1.5, True])
def test_export_relation_evidence_markdown_validates_max_examples(max_examples_per_relation):
    with pytest.raises(
        ValueError, match="max_examples_per_relation must be a non-negative integer or None"
    ):
        export_relation_evidence_markdown(
            [],
            [],
            max_examples_per_relation=max_examples_per_relation,
        )
