from __future__ import annotations

import pytest

from graph.export import export_edge_relation_summary_markdown
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    *,
    source: EdgeSource = EdgeSource.INFERRED,
    weight: float = 1.0,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=source,
        weight=weight,
    )


def test_edge_relation_summary_groups_by_relation_and_source():
    text = export_edge_relation_summary_markdown(
        [
            edge("b", "unit-b", "unit-c", EdgeRelation.BUILDS_ON, source=EdgeSource.MANUAL, weight=2.0),
            edge("a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON, source=EdgeSource.MANUAL, weight=1.5),
            edge("c", "unit-c", "unit-a", EdgeRelation.BUILDS_ON, source=EdgeSource.INFERRED),
            edge("d", "unit-d", "unit-a", EdgeRelation.REFERENCES, source=EdgeSource.MANUAL, weight=0.5),
        ]
    )

    assert "| Edges scanned | 4 |" in text
    assert "| Groups reported | 3 |" in text
    assert "| builds_on | manual | 2 | 3.50 | 1.75 | unit-a->unit-b; unit-b->unit-c |" in text
    assert "| builds_on | inferred | 1 | 1.00 | 1.00 | unit-c->unit-a |" in text
    assert "| references | manual | 1 | 0.50 | 0.50 | unit-d->unit-a |" in text


def test_edge_relation_summary_limits_examples_and_is_deterministic():
    edges = [
        edge("c", "unit-c", "unit-z"),
        edge("a", "unit-a", "unit-z"),
        edge("b", "unit-b", "unit-z"),
    ]

    first = export_edge_relation_summary_markdown(edges, top_examples=2)
    second = export_edge_relation_summary_markdown(reversed(edges), top_examples=2)

    assert first == second
    assert "| relates_to | inferred | 3 | 3.00 | 1.00 | unit-a->unit-z; unit-b->unit-z |" in first
    assert "unit-c->unit-z" not in first


def test_edge_relation_summary_escapes_examples_and_handles_zero_examples():
    text = export_edge_relation_summary_markdown(
        [edge("a", "from|id", r"to\\id")],
        top_examples=0,
    )

    assert "| relates_to | inferred | 1 | 1.00 | 1.00 | _None_ |" in text

    with_examples = export_edge_relation_summary_markdown([edge("a", "from|id", r"to\\id")])
    assert r"from\|id->to\\\\id" in with_examples


def test_edge_relation_summary_empty_report():
    assert export_edge_relation_summary_markdown([]) == (
        "# Edge Relation Summary\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        "| Edges scanned | 0 |\n"
        "| Groups reported | 0 |\n"
        "| Top examples | 3 |\n"
        "\n"
        "## Relations\n"
        "\n"
        "| Relation | Source | Edges | Total weight | Average weight | Examples |\n"
        "| --- | --- | ---: | ---: | ---: | --- |\n"
        "| _None_ | _None_ | 0 | 0.00 | 0.00 | _None_ |\n"
    )


def test_edge_relation_summary_writes_path_and_returns_stats(tmp_path):
    output_path = tmp_path / "reports" / "edges.md"
    edges = [edge("a", "unit-a", "unit-b")]

    text = export_edge_relation_summary_markdown(edges, top_examples=1)
    stats = export_edge_relation_summary_markdown(edges, output_path, top_examples=1)

    assert output_path.read_text(encoding="utf-8") == text
    assert stats == {
        "path": str(output_path),
        "edges_scanned": 1,
        "groups_exported": 1,
        "top_examples": 1,
        "bytes_written": output_path.stat().st_size,
    }


@pytest.mark.parametrize("top_examples", [-1, "2", None, True])
def test_edge_relation_summary_validates_top_examples(top_examples):
    with pytest.raises(ValueError, match="top_examples must be a non-negative integer"):
        export_edge_relation_summary_markdown([], top_examples=top_examples)


def test_edge_relation_summary_is_importable_from_graph_export():
    from graph.export import export_edge_relation_summary_markdown as imported

    assert imported is export_edge_relation_summary_markdown
