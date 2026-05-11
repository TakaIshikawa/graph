from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_graph_rdf_turtle
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(unit_id: str, title: str = "Title") -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content="Line 1\nLine \"2\" \\ end",
        content_type=ContentType.IDEA,
        tags=["zeta", "alpha"],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def test_export_graph_rdf_turtle_serializes_units_and_edges():
    text = export_graph_rdf_turtle(
        [unit("b"), unit("a", title="Alpha")],
        [
            KnowledgeEdge(
                id="edge-a",
                from_unit_id="a",
                to_unit_id="b",
                relation=EdgeRelation.RELATES_TO,
                created_at=UNIT_TIME,
            )
        ],
    )

    assert text.index("<https://example.org/graph/unit/a>") < text.index("<https://example.org/graph/unit/b>")
    assert 'dcterms:title "Alpha"' in text
    assert 'kg:content "Line 1\\nLine \\"2\\" \\\\ end"' in text
    assert "kg:tag \"alpha\"" in text
    assert "<https://example.org/graph/unit/a> <https://example.org/graph/relation/relates_to> <https://example.org/graph/unit/b> ." in text


def test_export_graph_rdf_turtle_writes_file(tmp_path):
    path = tmp_path / "graph.ttl"

    text = export_graph_rdf_turtle(unit("a"), path=path)

    assert path.read_text(encoding="utf-8") == text
