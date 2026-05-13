from __future__ import annotations

import csv
from io import StringIO

from graph.export.edge_cross_source_csv import export_edge_cross_source_csv
from graph.types.enums import ContentType, EdgeRelation
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, *, source_project: str, source_entity_type: str = "note") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
    )


def edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_cross_source_csv_empty_input_has_header_only():
    assert export_edge_cross_source_csv([], []) == (
        "edge_id,relation,from_unit_id,from_source_project,to_unit_id,to_source_project,"
        "from_source_entity_type,to_source_entity_type,direction_label\n"
    )


def test_edge_cross_source_csv_emits_known_endpoints_with_different_sources_only():
    text = export_edge_cross_source_csv(
        [unit("a", source_project="A"), unit("b", source_project="B"), unit("c", source_project="A")],
        [edge("e2", "a", "c"), edge("e1", "a", "b")],
    )

    assert rows(text) == [
        {
            "edge_id": "e1",
            "relation": "relates_to",
            "from_unit_id": "a",
            "from_source_project": "A",
            "to_unit_id": "b",
            "to_source_project": "B",
            "from_source_entity_type": "note",
            "to_source_entity_type": "note",
            "direction_label": "A -> B",
        }
    ]


def test_edge_cross_source_csv_can_include_unknown_endpoints():
    text = export_edge_cross_source_csv([unit("a", source_project="A")], [edge("e1", "a", "missing")], include_unknown=True)

    assert rows(text)[0]["to_source_project"] == "Unknown"
    assert rows(text)[0]["direction_label"] == "A -> Unknown"


def test_edge_cross_source_csv_is_deterministic_across_input_order():
    units = [unit("b", source_project="B"), unit("a", source_project="A")]
    edges = [edge("e2", "b", "a"), edge("e1", "a", "b")]

    forward = export_edge_cross_source_csv(units, edges)
    reverse = export_edge_cross_source_csv(list(reversed(units)), list(reversed(edges)))

    assert forward == reverse


def test_edge_cross_source_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "edge-cross-source.csv"
    units = [unit("a", source_project="A"), unit("b", source_project="B")]
    edges = [edge("e1", "a", "b")]

    expected = export_edge_cross_source_csv(units, edges)
    stats = export_edge_cross_source_csv(units, edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "edge_count": 1,
        "cross_source_edge_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
