from __future__ import annotations

import csv
from io import StringIO

from graph.export.source_orphan_ratio_csv import export_source_orphan_ratio_csv
from graph.types.enums import ContentType, EdgeRelation
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, *, source_project: str = "A", source_entity_type: str = "note") -> KnowledgeUnit:
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


def test_source_orphan_ratio_csv_empty_input_has_header_only():
    assert export_source_orphan_ratio_csv([]) == (
        "source_project,source_entity_type,unit_count,connected_unit_count,orphan_unit_count,"
        "orphan_ratio,incoming_edge_count,outgoing_edge_count\n"
    )


def test_source_orphan_ratio_csv_counts_units_with_known_incoming_or_outgoing_edges_as_connected():
    text = export_source_orphan_ratio_csv(
        [unit("a"), unit("b"), unit("c")],
        [edge("e1", "a", "b")],
    )

    assert rows(text) == [
        {
            "source_project": "A",
            "source_entity_type": "note",
            "unit_count": "3",
            "connected_unit_count": "2",
            "orphan_unit_count": "1",
            "orphan_ratio": "0.33",
            "incoming_edge_count": "1",
            "outgoing_edge_count": "1",
        }
    ]


def test_source_orphan_ratio_csv_ignores_unknown_edge_endpoint_for_connected_counts():
    text = export_source_orphan_ratio_csv([unit("a")], [edge("e1", "a", "missing")])

    assert rows(text)[0]["connected_unit_count"] == "0"
    assert rows(text)[0]["outgoing_edge_count"] == "0"


def test_source_orphan_ratio_csv_sorts_rows_and_uses_unknown_fallbacks():
    text = export_source_orphan_ratio_csv(
        [
            unit("b", source_project="B", source_entity_type="task"),
            KnowledgeUnit.model_construct(id="a", source_project="", source_entity_type="", source_id="a", title="A"),
        ]
    )

    assert [(row["source_project"], row["source_entity_type"]) for row in rows(text)] == [
        ("B", "task"),
        ("Unknown", "Unknown"),
    ]


def test_source_orphan_ratio_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "source-orphan-ratio.csv"
    units = [unit("a"), unit("b")]
    edges = [edge("e1", "a", "b")]

    expected = export_source_orphan_ratio_csv(units, edges)
    stats = export_source_orphan_ratio_csv(units, edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "edge_count": 1,
        "source_type_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
