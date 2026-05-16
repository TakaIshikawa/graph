from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_bridge_candidates_csv
from graph.types.enums import EdgeRelation
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    source: str,
    target: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.RELATES_TO,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=relation,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_bridge_candidates_computes_component_delta_and_context():
    text = export_relation_bridge_candidates_csv(
        [
            edge("2", "b", "c", metadata={"relation_type": "handoff"}),
            edge("1", "a", "b"),
            edge("3", "c", "d"),
            edge("4", "a", "d"),
            edge("5", "d", "e"),
        ]
    )

    exported = rows(text)
    assert exported[0] == {
        "relation": "relates_to",
        "relation_type": "Unknown",
        "source_unit_id": "d",
        "target_unit_id": "e",
        "source_degree": "3",
        "target_degree": "1",
        "component_delta_if_removed": "1",
        "shared_neighbor_count": "0",
        "bridge_score": "11.33",
    }
    assert {
        (row["source_unit_id"], row["target_unit_id"]): row["component_delta_if_removed"]
        for row in exported
    } == {
        ("d", "e"): "1",
        ("a", "b"): "0",
        ("a", "d"): "0",
        ("b", "c"): "0",
        ("c", "d"): "0",
    }


def test_relation_bridge_candidates_parallel_edges_do_not_split_components():
    text = export_relation_bridge_candidates_csv(
        [
            edge("a", "u1", "u2"),
            edge("b", "u2", "u1", relation="supports"),
        ]
    )

    assert [row["component_delta_if_removed"] for row in rows(text)] == ["0", "0"]


def test_relation_bridge_candidates_path_mode_and_dict_edges(tmp_path):
    edges = [{"relation": "links", "from_unit_id": "u1", "to_unit_id": "u2"}]
    expected = export_relation_bridge_candidates_csv(edges)
    path = tmp_path / "bridges.csv"

    stats = export_relation_bridge_candidates_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["edge_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
