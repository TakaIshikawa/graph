from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_tag_bridge_csv
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, *, tags: list[str] | None = None, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="Content",
        tags=tags or [],
        metadata=metadata or {},
    )


def edge(edge_id: str, source: str, target: str, relation: object = EdgeRelation.RELATES_TO) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=relation,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_tag_bridge_csv_empty_edge_input_returns_header():
    assert export_relation_tag_bridge_csv([], []) == (
        "relation,source_unit_id,target_unit_id,source_tag_count,target_tag_count,shared_tag_count,bridge_bucket\n"
    )


def test_export_relation_tag_bridge_csv_buckets_overlap_types():
    text = export_relation_tag_bridge_csv(
        [
            unit("a", tags=["AI", "Graph"]),
            unit("b", tags=["ai", "search"]),
            unit("c", tags=["finance"]),
            unit("d", tags=["ai", "graph"]),
        ],
        [edge("ab", "a", "b"), edge("ac", "a", "c"), edge("ad", "a", "d")],
    )

    assert [row["bridge_bucket"] for row in rows(text)] == [
        "partial_overlap",
        "disjoint",
        "same_tags",
    ]


def test_export_relation_tag_bridge_csv_uses_metadata_tags_fallback():
    text = export_relation_tag_bridge_csv(
        [unit("a", metadata={"tags": "review"}), unit("b", metadata={"tags": ["Review", "later"]})],
        [edge("ab", "a", "b")],
    )

    assert rows(text)[0]["shared_tag_count"] == "1"
    assert rows(text)[0]["bridge_bucket"] == "partial_overlap"


def test_export_relation_tag_bridge_csv_retains_missing_units_as_no_tags():
    text = export_relation_tag_bridge_csv([], [edge("ab", "missing-a", "missing-b")])

    assert rows(text)[0]["source_tag_count"] == "0"
    assert rows(text)[0]["target_tag_count"] == "0"
    assert rows(text)[0]["bridge_bucket"] == "no_tags"


def test_export_relation_tag_bridge_csv_path_mode(tmp_path):
    units = [unit("a", tags=["x"]), unit("b", tags=["y"])]
    edges = [edge("ab", "a", "b")]
    path = tmp_path / "bridge.csv"

    stats = export_relation_tag_bridge_csv(units, edges, path)

    assert rows(path.read_text(encoding="utf-8"))[0]["bridge_bucket"] == "disjoint"
    assert stats["unit_count"] == 2
    assert stats["edge_count"] == 1
    assert stats["rows_exported"] == 1
