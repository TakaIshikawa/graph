from __future__ import annotations

from datetime import datetime, timezone

import networkx as nx

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def test_export_gexf_writes_gephi_attributes_and_stats(tmp_path):
    store = Store(tmp_path / "graph.db")
    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 3, 4, 5, 6, tzinfo=timezone.utc)
    edge_created_at = datetime(2024, 1, 4, 5, 6, 7, tzinfo=timezone.utc)
    try:
        a = store.insert_unit(
            KnowledgeUnit(
                id="unit-a",
                source_project=SourceProject.FORTY_TWO,
                source_id="a",
                source_entity_type="knowledge_node",
                title="Node A",
                content="First node",
                content_type=ContentType.FINDING,
                tags=["energy", "solar"],
                confidence=0.82,
                utility_score=0.91,
                created_at=created_at,
                updated_at=updated_at,
            )
        )
        b = store.insert_unit(
            KnowledgeUnit(
                id="unit-b",
                source_project=SourceProject.MAX,
                source_id="b",
                source_entity_type="insight",
                title="Node B",
                content="Second node",
                content_type=ContentType.INSIGHT,
                tags=["storage"],
                confidence=0.73,
                utility_score=0.64,
                created_at=created_at,
                updated_at=updated_at,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                id="edge-a-b",
                from_unit_id=a.id,
                to_unit_id=b.id,
                relation=EdgeRelation.BUILDS_ON,
                weight=0.75,
                source=EdgeSource.MANUAL,
                created_at=edge_created_at,
            )
        )

        output_path = tmp_path / "exports" / "gephi" / "graph.gexf"
        stats = GraphService(store).export_gexf(output_path)

        assert stats == {
            "path": str(output_path),
            "nodes_exported": 2,
            "edges_exported": 1,
            "bytes_written": output_path.stat().st_size,
        }

        exported = nx.read_gexf(output_path)
        assert set(exported.nodes) == {"unit-a", "unit-b"}
        assert exported.nodes[a.id]["title"] == "Node A"
        assert exported.nodes[a.id]["source_project"] == "forty_two"
        assert exported.nodes[a.id]["source_entity_type"] == "knowledge_node"
        assert exported.nodes[a.id]["content_type"] == "finding"
        assert exported.nodes[a.id]["tags"] == "energy,solar"
        assert exported.nodes[a.id]["created_at"] == created_at.isoformat()
        assert exported.nodes[a.id]["updated_at"] == updated_at.isoformat()
        assert float(exported.nodes[a.id]["confidence"]) == 0.82
        assert float(exported.nodes[a.id]["utility_score"]) == 0.91

        edge_data = exported.get_edge_data(a.id, b.id)
        assert edge_data["relation"] == "builds_on"
        assert edge_data["source"] == "manual"
        assert float(edge_data["weight"]) == 0.75
        assert edge_data["created_at"] == edge_created_at.isoformat()
    finally:
        store.close()


def test_export_gexf_is_repeatable_for_same_graph(tmp_path):
    store = Store(tmp_path / "graph.db")
    timestamp = datetime(2024, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    try:
        store.insert_unit(
            KnowledgeUnit(
                id="unit-a",
                source_project=SourceProject.MAX,
                source_id="a",
                source_entity_type="note",
                title="Alpha",
                content="Alpha content",
                content_type=ContentType.INSIGHT,
                tags=["alpha"],
                confidence=0.5,
                utility_score=0.6,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

        service = GraphService(store)
        first_path = tmp_path / "first.gexf"
        second_path = tmp_path / "second.gexf"

        first_stats = service.export_gexf(first_path)
        second_stats = service.export_gexf(second_path)

        assert first_stats["nodes_exported"] == second_stats["nodes_exported"] == 1
        assert first_stats["edges_exported"] == second_stats["edges_exported"] == 0
        assert first_path.read_text(encoding="utf-8") == second_path.read_text(
            encoding="utf-8"
        )
    finally:
        store.close()


def test_export_gexf_handles_empty_graph_and_creates_parent_directories(tmp_path):
    store = Store(tmp_path / "empty.db")
    try:
        output_path = tmp_path / "missing" / "parents" / "empty.gexf"
        stats = GraphService(store).export_gexf(output_path)

        assert stats == {
            "path": str(output_path),
            "nodes_exported": 0,
            "edges_exported": 0,
            "bytes_written": output_path.stat().st_size,
        }
        exported = nx.read_gexf(output_path)
        assert exported.number_of_nodes() == 0
        assert exported.number_of_edges() == 0
    finally:
        store.close()
