"""Tests for the GraphService."""

from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


@pytest.fixture
def populated_store(store: Store):
    """Store with a small graph: A -> B -> C, D (isolated)."""
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node",
            utility_score=0.9,
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="b",
            source_entity_type="knowledge_node",
            title="Node B",
            content="Second node",
            utility_score=0.7,
        )
    )
    c = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="c",
            source_entity_type="insight",
            title="Node C",
            content="Third node",
            content_type=ContentType.INSIGHT,
            utility_score=0.5,
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="d",
            source_entity_type="knowledge_item",
            title="Node D",
            content="Isolated node",
            content_type=ContentType.ARTIFACT,
            utility_score=0.8,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=a.id,
            to_unit_id=b.id,
            relation=EdgeRelation.BUILDS_ON,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=b.id,
            to_unit_id=c.id,
            relation=EdgeRelation.INSPIRES,
        )
    )
    return store


class TestGraphService:
    def test_rebuild(self, populated_store: Store):
        gs = GraphService(populated_store)
        count = gs.rebuild()
        assert count == 4

    def test_get_neighbors_depth_1(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()

        # Get node B's ID
        b = populated_store.get_unit_by_source("forty_two", "b", "knowledge_node")
        result = gs.get_neighbors(b.id, depth=1)
        assert result["center"] == b.id
        assert len(result["neighbors"]) == 2  # A and C

    def test_get_neighbors_depth_2(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()

        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        result = gs.get_neighbors(a.id, depth=2)
        assert len(result["neighbors"]) == 2  # B and C

    def test_get_neighbors_nonexistent(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()
        result = gs.get_neighbors("nonexistent")
        assert result["center"] is None

    def test_shortest_path(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()

        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        c = populated_store.get_unit_by_source("max", "c", "insight")
        path = gs.shortest_path(a.id, c.id)
        assert path is not None
        assert len(path) == 3  # A -> B -> C

    def test_shortest_path_no_connection(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()

        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        d = populated_store.get_unit_by_source("presence", "d", "knowledge_item")
        path = gs.shortest_path(a.id, d.id)
        assert path is None

    def test_clusters(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()
        clusters = gs.get_clusters(min_size=2)
        assert len(clusters) == 1  # A-B-C cluster, D is isolated
        assert len(clusters[0]) == 3

    def test_central_nodes(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()
        central = gs.get_central_nodes(limit=2)
        assert len(central) == 2
        # Each entry is (node_id, pagerank_score)
        assert all(isinstance(score, float) for _, score in central)

    def test_find_gaps(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()
        gaps = gs.find_gaps()
        # D is isolated with utility 0.8
        isolated = [g for g in gaps if g["gap_type"] == "isolated"]
        assert len(isolated) == 1

    def test_cross_project_connections(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()
        connections = gs.cross_project_connections()
        # B(forty_two) -> C(max) is cross-project
        assert len(connections) >= 1
        assert connections[0]["edge_count"] >= 1

    def test_stats(self, populated_store: Store):
        gs = GraphService(populated_store)
        gs.rebuild()
        s = gs.stats()
        assert s["nodes"] == 4
        assert s["edges"] == 2
        assert s["components"] == 2  # A-B-C and D
        assert "forty_two" in s["by_project"]

    def test_empty_graph(self, store: Store):
        gs = GraphService(store)
        gs.rebuild()
        assert gs.stats()["nodes"] == 0
        assert gs.get_clusters() == []
        assert gs.get_central_nodes() == []
        assert gs.get_bridges() == []
        assert gs.find_gaps() == []
