"""Tests for Store.identify_knowledge_gaps."""

from __future__ import annotations

import os
import struct
import tempfile

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _make_embedding(*values: float) -> bytes:
    """Pack floats into a BLOB embedding."""
    return struct.pack(f"{len(values)}f", *values)


def _make_unit(
    source_id: str,
    title: str,
    content: str = "content",
    source_project: str = SourceProject.FORTY_TWO,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=source_project,
        source_id=source_id,
        source_entity_type="knowledge_node",
        title=title,
        content=content,
        content_type=ContentType.FINDING,
    )


class TestIdentifyKnowledgeGaps:
    def test_empty_database(self, store: Store):
        gaps = store.identify_knowledge_gaps()
        assert gaps == []

    def test_no_embeddings(self, store: Store):
        store.insert_unit(_make_unit("n1", "Unit without embedding"))
        gaps = store.identify_knowledge_gaps()
        assert gaps == []

    def test_isolated_unit_detected(self, store: Store):
        """A unit with an embedding but no edges should appear as isolated."""
        u1 = store.insert_unit(_make_unit("n1", "Machine learning basics"))
        u2 = store.insert_unit(_make_unit("n2", "Deep learning fundamentals"))
        u3 = store.insert_unit(_make_unit("n3", "Neural network architectures"))

        # Give them similar embeddings
        store.update_embedding(u1.id, _make_embedding(1.0, 0.0, 0.0))
        store.update_embedding(u2.id, _make_embedding(0.9, 0.1, 0.0))
        store.update_embedding(u3.id, _make_embedding(0.8, 0.2, 0.0))

        # No edges at all → all isolated
        gaps = store.identify_knowledge_gaps()

        isolated = [g for g in gaps if g["gap_type"] == "isolated_unit"]
        assert len(isolated) == 3
        for gap in isolated:
            assert gap["coverage_score"] == 0.0
            assert len(gap["unit_ids"]) == 1

    def test_isolated_unit_suggests_connections(self, store: Store):
        """Isolated units should suggest connections to semantically similar units."""
        u1 = store.insert_unit(_make_unit("n1", "Python programming"))
        u2 = store.insert_unit(_make_unit("n2", "Python libraries"))

        # Very similar embeddings
        store.update_embedding(u1.id, _make_embedding(1.0, 0.0, 0.0))
        store.update_embedding(u2.id, _make_embedding(0.95, 0.05, 0.0))

        gaps = store.identify_knowledge_gaps()
        isolated = [g for g in gaps if g["gap_type"] == "isolated_unit"]
        assert len(isolated) == 2

        # Each isolated unit should suggest connecting to the other
        for gap in isolated:
            assert len(gap["suggested_connections"]) >= 1
            conn = gap["suggested_connections"][0]
            assert "from_unit_id" in conn
            assert "to_unit_id" in conn
            assert "similarity" in conn
            assert conn["similarity"] > 0.3

    def test_under_connected_topic(self, store: Store):
        """Unit with few edges despite many semantic neighbors."""
        u1 = store.insert_unit(_make_unit("n1", "Topic A"))
        u2 = store.insert_unit(_make_unit("n2", "Topic B"))
        u3 = store.insert_unit(_make_unit("n3", "Topic C"))
        u4 = store.insert_unit(_make_unit("n4", "Topic D"))

        # All similar embeddings
        store.update_embedding(u1.id, _make_embedding(1.0, 0.0, 0.0))
        store.update_embedding(u2.id, _make_embedding(0.9, 0.1, 0.0))
        store.update_embedding(u3.id, _make_embedding(0.85, 0.15, 0.0))
        store.update_embedding(u4.id, _make_embedding(0.8, 0.2, 0.0))

        # u1 has only one edge, despite being similar to u2, u3, u4
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=u1.id,
                to_unit_id=u2.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )

        gaps = store.identify_knowledge_gaps(max_connections=2)
        under = [g for g in gaps if g["gap_type"] == "under_connected_topic"]
        # u1 has 1 edge (<=2 max_connections) and semantic neighbors u3,u4 not connected
        u1_gaps = [g for g in under if u1.id in g["unit_ids"]]
        assert len(u1_gaps) == 1
        assert u1_gaps[0]["coverage_score"] > 0.0
        assert u1_gaps[0]["coverage_score"] < 1.0
        assert len(u1_gaps[0]["suggested_connections"]) >= 1

    def test_under_connected_excludes_well_connected(self, store: Store):
        """Units with many edges should not appear as under-connected."""
        units = []
        for i in range(5):
            u = store.insert_unit(_make_unit(f"n{i}", f"Topic {i}"))
            store.update_embedding(u.id, _make_embedding(1.0, float(i) * 0.01, 0.0))
            units.append(u)

        # Connect u0 to all others (4 edges)
        for u in units[1:]:
            store.insert_edge(
                KnowledgeEdge(
                    from_unit_id=units[0].id,
                    to_unit_id=u.id,
                    relation=EdgeRelation.RELATES_TO,
                )
            )

        gaps = store.identify_knowledge_gaps(max_connections=2)
        under = [g for g in gaps if g["gap_type"] == "under_connected_topic"]
        # u0 has 4 edges, so it should NOT appear as under-connected
        u0_gaps = [g for g in under if units[0].id in g["unit_ids"]]
        assert len(u0_gaps) == 0

    def test_sparse_region_detected(self, store: Store):
        """Small clusters below threshold should be detected as sparse regions."""
        # Cluster A: 2 similar units (below cluster_threshold=5)
        ua1 = store.insert_unit(_make_unit("a1", "Alpha topic 1"))
        ua2 = store.insert_unit(_make_unit("a2", "Alpha topic 2"))
        store.update_embedding(ua1.id, _make_embedding(1.0, 0.0, 0.0))
        store.update_embedding(ua2.id, _make_embedding(0.95, 0.05, 0.0))

        # Cluster B: 2 units in a different region
        ub1 = store.insert_unit(_make_unit("b1", "Beta topic 1"))
        ub2 = store.insert_unit(_make_unit("b2", "Beta topic 2"))
        store.update_embedding(ub1.id, _make_embedding(0.0, 1.0, 0.0))
        store.update_embedding(ub2.id, _make_embedding(0.05, 0.95, 0.0))

        gaps = store.identify_knowledge_gaps(cluster_threshold=5)
        sparse = [g for g in gaps if g["gap_type"] == "sparse_region"]
        assert len(sparse) >= 2
        for gap in sparse:
            assert gap["semantic_cluster"] is not None
            assert len(gap["unit_ids"]) < 5

    def test_sparse_region_not_flagged_above_threshold(self, store: Store):
        """Clusters at or above threshold should NOT be flagged as sparse."""
        units = []
        for i in range(6):
            u = store.insert_unit(_make_unit(f"n{i}", f"Topic {i}"))
            # All very similar → same cluster
            store.update_embedding(u.id, _make_embedding(1.0, float(i) * 0.01, 0.0))
            units.append(u)

        gaps = store.identify_knowledge_gaps(cluster_threshold=5)
        sparse = [g for g in gaps if g["gap_type"] == "sparse_region"]
        # All 6 units should cluster together (above threshold 5)
        assert len(sparse) == 0

    def test_sparse_region_coverage_score(self, store: Store):
        """Sparse region coverage reflects intra-cluster edge density."""
        u1 = store.insert_unit(_make_unit("n1", "A"))
        u2 = store.insert_unit(_make_unit("n2", "B"))
        u3 = store.insert_unit(_make_unit("n3", "C"))
        store.update_embedding(u1.id, _make_embedding(1.0, 0.0, 0.0))
        store.update_embedding(u2.id, _make_embedding(0.95, 0.05, 0.0))
        store.update_embedding(u3.id, _make_embedding(0.9, 0.1, 0.0))

        # Add one edge within the cluster (out of 3 possible)
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=u1.id,
                to_unit_id=u2.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )

        gaps = store.identify_knowledge_gaps(cluster_threshold=5)
        sparse = [g for g in gaps if g["gap_type"] == "sparse_region"]
        assert len(sparse) == 1
        # 1 edge out of 3 possible → coverage = 1/3 ≈ 0.3333
        assert 0.3 <= sparse[0]["coverage_score"] <= 0.35

    def test_suggested_connections_in_sparse_region(self, store: Store):
        """Sparse regions should suggest unconnected pairs within the cluster."""
        u1 = store.insert_unit(_make_unit("n1", "A"))
        u2 = store.insert_unit(_make_unit("n2", "B"))
        store.update_embedding(u1.id, _make_embedding(1.0, 0.0, 0.0))
        store.update_embedding(u2.id, _make_embedding(0.95, 0.05, 0.0))

        gaps = store.identify_knowledge_gaps(cluster_threshold=5)
        sparse = [g for g in gaps if g["gap_type"] == "sparse_region"]
        assert len(sparse) == 1
        assert len(sparse[0]["suggested_connections"]) >= 1
        conn = sparse[0]["suggested_connections"][0]
        assert set([conn["from_unit_id"], conn["to_unit_id"]]) == set([u1.id, u2.id])

    def test_source_filter(self, store: Store):
        """Only units matching source_filter should be analyzed."""
        u1 = store.insert_unit(
            _make_unit("n1", "Forty-two unit", source_project=SourceProject.FORTY_TWO)
        )
        u2 = store.insert_unit(
            _make_unit("m1", "Max unit", source_project=SourceProject.MAX)
        )
        store.update_embedding(u1.id, _make_embedding(1.0, 0.0, 0.0))
        store.update_embedding(u2.id, _make_embedding(0.9, 0.1, 0.0))

        # Filter to only forty_two
        gaps = store.identify_knowledge_gaps(source_filter=SourceProject.FORTY_TWO)
        all_unit_ids = set()
        for gap in gaps:
            all_unit_ids.update(gap["unit_ids"])
        assert u1.id in all_unit_ids or len(gaps) == 0
        assert u2.id not in all_unit_ids

    def test_source_filter_excludes_unmatched(self, store: Store):
        """With source_filter, units from other projects are invisible."""
        u1 = store.insert_unit(
            _make_unit("n1", "Unit A", source_project=SourceProject.MAX)
        )
        store.update_embedding(u1.id, _make_embedding(1.0, 0.0, 0.0))

        gaps = store.identify_knowledge_gaps(source_filter=SourceProject.FORTY_TWO)
        assert gaps == []

    def test_fully_connected_graph(self, store: Store):
        """Fully connected graph: no isolated or under-connected gaps."""
        units = []
        for i in range(4):
            u = store.insert_unit(_make_unit(f"n{i}", f"Topic {i}"))
            store.update_embedding(u.id, _make_embedding(1.0, float(i) * 0.01, 0.0))
            units.append(u)

        # Connect every pair
        for i in range(4):
            for j in range(i + 1, 4):
                store.insert_edge(
                    KnowledgeEdge(
                        from_unit_id=units[i].id,
                        to_unit_id=units[j].id,
                        relation=EdgeRelation.RELATES_TO,
                    )
                )

        gaps = store.identify_knowledge_gaps(max_connections=100)
        isolated = [g for g in gaps if g["gap_type"] == "isolated_unit"]
        under = [g for g in gaps if g["gap_type"] == "under_connected_topic"]
        assert len(isolated) == 0
        assert len(under) == 0

    def test_completely_disconnected_units(self, store: Store):
        """All units disconnected → all should be isolated."""
        units = []
        for i in range(3):
            u = store.insert_unit(_make_unit(f"n{i}", f"Topic {i}"))
            # Orthogonal embeddings to avoid clustering confusion
            emb = [0.0, 0.0, 0.0]
            emb[i] = 1.0
            store.update_embedding(u.id, _make_embedding(*emb))
            units.append(u)

        gaps = store.identify_knowledge_gaps(min_similarity=0.5)
        isolated = [g for g in gaps if g["gap_type"] == "isolated_unit"]
        assert len(isolated) == 3
        # Orthogonal embeddings → no suggested connections above threshold
        for gap in isolated:
            assert len(gap["suggested_connections"]) == 0

    def test_similarity_threshold_filters_suggestions(self, store: Store):
        """Higher min_similarity should filter out weakly related suggestions."""
        u1 = store.insert_unit(_make_unit("n1", "A"))
        u2 = store.insert_unit(_make_unit("n2", "B"))

        # Moderate similarity
        store.update_embedding(u1.id, _make_embedding(1.0, 0.0, 0.0))
        store.update_embedding(u2.id, _make_embedding(0.6, 0.8, 0.0))

        sim = Store._cosine_similarity([1.0, 0.0, 0.0], [0.6, 0.8, 0.0])

        # Low threshold: should have suggestions
        gaps_low = store.identify_knowledge_gaps(min_similarity=0.3)
        isolated_low = [g for g in gaps_low if g["gap_type"] == "isolated_unit"]
        has_suggestions_low = any(len(g["suggested_connections"]) > 0 for g in isolated_low)

        # High threshold: might not have suggestions
        gaps_high = store.identify_knowledge_gaps(min_similarity=0.9)
        isolated_high = [g for g in gaps_high if g["gap_type"] == "isolated_unit"]
        has_suggestions_high = any(len(g["suggested_connections"]) > 0 for g in isolated_high)

        if sim >= 0.3:
            assert has_suggestions_low
        if sim < 0.9:
            assert not has_suggestions_high

    def test_gap_dict_structure(self, store: Store):
        """Every gap dict must have the required keys."""
        u = store.insert_unit(_make_unit("n1", "Lonely unit"))
        store.update_embedding(u.id, _make_embedding(1.0, 0.0, 0.0))

        gaps = store.identify_knowledge_gaps()
        assert len(gaps) >= 1
        for gap in gaps:
            assert "gap_type" in gap
            assert gap["gap_type"] in ("isolated_unit", "under_connected_topic", "sparse_region")
            assert "unit_ids" in gap
            assert isinstance(gap["unit_ids"], list)
            assert "suggested_connections" in gap
            assert isinstance(gap["suggested_connections"], list)
            assert "semantic_cluster" in gap
            assert "coverage_score" in gap
            assert isinstance(gap["coverage_score"], float)

    def test_max_connections_parameter(self, store: Store):
        """Varying max_connections changes which units are flagged."""
        u1 = store.insert_unit(_make_unit("n1", "Hub"))
        u2 = store.insert_unit(_make_unit("n2", "Spoke A"))
        u3 = store.insert_unit(_make_unit("n3", "Spoke B"))
        u4 = store.insert_unit(_make_unit("n4", "Spoke C"))

        for u in [u1, u2, u3, u4]:
            store.update_embedding(u.id, _make_embedding(1.0, 0.0, 0.0))

        # u1 connected to u2 and u3 (2 edges)
        store.insert_edge(
            KnowledgeEdge(from_unit_id=u1.id, to_unit_id=u2.id, relation=EdgeRelation.RELATES_TO)
        )
        store.insert_edge(
            KnowledgeEdge(from_unit_id=u1.id, to_unit_id=u3.id, relation=EdgeRelation.RELATES_TO)
        )

        # With max_connections=1: u1 should NOT be under-connected (has 2 edges > 1)
        gaps_strict = store.identify_knowledge_gaps(max_connections=1)
        under_strict = [g for g in gaps_strict if g["gap_type"] == "under_connected_topic"]
        u1_strict = [g for g in under_strict if u1.id in g["unit_ids"]]
        assert len(u1_strict) == 0

        # With max_connections=3: u1 SHOULD be under-connected (has 2 edges <= 3)
        gaps_loose = store.identify_knowledge_gaps(max_connections=3)
        under_loose = [g for g in gaps_loose if g["gap_type"] == "under_connected_topic"]
        u1_loose = [g for g in under_loose if u1.id in g["unit_ids"]]
        assert len(u1_loose) == 1
