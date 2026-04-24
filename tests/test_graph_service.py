"""Tests for the GraphService."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


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
            weight=0.75,
            source=EdgeSource.MANUAL,
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


@pytest.fixture
def tagged_store(store: Store):
    """Store with overlapping tags across projects and content types."""
    units = [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="solar-storage",
            source_entity_type="insight",
            title="Solar storage",
            content="Solar storage insight",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar", "storage"],
            utility_score=0.9,
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="solar-grid",
            source_entity_type="knowledge_node",
            title="Solar grid",
            content="Solar grid finding",
            content_type=ContentType.FINDING,
            tags=["energy", "solar", "grid"],
            utility_score=0.7,
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="battery-storage",
            source_entity_type="insight",
            title="Battery storage",
            content="Battery storage insight",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage", "battery"],
            utility_score=0.8,
        ),
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="writing",
            source_entity_type="knowledge_item",
            title="Writing workflow",
            content="Writing artifact",
            content_type=ContentType.ARTIFACT,
            tags=["writing"],
        ),
    ]
    for unit in units:
        store.insert_unit(unit)
    return store


@pytest.fixture
def tag_synonym_store(store: Store):
    """Store with variant tags that should suggest a canonical form."""
    units = [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="agent-hyphen",
            source_entity_type="insight",
            title="Agent hyphen",
            content="Agent note",
            content_type=ContentType.INSIGHT,
            tags=["ai-agent", "workflow"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="agent-underscore",
            source_entity_type="insight",
            title="Agent underscore",
            content="Agent note",
            content_type=ContentType.INSIGHT,
            tags=["ai_agent"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="agent-plural",
            source_entity_type="knowledge_node",
            title="Agent plural",
            content="Agent note",
            content_type=ContentType.FINDING,
            tags=["AI Agents"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="unrelated",
            source_entity_type="knowledge_item",
            title="Unrelated",
            content="Unrelated note",
            content_type=ContentType.ARTIFACT,
            tags=["storage", "writing"],
        ),
    ]
    for unit in units:
        store.insert_unit(unit)
    return store


@pytest.fixture
def external_link_store(store: Store):
    """Store with external links in content and metadata."""
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="markdown-link",
            source_entity_type="insight",
            title="Markdown citation",
            content="Read [the docs](https://Example.com/docs). Also see https://other.test/path,",
            content_type=ContentType.INSIGHT,
            metadata={"source_url": "https://meta.test/report)."},
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="duplicate-link",
            source_entity_type="knowledge_node",
            title="Repeated citation",
            content="Same citation https://example.com/docs and nested metadata.",
            content_type=ContentType.FINDING,
            metadata={"refs": {"homepage": "https://example.com/home"}},
        ),
    ]:
        store.insert_unit(unit)
    return store


@pytest.fixture
def timeline_store(store: Store):
    units = [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="jan-solar",
            source_entity_type="insight",
            title="January solar",
            content="Solar storage",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar"],
            created_at=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="feb-grid",
            source_entity_type="knowledge_node",
            title="February grid",
            content="Grid finding",
            content_type=ContentType.FINDING,
            tags=["energy", "grid"],
            created_at=datetime.fromisoformat("2026-02-05T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="feb-battery",
            source_entity_type="insight",
            title="February battery",
            content="Battery storage",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage"],
            created_at=datetime.fromisoformat("2026-02-20T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="mar-writing",
            source_entity_type="knowledge_item",
            title="March writing",
            content="Writing artifact",
            content_type=ContentType.ARTIFACT,
            tags=["writing"],
            created_at=datetime.fromisoformat("2026-03-01T10:00:00+00:00"),
        ),
    ]
    inserted = [store.insert_unit(unit) for unit in units]
    store.conn.execute(
        "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
        ("2026-04-01T00:00:00+00:00", inserted[0].id),
    )
    store.conn.execute(
        "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
        ("2026-04-02T00:00:00+00:00", inserted[1].id),
    )
    store.conn.commit()
    return store


class TestGraphService:
    def test_rebuild(self, populated_store: Store):
        gs = GraphService(populated_store)
        count = gs.rebuild()
        assert count == 4

    def test_analyze_timeline_month_buckets_are_chronological(self, timeline_store: Store):
        result = GraphService(timeline_store).analyze_timeline(bucket="month")

        assert result["total"] == 4
        assert [item["bucket"] for item in result["buckets"]] == [
            "2026-01",
            "2026-02",
            "2026-03",
        ]
        assert [item["count"] for item in result["buckets"]] == [1, 2, 1]
        february = result["buckets"][1]
        assert february["source_projects"] == {"forty_two": 1, "max": 1}
        assert february["content_types"] == {"finding": 1, "insight": 1}
        assert february["top_tags"][0] == {"tag": "energy", "count": 2}

    def test_analyze_timeline_filters_and_date_range_change_totals(self, timeline_store: Store):
        result = GraphService(timeline_store).analyze_timeline(
            bucket="month",
            start="2026-02-01",
            end="2026-02-28T23:59:59+00:00",
            source_project="max",
            content_type="insight",
            tag="storage",
        )

        assert result["total"] == 1
        assert [item["bucket"] for item in result["buckets"]] == ["2026-02"]
        assert result["buckets"][0]["source_projects"] == {"max": 1}
        assert result["buckets"][0]["content_types"] == {"insight": 1}

    def test_analyze_timeline_uses_requested_field_and_limit(self, timeline_store: Store):
        result = GraphService(timeline_store).analyze_timeline(
            bucket="day",
            field="ingested_at",
            start="2026-04-01",
            end="2026-04-02T23:59:59+00:00",
            limit=1,
        )

        assert result["field"] == "ingested_at"
        assert result["total"] == 2
        assert [item["bucket"] for item in result["buckets"]] == ["2026-04-01"]

    def test_analyze_timeline_empty_graph(self, store: Store):
        result = GraphService(store).analyze_timeline()

        assert result["total"] == 0
        assert result["buckets"] == []

    def test_analyze_timeline_validates_bucket_and_field(self, store: Store):
        gs = GraphService(store)

        with pytest.raises(ValueError, match="Unsupported timeline bucket"):
            gs.analyze_timeline(bucket="quarter")
        with pytest.raises(ValueError, match="Unsupported timeline field"):
            gs.analyze_timeline(field="deleted_at")

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

    def test_build_shortest_path_payload_includes_ordered_units_and_edges(
        self, populated_store: Store
    ):
        gs = GraphService(populated_store)
        gs.rebuild()

        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        b = populated_store.get_unit_by_source("forty_two", "b", "knowledge_node")
        c = populated_store.get_unit_by_source("max", "c", "insight")

        payload = gs.build_shortest_path_payload(a.id, c.id)

        assert [unit["id"] for unit in payload["path"]] == [a.id, b.id, c.id]
        assert [unit["title"] for unit in payload["path"]] == [
            "Node A",
            "Node B",
            "Node C",
        ]
        assert payload["edges"][0] == {
            "id": payload["edges"][0]["id"],
            "from_unit_id": a.id,
            "to_unit_id": b.id,
            "relation": "builds_on",
            "weight": 0.75,
            "source": "manual",
            "traversal_from_unit_id": a.id,
            "traversal_to_unit_id": b.id,
            "traversal_direction": "forward",
        }
        assert payload["edges"][1]["from_unit_id"] == b.id
        assert payload["edges"][1]["to_unit_id"] == c.id
        assert payload["edges"][1]["relation"] == "inspires"
        assert payload["edges"][1]["weight"] == 1.0
        assert payload["edges"][1]["source"] == "inferred"

    def test_build_shortest_path_payload_reports_missing_and_disconnected_units(
        self, populated_store: Store
    ):
        gs = GraphService(populated_store)
        gs.rebuild()

        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        d = populated_store.get_unit_by_source("presence", "d", "knowledge_item")

        missing = gs.build_shortest_path_payload("missing", d.id)
        assert missing["error"] == "unit_not_found"
        assert missing["missing_unit_ids"] == ["missing"]
        assert missing["path"] == []
        assert missing["edges"] == []

        disconnected = gs.build_shortest_path_payload(a.id, d.id)
        assert disconnected["error"] == "not_connected"
        assert disconnected["path"] == []
        assert disconnected["edges"] == []

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

    def test_export_graphml_writes_scalar_node_and_edge_attributes(
        self, store: Store, tmp_path
    ):
        a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="a",
                source_entity_type="knowledge_node",
                title="Node A",
                content="First node",
                content_type=ContentType.FINDING,
                tags=["energy", "solar"],
                utility_score=0.9,
            )
        )
        b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="b",
                source_entity_type="insight",
                title="Node B",
                content="Second node",
                content_type=ContentType.INSIGHT,
                tags=["storage"],
                utility_score=0.7,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=a.id,
                to_unit_id=b.id,
                relation=EdgeRelation.BUILDS_ON,
                weight=0.75,
            )
        )

        gs = GraphService(store)
        gs.rebuild()
        output_path = tmp_path / "graph.graphml"
        stats = gs.export_graphml(output_path)

        assert stats == {
            "path": str(output_path),
            "node_count": 2,
            "edge_count": 1,
        }
        assert output_path.exists()

        exported = nx.read_graphml(output_path)
        assert exported.nodes[a.id]["title"] == "Node A"
        assert exported.nodes[a.id]["source_project"] == "forty_two"
        assert exported.nodes[a.id]["source_entity_type"] == "knowledge_node"
        assert exported.nodes[a.id]["content_type"] == "finding"
        assert exported.nodes[a.id]["tags"] == "energy,solar"
        assert float(exported.nodes[a.id]["utility_score"]) == 0.9
        assert exported.nodes[a.id]["created_at"]

        edge_data = exported.get_edge_data(a.id, b.id)
        assert edge_data["relation"] == "builds_on"
        assert float(edge_data["weight"]) == 0.75
        assert edge_data["source"] == "inferred"
        assert edge_data["created_at"]

    def test_export_turtle_writes_units_edges_and_escaped_literals(
        self, store: Store, tmp_path
    ):
        a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="a",
                source_entity_type="knowledge_node",
                title='Node "A"\nalpha',
                content='First node with "quoted" content\nand a slash \\ marker',
                content_type=ContentType.FINDING,
                tags=["energy", 'solar "pv"', "line\nbreak"],
                utility_score=0.9,
            )
        )
        b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="b",
                source_entity_type="insight",
                title="Node B",
                content="Second node",
                content_type=ContentType.INSIGHT,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=a.id,
                to_unit_id=b.id,
                relation=EdgeRelation.BUILDS_ON,
            )
        )

        output_path = tmp_path / "graph.ttl"
        stats = GraphService(store).export_turtle(
            output_path, base_uri="https://example.test/unit/"
        )
        text = output_path.read_text()

        assert stats == {
            "path": str(output_path),
            "node_count": 2,
            "edge_count": 1,
            "base_uri": "https://example.test/unit/",
        }
        assert "@prefix graph: <https://graph.local/schema#> ." in text
        assert f"<https://example.test/unit/{a.id}>" in text
        assert f"<https://example.test/unit/{b.id}>" in text
        assert 'graph:title "Node \\"A\\"\\nalpha"' in text
        assert (
            'graph:contentSnippet "First node with \\"quoted\\" content\\nand '
            'a slash \\\\ marker"'
        ) in text
        assert 'graph:tag "solar \\"pv\\""' in text
        assert 'graph:tag "line\\nbreak"' in text
        assert f"graph:builds_on <https://example.test/unit/{b.id}>" in text

    def test_export_mermaid_escapes_labels_and_caps_whole_graph(
        self, store: Store, tmp_path
    ):
        a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="a",
                source_entity_type="knowledge_node",
                title='Node "A" | alpha\nnext',
                content="First node",
            )
        )
        b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="b",
                source_entity_type="insight",
                title="Node B",
                content="Second node",
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id="c",
                source_entity_type="knowledge_item",
                title="Node C",
                content="Third node",
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=a.id,
                to_unit_id=b.id,
                relation=EdgeRelation.BUILDS_ON,
            )
        )

        output_path = tmp_path / "graph.md"
        graph_service = GraphService(store)
        stats = graph_service.export_mermaid(output_path, limit=3)
        text = output_path.read_text()

        assert stats == {
            "path": str(output_path),
            "node_count": 3,
            "edge_count": 1,
            "capped": False,
        }
        assert text.startswith("```mermaid\ngraph TD\n")
        assert '["Node &quot;A&quot; &#124; alpha next"]' in text
        assert "-->|builds_on|" in text
        assert '"A" | alpha\nnext' not in text

        capped_path = tmp_path / "capped.md"
        capped_stats = graph_service.export_mermaid(capped_path, limit=2)
        assert capped_stats["node_count"] == 2
        assert capped_stats["capped"] is True

    def test_export_mermaid_neighborhood_respects_depth_cap_and_limit(
        self, populated_store: Store, tmp_path
    ):
        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        output_path = tmp_path / "neighborhood.md"

        gs = GraphService(populated_store)
        gs.rebuild()
        stats = gs.export_mermaid(output_path, unit_id=a.id, depth=9, limit=2)
        text = output_path.read_text()

        assert stats == {
            "path": str(output_path),
            "node_count": 2,
            "edge_count": 1,
            "capped": True,
            "depth": 3,
            "center_unit_id": a.id,
        }
        assert 'n0["Node A"]' in text
        assert 'n1["Node B"]' in text
        assert "Node C" not in text
        assert "n0 -->|builds_on| n1" in text
        assert "inspires" not in text

    def test_export_neighborhood_writes_capped_local_json(
        self, populated_store: Store, tmp_path
    ):
        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        c = populated_store.get_unit_by_source("max", "c", "insight")
        output_path = tmp_path / "neighborhood.json"

        gs = GraphService(populated_store)
        gs.rebuild()
        stats = gs.export_neighborhood(a.id, output_path, depth=9)
        payload = json.loads(output_path.read_text())

        assert stats == {
            "path": str(output_path),
            "unit_count": 3,
            "edge_count": 2,
            "depth": 3,
            "center_unit_id": a.id,
        }
        assert payload["schema_version"] == 1
        assert payload["exported_at"]
        assert payload["center"]["id"] == a.id
        assert payload["depth"] == 3
        assert {unit["title"] for unit in payload["units"]} == {
            "Node A",
            "Node B",
            "Node C",
        }
        assert {edge["relation"] for edge in payload["edges"]} == {
            "builds_on",
            "inspires",
        }
        assert c.id in {unit["id"] for unit in payload["units"]}

    def test_export_neighborhood_limits_depth_and_reports_missing_unit(
        self, populated_store: Store, tmp_path
    ):
        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        output_path = tmp_path / "depth-one.json"

        gs = GraphService(populated_store)
        gs.rebuild()
        gs.export_neighborhood(a.id, output_path, depth=1)
        payload = json.loads(output_path.read_text())

        assert payload["depth"] == 1
        assert {unit["title"] for unit in payload["units"]} == {"Node A", "Node B"}
        assert [edge["relation"] for edge in payload["edges"]] == ["builds_on"]

        with pytest.raises(ValueError) as exc_info:
            gs.export_neighborhood("missing", tmp_path / "missing.json")
        error = json.loads(str(exc_info.value))
        assert error == {
            "error": "unit_not_found",
            "message": "Unit not found: missing",
            "unit_id": "missing",
        }
        assert not (tmp_path / "missing.json").exists()

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

    def test_analyze_source_coverage_counts_units_edges_sync_and_orphans(
        self, store: Store
    ):
        a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="a",
                source_entity_type="knowledge_node",
                title="Node A",
                content="First node",
                created_at="2026-04-20T00:00:00+00:00",
            )
        )
        b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="b",
                source_entity_type="knowledge_node",
                title="Node B",
                content="Second node",
                created_at="2026-04-21T00:00:00+00:00",
            )
        )
        c = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="c",
                source_entity_type="insight",
                title="Node C",
                content="Third node",
                created_at="2026-04-22T00:00:00+00:00",
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id="d",
                source_entity_type="knowledge_item",
                title="Node D",
                content="Isolated node",
                created_at="2026-04-23T00:00:00+00:00",
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
        store.upsert_sync_state(
            SyncState(
                source_project="forty_two",
                source_entity_type="knowledge_node",
                last_sync_at="2026-04-24T00:00:00+00:00",
                last_source_id="b",
                items_synced=2,
            )
        )
        store.upsert_sync_state(
            SyncState(
                source_project="sota",
                source_entity_type="paper",
                last_sync_at="2026-04-24T01:00:00+00:00",
                last_source_id="paper-1",
                items_synced=5,
            )
        )

        result = GraphService(store).analyze_source_coverage()
        by_source = {
            (item["source_project"], item["source_entity_type"]): item
            for item in result["sources"]
        }

        forty_two = by_source[("forty_two", "knowledge_node")]
        assert forty_two["unit_count"] == 2
        assert forty_two["edge_count"] == 2
        assert forty_two["orphan_count"] == 0
        assert forty_two["oldest_created_at"] == "2026-04-20T00:00:00+00:00"
        assert forty_two["newest_created_at"] == "2026-04-21T00:00:00+00:00"
        assert forty_two["last_sync_at"] == "2026-04-24T00:00:00+00:00"
        assert forty_two["last_source_id"] == "b"
        assert forty_two["items_synced"] == 2
        assert forty_two["has_sync_state"] is True

        max_insight = by_source[("max", "insight")]
        assert max_insight["unit_count"] == 1
        assert max_insight["edge_count"] == 1
        assert max_insight["orphan_count"] == 0

        presence = by_source[("presence", "knowledge_item")]
        assert presence["unit_count"] == 1
        assert presence["edge_count"] == 0
        assert presence["orphan_count"] == 1

        sync_only = by_source[("sota", "paper")]
        assert sync_only["unit_count"] == 0
        assert sync_only["edge_count"] == 0
        assert sync_only["orphan_count"] == 0
        assert sync_only["oldest_created_at"] is None
        assert sync_only["newest_created_at"] is None
        assert sync_only["last_sync_at"] == "2026-04-24T01:00:00+00:00"
        assert sync_only["items_synced"] == 5

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

    def test_analyze_tags_lists_counts_and_breakdowns(self, tagged_store: Store):
        gs = GraphService(tagged_store)
        result = gs.analyze_tags(limit=3)

        assert [item["tag"] for item in result["tags"]] == [
            "energy",
            "solar",
            "storage",
        ]
        energy = result["tags"][0]
        assert energy["count"] == 3
        assert energy["source_projects"] == {"max": 2, "forty_two": 1}
        assert energy["content_types"] == {"insight": 2, "finding": 1}

    def test_analyze_tags_applies_filters(self, tagged_store: Store):
        gs = GraphService(tagged_store)
        result = gs.analyze_tags(source_project="max", content_type="insight")

        counts = {item["tag"]: item["count"] for item in result["tags"]}
        assert counts == {
            "energy": 2,
            "storage": 2,
            "battery": 1,
            "solar": 1,
        }
        assert result["filters"] == {
            "source_project": "max",
            "content_type": "insight",
        }

    def test_analyze_tags_detail_includes_units_and_ordered_co_occurrences(
        self, tagged_store: Store
    ):
        gs = GraphService(tagged_store)
        result = gs.analyze_tags(tag="energy", limit=10)

        assert result["tag"] == "energy"
        assert result["count"] == 3
        assert [item["tag"] for item in result["co_occurring_tags"]] == [
            "solar",
            "storage",
            "battery",
            "grid",
        ]
        assert [item["count"] for item in result["co_occurring_tags"]] == [2, 2, 1, 1]
        assert {unit["title"] for unit in result["units"]} == {
            "Solar storage",
            "Solar grid",
            "Battery storage",
        }

    def test_suggest_tag_synonyms_groups_variants_and_is_read_only(
        self, tag_synonym_store: Store
    ):
        before = {
            unit.source_id: list(unit.tags)
            for unit in tag_synonym_store.get_all_units(limit=100)
        }

        gs = GraphService(tag_synonym_store)
        result = gs.suggest_tag_synonyms(limit=10, min_similarity=0.8)

        assert result["limit"] == 10
        assert result["min_similarity"] == 0.8
        assert len(result["suggestions"]) == 1

        suggestion = result["suggestions"][0]
        assert suggestion["canonical_candidate"] == "ai-agent"
        assert suggestion["total_count"] == 3
        assert suggestion["variant_count"] == 3
        assert suggestion["similarity"] == 1.0
        assert {variant["tag"] for variant in suggestion["variants"]} == {
            "ai-agent",
            "ai_agent",
            "AI Agents",
        }
        assert {variant["count"] for variant in suggestion["variants"]} == {1}
        assert "storage" not in {
            variant["tag"]
            for found in result["suggestions"]
            for variant in found["variants"]
        }

        after = {
            unit.source_id: list(unit.tags)
            for unit in tag_synonym_store.get_all_units(limit=100)
        }
        assert after == before

    def test_suggest_edges_scores_reasons_excludes_existing_and_filters(
        self, store: Store
    ):
        solar = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="solar-storage",
                source_entity_type="insight",
                title="Solar storage roadmap",
                content="Battery grid planning references https://example.com/docs.",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar", "storage"],
            )
        )
        battery = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="battery-storage",
                source_entity_type="insight",
                title="Battery storage plan",
                content="Grid battery storage notes cite https://example.com/docs.",
                content_type=ContentType.INSIGHT,
                tags=["energy", "storage", "battery"],
            )
        )
        grid = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="solar-grid",
                source_entity_type="knowledge_node",
                title="Solar grid plan",
                content="Solar storage grid reference https://example.com/docs.",
                content_type=ContentType.FINDING,
                tags=["energy", "solar", "grid"],
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=grid.id,
                to_unit_id=solar.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )

        result = GraphService(store).suggest_edges(
            limit=10,
            min_score=0.4,
            source_project="max",
        )

        assert result["filters"] == {"source_project": "max"}
        assert len(result["candidates"]) == 1
        candidate = result["candidates"][0]
        assert candidate["from_id"] == battery.id
        assert candidate["to_id"] == solar.id
        assert candidate["score"] >= 0.8
        assert any(reason.startswith("shared tags:") for reason in candidate["reasons"])
        assert any(reason.startswith("shared links:") for reason in candidate["reasons"])
        assert any(
            reason.startswith("title/content token overlap:")
            for reason in candidate["reasons"]
        )
        assert candidate["from_unit"]["source_id"] == "battery-storage"
        assert candidate["to_unit"]["source_id"] == "solar-storage"

        all_projects = GraphService(store).suggest_edges(limit=10, min_score=0.4)
        suggested_pairs = {
            frozenset((item["from_id"], item["to_id"]))
            for item in all_projects["candidates"]
        }
        assert frozenset((solar.id, grid.id)) not in suggested_pairs

        high_threshold = GraphService(store).suggest_edges(limit=10, min_score=1.1)
        assert high_threshold["candidates"] == []

    def test_rename_tag_dry_run_and_execute_return_same_sample_schema(
        self, tag_synonym_store: Store
    ):
        gs = GraphService(tag_synonym_store)

        dry_run = gs.rename_tag(
            "ai_agent",
            "ai-agent",
            dry_run=True,
            source_project="max",
            content_type="insight",
        )

        assert dry_run["dry_run"] is True
        assert dry_run["changed_count"] == 1
        assert dry_run["sample_units"] == dry_run["changed_units"]
        unit = tag_synonym_store.get_unit_by_source("max", "agent-underscore", "insight")
        assert unit is not None
        assert unit.tags == ["ai_agent"]

        result = gs.rename_tag(
            "ai_agent",
            "ai-agent",
            source_project="max",
            content_type="insight",
        )

        assert result["dry_run"] is False
        assert result["changed_count"] == 1
        assert result["sample_units"][0]["old_tags"] == ["ai_agent"]
        assert result["sample_units"][0]["new_tags"] == ["ai-agent"]
        unit = tag_synonym_store.get_unit_by_source("max", "agent-underscore", "insight")
        assert unit is not None
        assert unit.tags == ["ai-agent"]

    def test_analyze_duplicates_finds_titles_content_and_applies_filters(
        self, store: Store
    ):
        for unit in [
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="same-title-a",
                source_entity_type="insight",
                title="Repeated Import",
                content="First independent note about sales workflow cleanup.",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="same-title-b",
                source_entity_type="insight",
                title=" repeated   import ",
                content="Second independent note about onboarding research.",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="same-content-a",
                source_entity_type="insight",
                title="Storage market signal",
                content="Solar storage adoption is accelerating across midmarket operations teams this quarter.",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="same-content-b",
                source_entity_type="insight",
                title="Battery adoption memo",
                content="Solar storage adoption is accelerating across midmarket operations teams this quarter rapidly.",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="unique",
                source_entity_type="insight",
                title="Unique note",
                content="A completely different topic about meeting notes and planning.",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="filtered-title",
                source_entity_type="knowledge_node",
                title="Repeated Import",
                content="Outside project duplicate title.",
                content_type=ContentType.FINDING,
            ),
        ]:
            store.insert_unit(unit)

        gs = GraphService(store)
        result = gs.analyze_duplicates(source_project="max", content_type="insight")

        by_reason = {item["reason"]: item for item in result["results"]}
        assert set(by_reason) == {"same_title", "similar_content"}

        assert by_reason["same_title"]["score"] == 1.0
        assert {unit["source_id"] for unit in by_reason["same_title"]["units"]} == {
            "same-title-a",
            "same-title-b",
        }
        assert {
            unit["source_id"] for unit in by_reason["similar_content"]["units"]
        } == {
            "same-content-a",
            "same-content-b",
        }
        assert by_reason["similar_content"]["score"] >= 0.9
        assert all(
            unit["source_project"] == "max"
            for item in result["results"]
            for unit in item["units"]
        )

    def test_build_review_queue_ranks_old_isolated_unreviewed_and_filters(
        self, store: Store
    ):
        now = datetime.now(timezone.utc)
        old_isolated = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="old-isolated",
                source_entity_type="insight",
                title="Old isolated",
                content="Old unreviewed insight",
                content_type=ContentType.INSIGHT,
                utility_score=0.6,
                created_at=now - timedelta(days=420),
            )
        )
        recent_reviewed = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="recent-reviewed",
                source_entity_type="insight",
                title="Recent reviewed",
                content="Recent reviewed insight",
                content_type=ContentType.INSIGHT,
                metadata={"reviewed_at": now.isoformat()},
                utility_score=1.0,
                created_at=now - timedelta(days=3),
            )
        )
        neighbor = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="neighbor",
                source_entity_type="artifact",
                title="Neighbor",
                content="Connected artifact",
                content_type=ContentType.ARTIFACT,
                created_at=now - timedelta(days=2),
            )
        )
        finding = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="old-finding",
                source_entity_type="knowledge_node",
                title="Old finding",
                content="Old finding",
                content_type=ContentType.FINDING,
                created_at=now - timedelta(days=300),
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=recent_reviewed.id,
                to_unit_id=neighbor.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=finding.id,
                to_unit_id=recent_reviewed.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )

        gs = GraphService(store)
        result = gs.build_review_queue()

        assert result["queue"][0]["unit"]["id"] == old_isolated.id
        assert result["queue"][0]["degree"] == 0
        assert result["queue"][0]["age_days"] >= 419
        reason_codes = {reason["code"] for reason in result["queue"][0]["reasons"]}
        assert {"age", "isolated", "utility_score", "unreviewed"} <= reason_codes

        by_id = {item["unit"]["id"]: item for item in result["queue"]}
        assert by_id[old_isolated.id]["score"] > by_id[recent_reviewed.id]["score"]
        assert any(
            reason["code"] == "reviewed"
            for reason in by_id[recent_reviewed.id]["reasons"]
        )

        filtered = gs.build_review_queue(source_project="max", content_type="insight")
        assert [item["unit"]["id"] for item in filtered["queue"]] == [
            old_isolated.id,
            recent_reviewed.id,
        ]
        assert filtered["filters"] == {
            "source_project": "max",
            "content_type": "insight",
        }

    def test_analyze_links_scans_content_metadata_normalizes_and_filters(
        self, external_link_store: Store
    ):
        gs = GraphService(external_link_store)

        result = gs.analyze_links(limit=10)

        assert result["total_occurrences"] == 5
        assert result["total_urls"] == 4
        assert result["domains"][0]["domain"] == "example.com"
        assert result["domains"][0]["count"] == 3

        links_by_url = {item["url"]: item for item in result["links"]}
        assert links_by_url["https://example.com/docs"]["count"] == 2
        assert {
            occurrence["source_project"]
            for occurrence in links_by_url["https://example.com/docs"]["occurrences"]
        } == {"max", "forty_two"}
        assert "https://Example.com/docs)." not in links_by_url
        assert "https://meta.test/report" in links_by_url
        assert links_by_url["https://meta.test/report"]["occurrences"][0]["field"] == (
            "metadata.source_url"
        )

        filtered = gs.analyze_links(domain="example.com", limit=10)

        assert filtered["filters"] == {"domain": "example.com"}
        assert filtered["total_occurrences"] == 3
        assert {item["domain"] for item in filtered["links"]} == {"example.com"}
        assert {item["url"] for item in filtered["links"]} == {
            "https://example.com/docs",
            "https://example.com/home",
        }


def test_integrity_audit_returns_zero_for_indexed_clean_graph(store: Store):
    first = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="clean-a",
            source_entity_type="insight",
            title="Clean A",
            content="Indexed clean content",
        )
    )
    second = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="clean-b",
            source_entity_type="insight",
            title="Clean B",
            content="Also indexed content",
        )
    )
    store.fts_index_unit(first)
    store.fts_index_unit(second)
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=first.id,
            to_unit_id=second.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )

    result = GraphService(store).integrity_audit()

    assert result["issue_count"] == 0
    assert result["has_issues"] is False
    assert all(category["count"] == 0 for category in result["categories"].values())


def test_integrity_audit_repairs_only_fts_drift(store: Store):
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="missing-fts",
            source_entity_type="insight",
            title="Missing FTS",
            content="Repairable search row",
        )
    )
    store.conn.execute(
        "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
        ("deleted-unit", "Deleted", "stale content", "stale"),
    )
    store.conn.commit()

    result = GraphService(store).integrity_audit(repair_fts=True)

    assert result["repair"] == {
        "fts_rows_inserted": 1,
        "fts_rows_deleted": 1,
        "requested": True,
    }
    assert result["categories"]["units_missing_fts_rows"]["count"] == 0
    assert result["categories"]["stale_fts_rows"]["count"] == 0
    assert store.get_unit(unit.id) is not None
    assert store.fts_search("Repairable")[0]["unit_id"] == unit.id
