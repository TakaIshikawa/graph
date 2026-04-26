"""Tests for the GraphService."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest
import yaml

from graph.graph.service import GraphService
from graph.rag.embeddings import serialize_embedding
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

    def test_delete_edges_bulk_dry_run_confirm_and_endpoint_summaries(
        self, populated_store: Store
    ):
        gs = GraphService(populated_store)

        dry_run = gs.delete_edges_bulk(
            relation="builds_on",
            source_project="forty_two",
            dry_run=True,
        )
        assert dry_run["dry_run"] is True
        assert dry_run["matched_count"] == 1
        assert dry_run["deleted_count"] == 0
        assert dry_run["edges"][0]["from_unit"]["title"] == "Node A"
        assert dry_run["edges"][0]["to_unit"]["title"] == "Node B"
        assert len(populated_store.get_all_edges()) == 2

        blocked = gs.delete_edges_bulk(
            relation="builds_on",
            source_project="forty_two",
            dry_run=False,
            confirm=False,
        )
        assert blocked["error"] == "confirmation_required"
        assert len(populated_store.get_all_edges()) == 2

        deleted = gs.delete_edges_bulk(
            relation="builds_on",
            source_project="forty_two",
            dry_run=False,
            confirm=True,
        )
        assert deleted["matched_count"] == 1
        assert deleted["deleted_count"] == 1
        assert len(populated_store.get_all_edges()) == 1

    def test_export_markdown_folder_filters_front_matter_collisions_and_clean(
        self, store: Store, tmp_path
    ):
        first = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="same-a",
                source_entity_type="insight",
                title="Same/Title?",
                content="First insight",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar"],
                metadata={"nested": {"url": "https://example.test/a"}},
                confidence=0.8,
                utility_score=0.9,
            )
        )
        second = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="same-b",
                source_entity_type="insight",
                title="Same/Title?",
                content="Second insight",
                content_type=ContentType.INSIGHT,
                tags=["energy"],
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id="other",
                source_entity_type="knowledge_item",
                title="Other",
                content="Other artifact",
                content_type=ContentType.ARTIFACT,
                tags=["energy"],
            )
        )
        output_dir = tmp_path / "markdown"
        output_dir.mkdir()
        (output_dir / "stale.md").write_text("old")
        keep_dir = output_dir / "old-subdir"
        keep_dir.mkdir()
        (keep_dir / "old.txt").write_text("old")

        stats = GraphService(store).export_markdown_folder(
            output_dir,
            clean=True,
            tag="energy",
            source_project="max",
            content_type="insight",
        )

        assert stats["path"] == str(output_dir)
        assert stats["units_exported"] == 2
        assert stats["files_written"] == 2
        assert stats["filters"] == {
            "tag": "energy",
            "source_project": "max",
            "content_type": "insight",
        }
        assert not (output_dir / "stale.md").exists()
        assert not keep_dir.exists()

        files = sorted(output_dir.glob("*.md"))
        assert len(files) == 2
        assert len({path.name for path in files}) == 2
        assert all(path.name.startswith("same-title--") for path in files)

        front_matters = {}
        for path in files:
            text = path.read_text()
            assert text.startswith("---\n")
            raw_front_matter = text.split("---", 2)[1]
            data = yaml.safe_load(raw_front_matter)
            front_matters[data["id"]] = data

        assert set(front_matters) == {first.id, second.id}
        first_front_matter = front_matters[first.id]
        assert first_front_matter["source_project"] == "max"
        assert first_front_matter["source_id"] == "same-a"
        assert first_front_matter["source_entity_type"] == "insight"
        assert first_front_matter["content_type"] == "insight"
        assert first_front_matter["tags"] == ["energy", "solar"]
        assert first_front_matter["confidence"] == 0.8
        assert first_front_matter["utility_score"] == 0.9
        assert first_front_matter["created_at"]
        assert first_front_matter["updated_at"]
        assert first_front_matter["metadata"] == {
            "nested": {"url": "https://example.test/a"}
        }

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

    def test_get_ego_metrics_counts_incident_edges_and_caps_depth(
        self, populated_store: Store
    ):
        gs = GraphService(populated_store)
        gs.rebuild()

        b = populated_store.get_unit_by_source("forty_two", "b", "knowledge_node")
        result = gs.get_ego_metrics(b.id, depth=9)

        assert result["center"]["title"] == "Node B"
        assert result["depth"] == 3
        assert result["metrics"]["degree"] == 2
        assert result["metrics"]["in_degree"] == 1
        assert result["metrics"]["out_degree"] == 1
        assert result["metrics"]["reachable_neighbor_count"] == 2
        assert result["metrics"]["local_clustering_coefficient"] == 0
        assert result["metrics"]["bridge_score"] > 0
        assert result["relation_counts"] == {"builds_on": 1, "inspires": 1}

    def test_get_ego_metrics_missing_unit_returns_empty_payload(self, populated_store: Store):
        result = GraphService(populated_store).get_ego_metrics("missing", depth=0)

        assert result["center"] is None
        assert result["metrics"] == {}
        assert result["relation_counts"] == {}
        assert result["depth"] == 1
        assert result["error"] == "unit_not_found"

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

    def test_get_backlinks_returns_incoming_source_summaries_with_filters(
        self, store: Store
    ):
        target = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="target",
                source_entity_type="knowledge_node",
                title="Target",
                content="Target content",
            )
        )
        source = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="source",
                source_entity_type="insight",
                title="Source title",
                content="Source content",
                content_type=ContentType.INSIGHT,
                tags=["backlink"],
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=source.id,
                to_unit_id=target.id,
                relation=EdgeRelation.REFERENCES,
                weight=0.8,
            )
        )

        payload = GraphService(store).get_backlinks(
            target.id,
            relation="references",
            source_project="max",
            content_type="insight",
            tag="backlink",
        )

        assert payload["center"]["title"] == "Target"
        assert payload["links"][0]["relation"] == "references"
        assert payload["links"][0]["source_unit"]["title"] == "Source title"
        assert payload["links"][0]["edge"]["weight"] == 0.8

    def test_get_backlinks_missing_unit_returns_error(self, store: Store):
        payload = GraphService(store).get_backlinks("missing")

        assert payload["error"] == "unit_not_found"
        assert payload["message"] == "Unit not found: missing"

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

    def test_export_cytoscape_writes_elements_json_and_caps_whole_graph(
        self, populated_store: Store, tmp_path
    ):
        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        b = populated_store.get_unit_by_source("forty_two", "b", "knowledge_node")
        output_path = tmp_path / "graph.cy.json"

        gs = GraphService(populated_store)
        stats = gs.export_cytoscape(output_path, limit=4)
        payload = json.loads(output_path.read_text())

        assert stats == {
            "path": str(output_path),
            "node_count": 4,
            "edge_count": 2,
            "mode": "whole_graph",
            "capped": False,
        }
        assert set(payload["elements"]) == {"nodes", "edges"}
        node_data = {node["data"]["id"]: node["data"] for node in payload["elements"]["nodes"]}
        assert a.id in node_data
        assert b.id in node_data
        assert node_data[a.id]["label"] == "Node A"
        assert node_data[a.id]["title"] == "Node A"
        assert node_data[a.id]["source_project"] == "forty_two"
        assert node_data[a.id]["content_type"] == "insight"
        assert node_data[a.id]["tags"] == []
        assert node_data[a.id]["utility_score"] == 0.9
        assert node_data[a.id]["confidence"] is None
        assert node_data[a.id]["created_at"]
        assert node_data[a.id]["updated_at"]

        assert len(payload["elements"]["edges"]) == 2
        edge_data = next(
            edge["data"]
            for edge in payload["elements"]["edges"]
            if edge["data"]["relation"] == "builds_on"
        )
        assert edge_data["id"]
        assert edge_data["source"] == a.id
        assert edge_data["target"] == b.id
        assert edge_data["relation"] == "builds_on"
        assert edge_data["weight"] == 0.75
        assert edge_data["edge_source"] == "manual"
        assert edge_data["created_at"]

        capped_stats = gs.export_cytoscape(tmp_path / "capped.cy.json", limit=2)
        assert capped_stats["node_count"] == 2
        assert capped_stats["capped"] is True

    def test_export_cytoscape_neighborhood_respects_unit_and_depth_cap(
        self, populated_store: Store, tmp_path
    ):
        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        d = populated_store.get_unit_by_source("presence", "d", "knowledge_item")
        output_path = tmp_path / "neighborhood.cy.json"

        gs = GraphService(populated_store)
        gs.rebuild()
        stats = gs.export_cytoscape(output_path, unit_id=a.id, depth=9)
        payload = json.loads(output_path.read_text())

        assert stats == {
            "path": str(output_path),
            "node_count": 3,
            "edge_count": 2,
            "mode": "neighborhood",
            "capped": False,
            "depth": 3,
            "center_unit_id": a.id,
        }
        node_ids = {node["data"]["id"] for node in payload["elements"]["nodes"]}
        assert d.id not in node_ids
        assert {edge["data"]["relation"] for edge in payload["elements"]["edges"]} == {
            "builds_on",
            "inspires",
        }

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

    def test_find_orphan_units_excludes_incoming_and_outgoing_and_filters(
        self, store: Store
    ):
        source = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="source",
                source_entity_type="knowledge_node",
                title="Source node",
                content="Has outgoing edge",
                content_type=ContentType.FINDING,
                tags=["energy"],
            )
        )
        target = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="target",
                source_entity_type="knowledge_node",
                title="Target node",
                content="Has incoming edge",
                content_type=ContentType.FINDING,
                tags=["energy"],
            )
        )
        max_orphan = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="max-orphan",
                source_entity_type="insight",
                title="Max orphan",
                content="No edges",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar"],
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id="presence-orphan",
                source_entity_type="knowledge_item",
                title="Presence orphan",
                content="No edges",
                content_type=ContentType.ARTIFACT,
                tags=["archive"],
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=source.id,
                to_unit_id=target.id,
                relation=EdgeRelation.BUILDS_ON,
            )
        )

        result = GraphService(store).find_orphan_units(
            source_project="max",
            content_type="insight",
            tag="energy",
            limit=1,
        )

        assert result["total_count"] == 1
        assert result["returned_count"] == 1
        assert result["filters"] == {
            "source_project": "max",
            "content_type": "insight",
            "tag": "energy",
            "limit": 1,
        }
        assert [unit["id"] for unit in result["units"]] == [max_orphan.id]

        all_orphans = GraphService(store).find_orphan_units(limit=10)
        assert all_orphans["total_count"] == 2
        assert {unit["title"] for unit in all_orphans["units"]} == {
            "Max orphan",
            "Presence orphan",
        }

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

    def test_stats_snapshot_counts_units_edges_embeddings_and_degrees(
        self, populated_store: Store
    ):
        a = populated_store.get_unit_by_source("forty_two", "a", "knowledge_node")
        b = populated_store.get_unit_by_source("forty_two", "b", "knowledge_node")
        populated_store.update_unit_fields(a.id, tags=["alpha", "shared"])
        populated_store.update_unit_fields(b.id, tags=["shared"])
        populated_store.update_embedding(a.id, serialize_embedding([1.0, 0.0]))

        gs = GraphService(populated_store)
        gs.rebuild()
        snapshot = gs.stats_snapshot()

        assert set(snapshot) == {
            "unit_counts",
            "edge_counts",
            "embedding_counts",
            "isolated_count",
            "top_degree_units",
        }
        assert snapshot["unit_counts"] == {
            "total": 4,
            "by_source_project": {"forty_two": 2, "max": 1, "presence": 1},
            "by_content_type": {"artifact": 1, "insight": 3},
            "by_tag": {"alpha": 1, "shared": 2},
        }
        assert snapshot["edge_counts"] == {
            "total": 2,
            "by_relation": {"builds_on": 1, "inspires": 1},
            "by_source": {"inferred": 1, "manual": 1},
        }
        assert snapshot["embedding_counts"] == {
            "with_embeddings": 1,
            "without_embeddings": 3,
        }
        assert snapshot["isolated_count"] == 1
        assert snapshot["top_degree_units"][0]["id"] == b.id
        assert snapshot["top_degree_units"][0]["degree"] == 2
        assert snapshot["top_degree_units"][0]["in_degree"] == 1
        assert snapshot["top_degree_units"][0]["out_degree"] == 1

    def test_empty_graph(self, store: Store):
        gs = GraphService(store)
        gs.rebuild()
        assert gs.stats()["nodes"] == 0
        assert gs.stats_snapshot() == {
            "unit_counts": {
                "total": 0,
                "by_source_project": {},
                "by_content_type": {},
                "by_tag": {},
            },
            "edge_counts": {"total": 0, "by_relation": {}, "by_source": {}},
            "embedding_counts": {"with_embeddings": 0, "without_embeddings": 0},
            "isolated_count": 0,
            "top_degree_units": [],
        }
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

    def test_tag_graph_returns_deterministic_nodes_edges_and_representatives(
        self, tagged_store: Store
    ):
        gs = GraphService(tagged_store)
        result = gs.tag_graph(min_count=2, limit=10)

        assert result["nodes"] == [
            {"id": "energy", "tag": "energy", "unit_count": 3},
            {"id": "solar", "tag": "solar", "unit_count": 2},
            {"id": "storage", "tag": "storage", "unit_count": 2},
        ]
        assert [
            (edge["source"], edge["target"], edge["co_occurrence_count"])
            for edge in result["edges"]
        ] == [
            ("energy", "solar", 2),
            ("energy", "storage", 2),
        ]
        assert all(edge["representative_unit_ids"] for edge in result["edges"])
        assert result["filters"] == {
            "source_project": None,
            "content_type": None,
            "min_count": 2,
            "limit": 10,
        }

    def test_tag_graph_applies_filters_and_limit(self, tagged_store: Store):
        gs = GraphService(tagged_store)
        result = gs.tag_graph(
            source_project="max",
            content_type="insight",
            min_count=2,
            limit=1,
        )

        assert result["nodes"] == [
            {"id": "energy", "tag": "energy", "unit_count": 2},
            {"id": "storage", "tag": "storage", "unit_count": 2},
        ]
        assert [
            (edge["source"], edge["target"], edge["co_occurrence_count"])
            for edge in result["edges"]
        ] == [("energy", "storage", 2)]

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

    def test_suggest_tags_uses_existing_elsewhere_excludes_assigned_and_is_sorted(
        self, store: Store
    ):
        target = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="target",
                source_entity_type="insight",
                title="Solar battery planning",
                content="Battery storage roadmap for grid operations.",
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="battery",
                source_entity_type="insight",
                title="Battery note",
                content="Existing battery note",
                tags=["battery", "energy"],
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="storage",
                source_entity_type="knowledge_node",
                title="Storage note",
                content="Existing storage note",
                tags=["energy", "storage"],
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id="writing",
                source_entity_type="knowledge_item",
                title="Writing",
                content="Unrelated note",
                tags=["writing"],
            )
        )

        before = {unit.id: list(unit.tags) for unit in store.get_all_units(limit=100)}
        result = GraphService(store).suggest_tags(target.id, limit=10, min_score=0.25)

        assert result["unit_id"] == target.id
        assert [item["tag"] for item in result["suggestions"]] == [
            "battery",
            "storage",
        ]
        assert "solar" not in {item["tag"] for item in result["suggestions"]}
        assert "writing" not in {item["tag"] for item in result["suggestions"]}
        assert result["suggestions"][0]["score"] >= result["suggestions"][1]["score"]
        assert result["suggestions"][0]["reasons"]
        after = {unit.id: list(unit.tags) for unit in store.get_all_units(limit=100)}
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

    def test_extract_references_dry_run_insert_and_skip_counts(self, store: Store):
        target = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="target",
                source_entity_type="insight",
                title="Canonical target",
                content="target content",
                content_type=ContentType.ARTIFACT,
                metadata={"canonical_url": "https://example.com/target"},
            )
        )
        source = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="source",
                source_entity_type="insight",
                title="Source",
                content="See https://example.com/target for details.",
                metadata={"note": "Also saved at https://example.com/ignored"},
                content_type=ContentType.INSIGHT,
            )
        )
        duplicate_source = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="duplicate-source",
                source_entity_type="insight",
                title="Duplicate source",
                content="Already references https://example.com/target.",
                content_type=ContentType.INSIGHT,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=duplicate_source.id,
                to_unit_id=target.id,
                relation=EdgeRelation.REFERENCES,
                source=EdgeSource.MANUAL,
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="self",
                source_entity_type="insight",
                title="Self reference",
                content="Self link https://example.com/self.",
                metadata={"url": "https://example.com/self"},
                content_type=ContentType.INSIGHT,
            )
        )
        for suffix in ("a", "b"):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"ambiguous-{suffix}",
                    source_entity_type="insight",
                    title=f"Ambiguous {suffix}",
                    content="ambiguous target",
                    content_type=ContentType.ARTIFACT,
                    metadata={"url": "https://example.com/ambiguous"},
                )
            )
        ambiguous_source = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="ambiguous-source",
                source_entity_type="insight",
                title="Ambiguous source",
                content="Ambiguous link https://example.com/ambiguous.",
                content_type=ContentType.INSIGHT,
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="filtered",
                source_entity_type="knowledge_node",
                title="Filtered source",
                content="Filtered mention https://example.com/target.",
                content_type=ContentType.INSIGHT,
            )
        )

        dry_run = GraphService(store).extract_references(
            dry_run=True,
            source_project="max",
            content_type="insight",
        )

        assert dry_run["inserted"] == 0
        assert dry_run["would_insert"] == 1
        assert dry_run["skipped_duplicates"] == 1
        assert dry_run["skipped_self"] == 1
        assert dry_run["skipped_ambiguous"] == 1
        assert dry_run["filters"] == {"source_project": "max", "content_type": "insight"}
        assert len(store.get_all_edges()) == 1
        statuses = {candidate["status"] for candidate in dry_run["candidates"]}
        assert {
            "would_insert",
            "skipped_duplicate",
            "skipped_self_reference",
            "skipped_ambiguous_match",
        } <= statuses

        result = GraphService(store).extract_references(
            source_project="max",
            content_type="insight",
        )

        assert result["inserted"] == 1
        assert result["skipped_duplicates"] == 1
        assert result["skipped_self"] == 1
        assert result["skipped_ambiguous"] == 1
        inserted = [edge for edge in store.get_all_edges() if edge.source == EdgeSource.INFERRED]
        assert len(inserted) == 1
        assert inserted[0].from_unit_id == source.id
        assert inserted[0].to_unit_id == target.id
        assert inserted[0].relation == EdgeRelation.REFERENCES
        assert inserted[0].metadata["url"] == "https://example.com/target"
        assert all(
            candidate["from_unit_id"] != ambiguous_source.id
            or candidate["status"] == "skipped_ambiguous_match"
            for candidate in result["candidates"]
        )

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

    def test_remove_tag_dry_run_and_execute_return_same_sample_schema(self, store: Store):
        unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="service-remove",
                source_entity_type="insight",
                title="Service remove tag",
                content="Service removable content",
                content_type=ContentType.INSIGHT,
                tags=["retire_tag", "keep"],
            )
        )
        filtered = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="service-remove-filtered",
                source_entity_type="knowledge_node",
                title="Service remove filtered",
                content="Filtered removable content",
                content_type=ContentType.FINDING,
                tags=["retire_tag"],
            )
        )
        store.fts_index_unit(unit)
        store.fts_index_unit(filtered)
        gs = GraphService(store)

        dry_run = gs.remove_tag(
            "retire_tag",
            dry_run=True,
            source_project="max",
            content_type="insight",
            limit=10,
        )

        assert dry_run["dry_run"] is True
        assert dry_run["matched_count"] == 1
        assert dry_run["changed_count"] == 1
        assert dry_run["sample_units"] == dry_run["changed_units"]
        assert store.get_unit(unit.id).tags == ["retire_tag", "keep"]  # type: ignore[union-attr]

        result = gs.remove_tag(
            "retire_tag",
            source_project="max",
            content_type="insight",
            limit=10,
        )

        assert result["dry_run"] is False
        assert result["matched_count"] == dry_run["matched_count"]
        assert result["changed_count"] == dry_run["changed_count"]
        assert result["sample_units"][0]["old_tags"] == ["retire_tag", "keep"]
        assert result["sample_units"][0]["new_tags"] == ["keep"]
        assert store.get_unit(unit.id).tags == ["keep"]  # type: ignore[union-attr]
        assert store.get_unit(filtered.id).tags == ["retire_tag"]  # type: ignore[union-attr]
        assert unit.id not in {row["unit_id"] for row in store.fts_search("retire_tag")}

    def test_analyze_duplicates_reports_candidate_groups_and_remains_read_only(
        self, store: Store
    ):
        units = [
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="canonical-a",
                source_entity_type="insight",
                title="Canonical A",
                content="Canonical duplicate A.",
                content_type=ContentType.INSIGHT,
                metadata={"canonical_url": "HTTPS://Example.com/Shared"},
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="canonical-b",
                source_entity_type="insight",
                title="Canonical B",
                content="Canonical duplicate B.",
                content_type=ContentType.INSIGHT,
                metadata={"canonical_url": "https://example.com/Shared"},
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=" Link Import ",
                source_entity_type="insight",
                title="Link duplicate A",
                content="Link duplicate A.",
                content_type=ContentType.INSIGHT,
                metadata={"link": "https://example.com/link"},
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="link-import",
                source_entity_type="insight",
                title="Link duplicate B",
                content="Link duplicate B.",
                content_type=ContentType.INSIGHT,
                metadata={"link": "https://example.com/link"},
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="title-a",
                source_entity_type="insight",
                title="Solar storage roadmap",
                content="Title duplicate A.",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="title-b",
                source_entity_type="insight",
                title="Solar storage road map",
                content="Title duplicate B.",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="filtered-title",
                source_entity_type="knowledge_node",
                title="Solar storage roadmap",
                content="Outside project duplicate title.",
                content_type=ContentType.FINDING,
            ),
        ]
        for unit in units:
            store.insert_unit(unit)
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=store.get_unit_by_source("max", "canonical-a", "insight").id,
                to_unit_id=store.get_unit_by_source("max", "title-a", "insight").id,
                relation=EdgeRelation.RELATES_TO,
                source=EdgeSource.MANUAL,
            )
        )
        store.update_embedding(
            store.get_unit_by_source("max", "canonical-a", "insight").id,
            serialize_embedding([0.1, 0.2, 0.3]),
        )
        before = {
            "units": store.count_units(),
            "edges": len(store.get_all_edges()),
            "embeddings": len(store.get_units_with_embeddings()),
            "fts": len(store.fts_search("Canonical")),
        }

        gs = GraphService(store)
        result = gs.analyze_duplicates(
            source_project="max",
            content_type="insight",
            min_title_similarity=0.85,
        )

        after = {
            "units": store.count_units(),
            "edges": len(store.get_all_edges()),
            "embeddings": len(store.get_units_with_embeddings()),
            "fts": len(store.fts_search("Canonical")),
        }
        assert after == before

        assert result["groups"] == result["results"]
        by_reason = {item["reason"]: item for item in result["groups"]}
        assert set(by_reason) == {
            "canonical_url",
            "link",
            "title_similarity",
        }
        assert all(item["id"].startswith("dup_") for item in result["groups"])
        assert by_reason["canonical_url"]["reasons"] == ["canonical_url", "title_similarity"]
        assert by_reason["canonical_url"]["score"] == 1.0
        assert {unit["source_id"] for unit in by_reason["canonical_url"]["units"]} == {
            "canonical-a",
            "canonical-b",
        }
        assert by_reason["link"]["reasons"] == ["link", "source_identity", "title_similarity"]
        assert {unit["source_id"] for unit in by_reason["link"]["units"]} == {
            " Link Import ",
            "link-import",
        }
        assert {unit["source_id"] for unit in by_reason["title_similarity"]["units"]} == {
            "title-a",
            "title-b",
        }
        assert by_reason["title_similarity"]["score"] >= 0.85
        assert all(
            unit["source_project"] == "max"
            for item in result["groups"]
            for unit in item["units"]
        )

        stricter = gs.analyze_duplicates(
            source_project="max",
            content_type="insight",
            min_title_similarity=0.99,
        )
        assert "title_similarity" not in {item["reason"] for item in stricter["groups"]}

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
