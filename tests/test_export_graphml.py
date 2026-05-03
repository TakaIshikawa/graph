from __future__ import annotations

from datetime import datetime, timezone

import networkx as nx

from graph.export.graphml import export_graph_graphml
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def test_export_graphml_writes_valid_xml_with_attributes(tmp_path):
    """GraphML export generates valid XML with node and edge attributes."""
    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 3, 4, 5, 6, tzinfo=timezone.utc)
    edge_created_at = datetime(2024, 1, 4, 5, 6, 7, tzinfo=timezone.utc)

    units = [
        KnowledgeUnit(
            id="unit-a",
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node content",
            content_type=ContentType.FINDING,
            tags=["energy", "solar"],
            metadata={"category": "renewable"},
            confidence=0.82,
            utility_score=0.91,
            created_at=created_at,
            updated_at=updated_at,
        ),
        KnowledgeUnit(
            id="unit-b",
            source_project=SourceProject.MAX,
            source_id="b",
            source_entity_type="insight",
            title="Node B",
            content="Second node content",
            content_type=ContentType.INSIGHT,
            tags=["storage"],
            metadata={"category": "technology"},
            confidence=0.73,
            utility_score=0.64,
            created_at=created_at,
            updated_at=updated_at,
        ),
    ]

    edges = [
        KnowledgeEdge(
            id="edge-a-b",
            from_unit_id="unit-a",
            to_unit_id="unit-b",
            relation=EdgeRelation.BUILDS_ON,
            weight=0.75,
            source=EdgeSource.MANUAL,
            metadata={"reason": "test"},
            created_at=edge_created_at,
        )
    ]

    output_path = tmp_path / "exports" / "graphml" / "graph.graphml"
    stats = export_graph_graphml(units, edges, output_path)

    assert stats == {
        "path": str(output_path),
        "units_exported": 2,
        "edges_exported": 1,
        "skipped_edges": [],
    }

    # Verify the file can be parsed as valid GraphML
    exported = nx.read_graphml(output_path)
    assert set(exported.nodes) == {"unit-a", "unit-b"}

    # Check node attributes
    assert exported.nodes["unit-a"]["title"] == "Node A"
    assert exported.nodes["unit-a"]["source_project"] == "forty_two"
    assert exported.nodes["unit-a"]["source_entity_type"] == "knowledge_node"
    assert exported.nodes["unit-a"]["content"] == "First node content"
    assert exported.nodes["unit-a"]["content_type"] == "finding"
    assert exported.nodes["unit-a"]["tags"] == "energy,solar"
    assert exported.nodes["unit-a"]["metadata"] == '{"category": "renewable"}'
    assert exported.nodes["unit-a"]["created_at"] == created_at.isoformat()
    assert exported.nodes["unit-a"]["updated_at"] == updated_at.isoformat()
    assert float(exported.nodes["unit-a"]["confidence"]) == 0.82
    assert float(exported.nodes["unit-a"]["utility_score"]) == 0.91

    # Check edge attributes
    edge_data = exported.get_edge_data("unit-a", "unit-b")
    assert edge_data is not None
    assert edge_data["relation"] == "builds_on"
    assert edge_data["source"] == "manual"
    assert float(edge_data["weight"]) == 0.75
    assert edge_data["metadata"] == '{"reason": "test"}'
    assert edge_data["created_at"] == edge_created_at.isoformat()


def test_export_graphml_is_deterministic_and_repeatable(tmp_path):
    """GraphML export produces identical output for the same graph."""
    timestamp = datetime(2024, 2, 3, 4, 5, 6, tzinfo=timezone.utc)

    units = [
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
        ),
        KnowledgeUnit(
            id="unit-b",
            source_project=SourceProject.MAX,
            source_id="b",
            source_entity_type="note",
            title="Beta",
            content="Beta content",
            content_type=ContentType.INSIGHT,
            tags=["beta"],
            confidence=0.7,
            utility_score=0.8,
            created_at=timestamp,
            updated_at=timestamp,
        ),
    ]

    edges = [
        KnowledgeEdge(
            id="edge-a-b",
            from_unit_id="unit-a",
            to_unit_id="unit-b",
            relation=EdgeRelation.RELATES_TO,
            weight=0.5,
            source=EdgeSource.INFERRED,
            created_at=timestamp,
        )
    ]

    first_path = tmp_path / "first.graphml"
    second_path = tmp_path / "second.graphml"

    first_stats = export_graph_graphml(units, edges, first_path)
    second_stats = export_graph_graphml(units, edges, second_path)

    assert first_stats["units_exported"] == second_stats["units_exported"] == 2
    assert first_stats["edges_exported"] == second_stats["edges_exported"] == 1
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(
        encoding="utf-8"
    )


def test_export_graphml_handles_empty_graph(tmp_path):
    """GraphML export handles empty graphs and creates parent directories."""
    output_path = tmp_path / "missing" / "parents" / "empty.graphml"
    stats = export_graph_graphml([], [], output_path)

    assert stats == {
        "path": str(output_path),
        "units_exported": 0,
        "edges_exported": 0,
        "skipped_edges": [],
    }

    # Verify the file exists and is valid
    exported = nx.read_graphml(output_path)
    assert exported.number_of_nodes() == 0
    assert exported.number_of_edges() == 0


def test_export_graphml_skips_edges_with_missing_units(tmp_path):
    """GraphML export skips edges referencing non-existent units."""
    timestamp = datetime(2024, 3, 4, 5, 6, 7, tzinfo=timezone.utc)

    units = [
        KnowledgeUnit(
            id="unit-a",
            source_project=SourceProject.MAX,
            source_id="a",
            source_entity_type="note",
            title="Alpha",
            content="Content",
            content_type=ContentType.INSIGHT,
            tags=[],
            created_at=timestamp,
            updated_at=timestamp,
        )
    ]

    edges = [
        KnowledgeEdge(
            id="edge-valid",
            from_unit_id="unit-a",
            to_unit_id="unit-a",
            relation=EdgeRelation.RELATES_TO,
            weight=0.5,
            source=EdgeSource.MANUAL,
            created_at=timestamp,
        ),
        KnowledgeEdge(
            id="edge-missing-target",
            from_unit_id="unit-a",
            to_unit_id="unit-nonexistent",
            relation=EdgeRelation.RELATES_TO,
            weight=0.5,
            source=EdgeSource.MANUAL,
            created_at=timestamp,
        ),
        KnowledgeEdge(
            id="edge-missing-source",
            from_unit_id="unit-nonexistent",
            to_unit_id="unit-a",
            relation=EdgeRelation.RELATES_TO,
            weight=0.5,
            source=EdgeSource.MANUAL,
            created_at=timestamp,
        ),
    ]

    output_path = tmp_path / "graph.graphml"
    stats = export_graph_graphml(units, edges, output_path)

    assert stats["units_exported"] == 1
    assert stats["edges_exported"] == 1
    assert len(stats["skipped_edges"]) == 2

    # Verify skipped edges are properly reported
    skipped_ids = {edge["id"] for edge in stats["skipped_edges"]}
    assert skipped_ids == {"edge-missing-target", "edge-missing-source"}

    # Check the valid edge was exported
    exported = nx.read_graphml(output_path)
    assert exported.number_of_edges() == 1
    assert exported.get_edge_data("unit-a", "unit-a") is not None


def test_export_graphml_preserves_metadata(tmp_path):
    """GraphML export preserves complex metadata structures."""
    timestamp = datetime(2024, 4, 5, 6, 7, 8, tzinfo=timezone.utc)

    units = [
        KnowledgeUnit(
            id="unit-1",
            source_project=SourceProject.MAX,
            source_id="1",
            source_entity_type="test",
            title="Test Unit",
            content="Test content",
            content_type=ContentType.FINDING,
            tags=["tag1", "tag2", "tag3"],
            metadata={
                "author": "Test Author",
                "year": 2024,
                "nested": {"key": "value", "number": 42},
                "list": [1, 2, 3],
            },
            confidence=0.95,
            utility_score=0.88,
            created_at=timestamp,
            updated_at=timestamp,
        )
    ]

    output_path = tmp_path / "metadata.graphml"
    stats = export_graph_graphml(units, [], output_path)

    assert stats["units_exported"] == 1

    exported = nx.read_graphml(output_path)
    metadata_json = exported.nodes["unit-1"]["metadata"]

    # Verify metadata is preserved as JSON
    import json

    metadata = json.loads(metadata_json)
    assert metadata["author"] == "Test Author"
    assert metadata["year"] == 2024
    assert metadata["nested"]["key"] == "value"
    assert metadata["nested"]["number"] == 42
    assert metadata["list"] == [1, 2, 3]


def test_export_graphml_handles_special_characters(tmp_path):
    """GraphML export handles special characters in text fields."""
    timestamp = datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc)

    units = [
        KnowledgeUnit(
            id="unit-special",
            source_project=SourceProject.MAX,
            source_id="special",
            source_entity_type="test",
            title='Title with "quotes" and <tags>',
            content="Content with\nnewlines\tand\ttabs & symbols",
            content_type=ContentType.INSIGHT,
            tags=["tag-with-dash", "tag_with_underscore", "tag/with/slash"],
            confidence=0.5,
            utility_score=0.5,
            created_at=timestamp,
            updated_at=timestamp,
        )
    ]

    output_path = tmp_path / "special.graphml"
    stats = export_graph_graphml(units, [], output_path)

    assert stats["units_exported"] == 1

    # Verify the file is valid and data is preserved
    exported = nx.read_graphml(output_path)
    assert exported.nodes["unit-special"]["title"] == 'Title with "quotes" and <tags>'
    assert (
        exported.nodes["unit-special"]["content"]
        == "Content with\nnewlines\tand\ttabs & symbols"
    )
    assert (
        exported.nodes["unit-special"]["tags"]
        == "tag-with-dash,tag_with_underscore,tag/with/slash"
    )


def test_export_graphml_sorts_units_and_edges_deterministically(tmp_path):
    """GraphML export sorts units and edges for deterministic output."""
    timestamp = datetime(2024, 6, 7, 8, 9, 10, tzinfo=timezone.utc)

    # Create units in non-alphabetical order
    units = [
        KnowledgeUnit(
            id="unit-z",
            source_project=SourceProject.MAX,
            source_id="z",
            source_entity_type="test",
            title="Z",
            content="Z",
            content_type=ContentType.INSIGHT,
            tags=[],
            created_at=timestamp,
            updated_at=timestamp,
        ),
        KnowledgeUnit(
            id="unit-a",
            source_project=SourceProject.MAX,
            source_id="a",
            source_entity_type="test",
            title="A",
            content="A",
            content_type=ContentType.INSIGHT,
            tags=[],
            created_at=timestamp,
            updated_at=timestamp,
        ),
        KnowledgeUnit(
            id="unit-m",
            source_project=SourceProject.MAX,
            source_id="m",
            source_entity_type="test",
            title="M",
            content="M",
            content_type=ContentType.INSIGHT,
            tags=[],
            created_at=timestamp,
            updated_at=timestamp,
        ),
    ]

    # Create edges in non-alphabetical order
    edges = [
        KnowledgeEdge(
            id="edge-z-a",
            from_unit_id="unit-z",
            to_unit_id="unit-a",
            relation=EdgeRelation.RELATES_TO,
            weight=0.5,
            source=EdgeSource.MANUAL,
            created_at=timestamp,
        ),
        KnowledgeEdge(
            id="edge-a-m",
            from_unit_id="unit-a",
            to_unit_id="unit-m",
            relation=EdgeRelation.RELATES_TO,
            weight=0.5,
            source=EdgeSource.MANUAL,
            created_at=timestamp,
        ),
    ]

    first_path = tmp_path / "sorted1.graphml"
    second_path = tmp_path / "sorted2.graphml"

    # Export twice with different input order
    export_graph_graphml(units, edges, first_path)
    export_graph_graphml(list(reversed(units)), list(reversed(edges)), second_path)

    # Output should be identical
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(
        encoding="utf-8"
    )
