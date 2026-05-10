from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph.adapters.tana import TanaAdapter
from graph.types.enums import ContentType, EdgeRelation, SourceProject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_export(
    nodes: list | None = None,
    supertags: list | None = None,
) -> dict:
    export: dict = {}
    if supertags is not None:
        export["supertags"] = supertags
    if nodes is not None:
        export["nodes"] = nodes
    return export


# ---------------------------------------------------------------------------
# Basic node parsing
# ---------------------------------------------------------------------------

def test_parses_basic_nodes(tmp_path):
    data = _minimal_export(nodes=[
        {"id": "n1", "name": "First node"},
        {"id": "n2", "name": "Second node"},
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    assert len(result.units) == 2
    titles = {u.title for u in result.units}
    assert "First node" in titles
    assert "Second node" in titles
    for unit in result.units:
        assert unit.source_project == SourceProject.TANA
        assert unit.source_entity_type == "node"
        assert unit.content_type == ContentType.ARTIFACT


def test_node_with_description(tmp_path):
    data = _minimal_export(nodes=[
        {"id": "n1", "name": "My Node", "description": "Detailed description here."},
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    unit = result.units[0]
    assert "My Node" in unit.content
    assert "Detailed description here." in unit.content


# ---------------------------------------------------------------------------
# Supertags
# ---------------------------------------------------------------------------

def test_parses_supertag_definitions(tmp_path):
    data = _minimal_export(
        supertags=[
            {
                "id": "tag1",
                "name": "Project",
                "description": "A project supertag",
                "fields": [
                    {"name": "Status", "type": "option", "id": "f1"},
                    {"name": "Due Date", "type": "date", "id": "f2"},
                ],
            }
        ],
        nodes=[],
    )
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    tag_units = [u for u in result.units if u.source_entity_type == "supertag"]
    assert len(tag_units) == 1
    tag = tag_units[0]
    assert tag.title == "Project"
    assert tag.content_type == ContentType.METADATA
    assert tag.metadata["tag_name"] == "Project"
    assert len(tag.metadata["fields_schema"]) == 2
    assert tag.metadata["fields_schema"][0]["name"] == "Status"


def test_supertag_applied_to_node(tmp_path):
    data = _minimal_export(
        supertags=[
            {"id": "tag1", "name": "Task"},
        ],
        nodes=[
            {"id": "n1", "name": "Buy groceries", "supertags": ["tag1"]},
        ],
    )
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    node_units = [u for u in result.units if u.source_entity_type == "node"]
    assert len(node_units) == 1
    node = node_units[0]
    assert "Task" in node.metadata["supertags"]
    assert "task" in node.tags


def test_supertag_ref_as_dict(tmp_path):
    data = _minimal_export(
        supertags=[
            {"id": "tag1", "name": "Person"},
        ],
        nodes=[
            {"id": "n1", "name": "Alice", "supertags": [{"id": "tag1"}]},
        ],
    )
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    node = [u for u in result.units if u.source_entity_type == "node"][0]
    assert "Person" in node.metadata["supertags"]


def test_supertag_with_extends(tmp_path):
    data = _minimal_export(
        supertags=[
            {"id": "tag1", "name": "Base", "fields": [{"name": "Name", "type": "text", "id": "f1"}]},
            {"id": "tag2", "name": "Child", "extends": "tag1"},
        ],
        nodes=[],
    )
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    child_tag = [u for u in result.units if u.title == "Child"][0]
    assert child_tag.metadata["extends"] == "tag1"


# ---------------------------------------------------------------------------
# Field values
# ---------------------------------------------------------------------------

def test_extracts_field_values(tmp_path):
    data = _minimal_export(nodes=[
        {
            "id": "n1",
            "name": "Project Alpha",
            "fields": [
                {"name": "Status", "value": "In Progress"},
                {"name": "Priority", "value": "High"},
            ],
        },
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    unit = result.units[0]
    assert unit.metadata["fields"]["Status"] == "In Progress"
    assert unit.metadata["fields"]["Priority"] == "High"


def test_extracts_props_as_fields(tmp_path):
    data = _minimal_export(nodes=[
        {
            "id": "n1",
            "name": "Node with props",
            "props": {"color": "blue", "weight": 42},
        },
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    unit = result.units[0]
    assert unit.metadata["fields"]["color"] == "blue"
    assert unit.metadata["fields"]["weight"] == 42


# ---------------------------------------------------------------------------
# Hierarchies
# ---------------------------------------------------------------------------

def test_parent_child_hierarchy(tmp_path):
    data = _minimal_export(nodes=[
        {
            "id": "parent1",
            "name": "Parent Node",
            "children": [
                {"id": "child1", "name": "Child Node"},
            ],
        },
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    # Should have both parent and child as units
    node_units = [u for u in result.units if u.source_entity_type == "node"]
    assert len(node_units) == 2

    # Should have a CONTAINS edge from parent to child
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.relation == EdgeRelation.CONTAINS
    assert edge.metadata["relation_type"] == "hierarchy"


def test_deep_hierarchy(tmp_path):
    data = _minimal_export(nodes=[
        {
            "id": "root",
            "name": "Root",
            "children": [
                {
                    "id": "mid",
                    "name": "Middle",
                    "children": [
                        {"id": "leaf", "name": "Leaf"},
                    ],
                },
            ],
        },
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    node_units = [u for u in result.units if u.source_entity_type == "node"]
    assert len(node_units) == 3

    contains_edges = [e for e in result.edges if e.relation == EdgeRelation.CONTAINS]
    assert len(contains_edges) == 2


# ---------------------------------------------------------------------------
# Calendar dates
# ---------------------------------------------------------------------------

def test_calendar_date_from_date_field(tmp_path):
    data = _minimal_export(nodes=[
        {"id": "n1", "name": "Meeting", "date": "2024-03-15"},
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    unit = result.units[0]
    assert unit.metadata["date"] == "2024-03-15"


def test_calendar_date_from_name(tmp_path):
    data = _minimal_export(nodes=[
        {"id": "n1", "name": "2024-01-20"},
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    unit = result.units[0]
    assert unit.metadata["date"] == "2024-01-20"


def test_epoch_millis_datetime(tmp_path):
    data = _minimal_export(nodes=[
        {"id": "n1", "name": "Timestamped", "createdAt": 1700000000000},
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    unit = result.units[0]
    assert unit.created_at.year == 2023


# ---------------------------------------------------------------------------
# Inline references
# ---------------------------------------------------------------------------

def test_inline_references_create_edges(tmp_path):
    data = _minimal_export(nodes=[
        {"id": "n1", "name": "See [[n2]] for details"},
        {"id": "n2", "name": "Target node"},
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    ref_edges = [e for e in result.edges if e.relation == EdgeRelation.REFERENCES]
    assert len(ref_edges) == 1
    assert ref_edges[0].metadata["relation_type"] == "inline_ref"


# ---------------------------------------------------------------------------
# Entity type filtering
# ---------------------------------------------------------------------------

def test_filter_nodes_only(tmp_path):
    data = _minimal_export(
        supertags=[{"id": "tag1", "name": "Task"}],
        nodes=[{"id": "n1", "name": "A node"}],
    )
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest(
        entity_types=["node"]
    )

    assert all(u.source_entity_type == "node" for u in result.units)


def test_filter_supertags_only(tmp_path):
    data = _minimal_export(
        supertags=[{"id": "tag1", "name": "Task"}],
        nodes=[{"id": "n1", "name": "A node"}],
    )
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest(
        entity_types=["supertag"]
    )

    assert all(u.source_entity_type == "supertag" for u in result.units)


# ---------------------------------------------------------------------------
# Flat list format
# ---------------------------------------------------------------------------

def test_flat_list_format(tmp_path):
    data = [
        {"id": "n1", "name": "Node one"},
        {"id": "n2", "name": "Node two"},
    ]
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    node_units = [u for u in result.units if u.source_entity_type == "node"]
    assert len(node_units) == 2


def test_flat_list_with_supertag_type(tmp_path):
    data = [
        {"id": "tag1", "name": "Book", "type": "supertag", "fields": [{"name": "Author", "type": "text", "id": "f1"}]},
        {"id": "n1", "name": "Dune", "supertags": ["tag1"]},
    ]
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    tag_units = [u for u in result.units if u.source_entity_type == "supertag"]
    assert len(tag_units) == 1
    assert tag_units[0].title == "Book"

    node_units = [u for u in result.units if u.source_entity_type == "node"]
    assert len(node_units) == 1
    assert "Book" in node_units[0].metadata["supertags"]


# ---------------------------------------------------------------------------
# Root path with multiple files
# ---------------------------------------------------------------------------

def test_root_path_multiple_files(tmp_path):
    _write_json(tmp_path / "a.json", _minimal_export(nodes=[{"id": "n1", "name": "From A"}]))
    _write_json(tmp_path / "b.json", _minimal_export(nodes=[{"id": "n2", "name": "From B"}]))

    result = TanaAdapter(root_path=str(tmp_path)).ingest()

    node_units = [u for u in result.units if u.source_entity_type == "node"]
    assert len(node_units) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_export(tmp_path):
    _write_json(tmp_path / "empty.json", _minimal_export(nodes=[], supertags=[]))

    result = TanaAdapter(file_path=str(tmp_path / "empty.json")).ingest()

    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        TanaAdapter(file_path="/nonexistent/path.json").ingest()


def test_no_path():
    with pytest.raises(ValueError, match="requires file_path or root_path"):
        TanaAdapter().ingest()


def test_malformed_json(tmp_path):
    (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed"):
        TanaAdapter(file_path=str(tmp_path / "bad.json")).ingest()


def test_registry_lookup():
    from graph.adapters.registry import get_adapter

    adapter = get_adapter("tana", file_path="/tmp/fake.json")
    assert adapter.name == "tana"


def test_children_format_without_nodes_key(tmp_path):
    """Test workspace-style export with top-level children instead of nodes."""
    data = {
        "children": [
            {"id": "n1", "name": "Child one"},
            {"id": "n2", "name": "Child two"},
        ]
    }
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    node_units = [u for u in result.units if u.source_entity_type == "node"]
    assert len(node_units) == 2


def test_iso_datetime_parsing(tmp_path):
    data = _minimal_export(nodes=[
        {"id": "n1", "name": "ISO dated", "createdAt": "2024-06-15T10:30:00Z"},
    ])
    _write_json(tmp_path / "export.json", data)

    result = TanaAdapter(file_path=str(tmp_path / "export.json")).ingest()

    unit = result.units[0]
    assert unit.created_at.year == 2024
    assert unit.created_at.month == 6
    assert unit.created_at.day == 15
