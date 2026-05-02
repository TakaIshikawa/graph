from __future__ import annotations

import json

import pytest

from graph.adapters.obsidian_canvas import ObsidianCanvasAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject


def write_canvas(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_obsidian_canvas_ingests_nodes_edges_and_group_membership(tmp_path):
    canvas_path = tmp_path / "map.canvas"
    write_canvas(
        canvas_path,
        {
            "nodes": [
                {
                    "id": "group-1",
                    "type": "group",
                    "label": "Research cluster",
                    "x": 0,
                    "y": 0,
                    "width": 500,
                    "height": 400,
                    "color": "2",
                },
                {
                    "id": "text-1",
                    "type": "text",
                    "text": "Canvas note\nwith context",
                    "x": 40,
                    "y": 50,
                    "width": 220,
                    "height": 120,
                    "color": "#ffaa00",
                },
                {
                    "id": "file-1",
                    "type": "file",
                    "file": "Notes/Source.md",
                    "subpath": "#Evidence",
                    "x": 280,
                    "y": 80,
                    "width": 180,
                    "height": 140,
                    "color": "4",
                },
                {
                    "id": "link-1",
                    "type": "link",
                    "url": "https://example.com",
                    "x": 600,
                    "y": 50,
                    "width": 180,
                    "height": 100,
                },
            ],
            "edges": [
                {
                    "id": "edge-1",
                    "fromNode": "text-1",
                    "fromSide": "right",
                    "toNode": "file-1",
                    "toSide": "left",
                    "label": "supports",
                    "color": "#00aaff",
                },
                {
                    "id": "edge-missing",
                    "fromNode": "text-1",
                    "toNode": "missing",
                },
            ],
        },
    )

    result = ObsidianCanvasAdapter(path=str(canvas_path)).ingest()

    units = {unit.source_id: unit for unit in result.units}
    assert set(units) == {
        "obsidian_canvas:map.canvas#group-1",
        "obsidian_canvas:map.canvas#text-1",
        "obsidian_canvas:map.canvas#file-1",
        "obsidian_canvas:map.canvas#link-1",
    }

    text = units["obsidian_canvas:map.canvas#text-1"]
    assert text.source_project == SourceProject.OBSIDIAN_CANVAS
    assert text.source_entity_type == "canvas_text"
    assert text.title == "Canvas note"
    assert text.content == "Canvas note\nwith context"
    assert text.content_type == ContentType.ARTIFACT
    assert text.metadata["canvas_path"] == "map.canvas"
    assert text.metadata["node_id"] == "text-1"
    assert text.metadata["x"] == 40
    assert text.metadata["y"] == 50
    assert text.metadata["width"] == 220
    assert text.metadata["height"] == 120
    assert text.metadata["color"] == "#ffaa00"

    file_unit = units["obsidian_canvas:map.canvas#file-1"]
    assert file_unit.source_entity_type == "canvas_file"
    assert file_unit.title == "Source.md"
    assert file_unit.content == "Notes/Source.md\n#Evidence"
    assert file_unit.metadata["file"] == "Notes/Source.md"
    assert file_unit.metadata["subpath"] == "#Evidence"

    group = units["obsidian_canvas:map.canvas#group-1"]
    assert group.source_entity_type == "canvas_group"
    assert group.title == "Research cluster"
    assert group.metadata["label"] == "Research cluster"
    assert group.metadata["color"] == "2"

    link = units["obsidian_canvas:map.canvas#link-1"]
    assert link.source_entity_type == "canvas_link"
    assert link.content == "https://example.com"
    assert link.metadata["url"] == "https://example.com"

    explicit_edges = [
        edge for edge in result.edges if edge.metadata["relation_type"] == "canvas_edge"
    ]
    assert len(explicit_edges) == 1
    explicit = explicit_edges[0]
    assert explicit.from_unit_id == "obsidian_canvas:map.canvas#text-1"
    assert explicit.to_unit_id == "obsidian_canvas:map.canvas#file-1"
    assert explicit.relation == EdgeRelation.RELATES_TO
    assert explicit.source == EdgeSource.SOURCE
    assert explicit.metadata["edge_id"] == "edge-1"
    assert explicit.metadata["fromSide"] == "right"
    assert explicit.metadata["toSide"] == "left"
    assert explicit.metadata["label"] == "supports"
    assert explicit.metadata["color"] == "#00aaff"

    group_edges = [
        edge for edge in result.edges if edge.metadata["relation_type"] == "canvas_group_contains"
    ]
    assert {
        (edge.from_unit_id, edge.to_unit_id, edge.relation)
        for edge in group_edges
    } == {
        (
            "obsidian_canvas:map.canvas#group-1",
            "obsidian_canvas:map.canvas#text-1",
            EdgeRelation.CONTAINS,
        ),
        (
            "obsidian_canvas:map.canvas#group-1",
            "obsidian_canvas:map.canvas#file-1",
            EdgeRelation.CONTAINS,
        ),
    }


def test_obsidian_canvas_directory_ingestion_is_sorted_and_filters_entity_types(tmp_path):
    write_canvas(
        tmp_path / "b.canvas",
        {
            "nodes": [
                {
                    "id": "b",
                    "type": "text",
                    "text": "B",
                    "x": 0,
                    "y": 0,
                    "width": 10,
                    "height": 10,
                }
            ]
        },
    )
    write_canvas(
        tmp_path / "a.canvas",
        {
            "nodes": [
                {
                    "id": "a",
                    "type": "file",
                    "file": "A.md",
                    "x": 0,
                    "y": 0,
                    "width": 10,
                    "height": 10,
                }
            ]
        },
    )

    result = ObsidianCanvasAdapter(path=str(tmp_path)).ingest(entity_types=["canvas_file"])

    assert [unit.source_id for unit in result.units] == ["obsidian_canvas:a.canvas#a"]
    assert result.units[0].source_entity_type == "canvas_file"


def test_obsidian_canvas_malformed_json_identifies_file(tmp_path):
    bad = tmp_path / "bad.canvas"
    bad.write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError, match=f"Malformed Obsidian Canvas JSON in {bad}"):
        ObsidianCanvasAdapter(path=str(bad)).ingest()


def test_obsidian_canvas_adapter_is_registered():
    assert "obsidian_canvas" in list_adapters()
    adapter = get_adapter("obsidian_canvas", path="/tmp/map.canvas")
    assert isinstance(adapter, ObsidianCanvasAdapter)
    assert adapter.name == "obsidian_canvas"
