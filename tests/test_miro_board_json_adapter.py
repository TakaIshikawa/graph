from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.miro_board_json import MiroBoardJsonAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_miro_board_json_ingests_supported_items_and_metadata(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "frame-1",
                        "type": "frame",
                        "data": {"title": "Research Plan"},
                        "position": {"x": 10, "y": 20},
                        "geometry": {"width": 800, "height": 600},
                        "style": {"fillColor": "#ffffff"},
                        "createdBy": {"id": "user-1", "name": "Ada Lovelace"},
                        "modifiedAt": "2025-01-02T03:04:05Z",
                    },
                    {
                        "id": "sticky-1",
                        "type": "sticky_note",
                        "data": {"content": "Check import edge cases"},
                        "parentFrameId": "frame-1",
                        "position": {"x": 100, "y": 120},
                        "geometry": {"width": 200, "height": 180},
                        "style": {"fillColor": "yellow"},
                        "links": [{"url": "https://example.test/note"}],
                        "tags": [{"title": "Import"}, "Miro"],
                        "createdBy": {"id": "user-1", "name": "Ada Lovelace"},
                        "updatedAt": "2025-01-03T00:00:00Z",
                    },
                    {
                        "id": "shape-1",
                        "type": "shape",
                        "data": {"content": "Decision"},
                        "parent": {"id": "frame-1"},
                        "x": 140,
                        "y": 180,
                        "width": 120,
                        "height": 80,
                        "style": {"shapeType": "round_rectangle"},
                        "url": "https://miro.com/app/board/shape-1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MiroBoardJsonAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["frame", "shape", "sticky_note"]
    frame = next(unit for unit in result.units if unit.source_entity_type == "frame")
    sticky = next(unit for unit in result.units if unit.source_entity_type == "sticky_note")
    shape = next(unit for unit in result.units if unit.source_entity_type == "shape")
    assert frame.source_project == SourceProject.MIRO_BOARD_JSON
    assert frame.source_id == "miro_board_json:frame-1"
    assert frame.title == "Research Plan"
    assert frame.metadata["position"] == {"x": 10, "y": 20}
    assert frame.metadata["dimensions"] == {"width": 800, "height": 600}
    assert frame.metadata["style"] == {"fillColor": "#ffffff"}
    assert frame.metadata["creator"] == {"id": "user-1", "name": "Ada Lovelace"}
    assert frame.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert sticky.metadata["parent_frame_id"] == "frame-1"
    assert sticky.metadata["links"] == [{"url": "https://example.test/note"}]
    assert sticky.metadata["tags"] == ["Import", "Miro"]
    assert sticky.metadata["source_file"] == "board.json"
    assert shape.metadata["parent_frame_id"] == "frame-1"
    assert shape.metadata["width"] == 120
    assert len(result.edges) == 2
    assert {edge.to_unit_id for edge in result.edges} == {sticky.source_id, shape.source_id}
    assert all(edge.from_unit_id == frame.source_id for edge in result.edges)
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in result.edges)
    assert all(edge.source == EdgeSource.SOURCE for edge in result.edges)


def test_miro_board_json_reads_directory_and_filters_entity_types(tmp_path):
    (tmp_path / "one.json").write_text(
        json.dumps({"items": [{"id": "text-1", "type": "text", "data": {"content": "Hello"}}]}),
        encoding="utf-8",
    )
    (tmp_path / "two.json").write_text(
        json.dumps({"data": [{"id": "card-1", "type": "card", "data": {"title": "Task", "description": "Do it"}}]}),
        encoding="utf-8",
    )

    result = MiroBoardJsonAdapter(path=str(tmp_path)).ingest(entity_types=["text", "card"])

    assert [unit.source_id for unit in result.units] == ["miro_board_json:card-1", "miro_board_json:text-1"]
    assert {unit.source_entity_type for unit in result.units} == {"text", "card"}
    assert MiroBoardJsonAdapter(path=str(tmp_path)).ingest(entity_types=["sticky_note"]).units == []


def test_miro_board_json_filters_since_by_updated_at(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "old", "type": "text", "data": {"content": "Old"}, "updatedAt": "2025-01-01T00:00:00Z"},
                    {
                        "id": "boundary",
                        "type": "text",
                        "data": {"content": "Boundary"},
                        "updatedAt": "2025-01-02T00:00:00Z",
                    },
                    {"id": "new", "type": "text", "data": {"content": "New"}, "updatedAt": "2025-01-03T00:00:00Z"},
                ]
            }
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="miro_board_json",
        source_entity_type="text",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = MiroBoardJsonAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["miro_board_json:new"]


def test_miro_board_json_skips_unsupported_and_missing_ids(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "connector-1", "type": "connector", "data": {"content": "Line"}},
                    {"type": "sticky_note", "data": {"content": "No ID"}},
                    {"id": "sticky-1", "type": "sticky", "data": {"content": "Has ID"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MiroBoardJsonAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == ["miro_board_json:sticky-1"]
    assert result.units[0].source_entity_type == "sticky_note"


def test_miro_board_json_emits_connector_edges_between_ingested_units(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "sticky-1", "type": "sticky_note", "data": {"content": "Start"}},
                    {"id": "shape-1", "type": "shape", "data": {"content": "End"}},
                    {"id": "text-1", "type": "text", "data": {"content": "Excluded"}},
                    {"id": "connector-1", "type": "connector", "startItem": {"id": "sticky-1"}, "endItem": {"id": "shape-1"}},
                    {"id": "connector-dup", "type": "connector", "startItemId": "sticky-1", "endItemId": "shape-1"},
                    {"id": "connector-skip", "type": "line", "data": {"start": {"itemId": "shape-1"}, "end": {"itemId": "missing"}}},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MiroBoardJsonAdapter(path=str(export)).ingest(entity_types=["sticky_note", "shape"])

    assert [unit.source_id for unit in result.units] == ["miro_board_json:shape-1", "miro_board_json:sticky-1"]
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == "miro_board_json:sticky-1"
    assert edge.to_unit_id == "miro_board_json:shape-1"
    assert edge.relation == EdgeRelation.RELATES_TO
    assert edge.source == EdgeSource.SOURCE
    assert edge.metadata["relation_type"] == "miro_connector_connects_items"
    assert edge.metadata["connector_type"] == "connector"
    assert edge.metadata["start_item_id"] == "sticky-1"
    assert edge.metadata["end_item_id"] == "shape-1"
