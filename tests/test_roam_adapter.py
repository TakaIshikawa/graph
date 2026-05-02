from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.roam import RoamAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject


def test_roam_json_export_ingests_pages_nested_blocks_and_references(tmp_path):
    export = tmp_path / "roam.json"
    export.write_text(
        json.dumps(
            [
                {
                    "title": "Projects",
                    "uid": "page-projects",
                    "create-time": 1735689600000,
                    "edit-time": 1735693200000,
                    "children": [
                        {
                            "string": "Build [[Graph Tool]] #Research",
                            "uid": "block-alpha",
                            "create-time": 1735689660000,
                            "edit-time": 1735689720000,
                            "children": [
                                {
                                    "string": "Nested note references ((block-target)) and [[Missing Page]]",
                                    "uid": "block-beta",
                                }
                            ],
                        }
                    ],
                },
                {
                    "title": "Graph Tool",
                    "uid": "page-graph-tool",
                    "children": [
                        {
                            "string": "Target block",
                            "uid": "block-target",
                        }
                    ],
                },
                {
                    "title": "Research",
                    "uid": "page-research",
                    "children": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    first = RoamAdapter(file_path=str(export)).ingest()
    second = RoamAdapter(file_path=str(export)).ingest()

    assert [unit.source_id for unit in first.units] == [
        "roam:block:block-alpha",
        "roam:block:block-beta",
        "roam:block:block-target",
        "roam:page:page-graph-tool",
        "roam:page:page-projects",
        "roam:page:page-research",
    ]
    assert [unit.source_id for unit in second.units] == [unit.source_id for unit in first.units]

    projects = next(unit for unit in first.units if unit.source_id == "roam:page:page-projects")
    assert projects.source_project == SourceProject.ROAM
    assert projects.source_entity_type == "page"
    assert projects.content_type == ContentType.ARTIFACT
    assert projects.title == "Projects"
    assert "Build [[Graph Tool]] #Research" in projects.content
    assert projects.tags == ["graph tool", "missing page", "research"]
    assert projects.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert projects.updated_at == datetime(2025, 1, 1, 1, tzinfo=timezone.utc)

    alpha = next(unit for unit in first.units if unit.source_id == "roam:block:block-alpha")
    assert alpha.title == "Build [[Graph Tool]] #Research"
    assert alpha.tags == ["graph tool", "research"]
    assert alpha.metadata["page_source_id"] == "roam:page:page-projects"
    assert alpha.metadata["parent_source_id"] == "roam:page:page-projects"
    assert alpha.metadata["position"] == [1, 1]
    assert alpha.created_at == datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc)
    assert alpha.updated_at == datetime(2025, 1, 1, 0, 2, tzinfo=timezone.utc)

    assert [(edge.from_unit_id, edge.to_unit_id, edge.relation, edge.source) for edge in first.edges] == [
        (
            "roam:block:block-alpha",
            "roam:page:page-graph-tool",
            EdgeRelation.REFERENCES,
            EdgeSource.SOURCE,
        ),
        (
            "roam:block:block-alpha",
            "roam:page:page-research",
            EdgeRelation.REFERENCES,
            EdgeSource.SOURCE,
        ),
        (
            "roam:block:block-beta",
            "roam:block:block-target",
            EdgeRelation.REFERENCES,
            EdgeSource.SOURCE,
        ),
    ]
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]


def test_roam_root_path_and_uidless_source_ids_are_deterministic(tmp_path):
    export = tmp_path / "nested" / "export.json"
    export.parent.mkdir()
    payload = [
        {
            "title": "Daily Notes",
            "children": [
                {
                    "string": "Remember [[Projects]]",
                    "children": [{"string": "Child #Projects"}],
                }
            ],
        },
        {"title": "Projects", "children": []},
    ]
    export.write_text(json.dumps(payload), encoding="utf-8")

    first = RoamAdapter(root_path=str(tmp_path)).ingest()
    second = RoamAdapter(root_path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]
    assert len(first.units) == 4
    assert len(first.edges) == 2


def test_roam_respects_entity_types(tmp_path):
    export = tmp_path / "roam.json"
    export.write_text(json.dumps([{"title": "Page", "uid": "page", "children": [{"string": "Block", "uid": "block"}]}]), encoding="utf-8")

    result = RoamAdapter(file_path=str(export)).ingest(entity_types=["page"])

    assert [unit.source_id for unit in result.units] == ["roam:page:page"]
    assert result.edges == []


def test_roam_missing_and_malformed_paths_raise_clear_exceptions(tmp_path):
    with pytest.raises(FileNotFoundError, match="Roam JSON export path does not exist"):
        RoamAdapter(file_path=str(tmp_path / "missing.json")).ingest()

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed Roam JSON export"):
        RoamAdapter(file_path=str(malformed)).ingest()

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="pages list"):
        RoamAdapter(file_path=str(invalid)).ingest()


def test_roam_adapter_is_registered():
    assert "roam" in list_adapters()
    adapter = get_adapter("roam", file_path="/tmp/roam.json")
    assert adapter.name == "roam"
