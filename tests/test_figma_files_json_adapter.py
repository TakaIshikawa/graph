from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.figma_files_json import FigmaFilesJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_figma_files_json_ingests_wrapper_metadata_and_registry(tmp_path):
    export = tmp_path / "figma.json"
    export.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "key": "abc123",
                        "name": "Mobile App",
                        "url": "https://figma.com/file/abc123",
                        "project": {"name": "Design System"},
                        "team": {"name": "Product"},
                        "thumbnail_url": "https://img.example/thumb.png",
                        "last_modified": "2025-01-02T00:00:00Z",
                        "version_count": 7,
                        "description": "Primary app file",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = FigmaFilesJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == SourceProject.FIGMA_FILES_JSON
    assert unit.source_id == "figma_files_json:abc123"
    assert unit.title == "Mobile App"
    assert unit.metadata["project"] == "Design System"
    assert unit.metadata["team"] == "Product"
    assert unit.metadata["thumbnail_url"] == "https://img.example/thumb.png"
    assert unit.metadata["version_count"] == 7
    assert unit.updated_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert "Primary app file" in unit.content
    assert get_adapter("figma_files_json", path=str(export)).name == "figma_files_json"


def test_figma_files_json_handles_missing_thumbnail_and_filters(tmp_path):
    export = tmp_path / "figma.json"
    export.write_text(json.dumps([{"key": "no-thumb", "name": "No Thumbnail", "lastModified": "2025-01-01T00:00:00Z"}]), encoding="utf-8")

    adapter = FigmaFilesJsonAdapter(path=str(export))
    unit = adapter.ingest().units[0]

    assert unit.metadata["key"] == "no-thumb"
    assert "thumbnail_url" not in unit.metadata
    assert adapter.ingest(entity_types=["project"]).units == []


def test_figma_files_json_multiple_projects_and_stable_ids(tmp_path):
    export = tmp_path / "figma.json"
    export.write_text(
        json.dumps({"items": [{"key": "one", "name": "One", "project_name": "A"}, {"key": "two", "name": "Two", "project_name": "B"}]}),
        encoding="utf-8",
    )

    adapter = FigmaFilesJsonAdapter(path=str(export))
    first = adapter.ingest()
    second = adapter.ingest()

    assert [unit.metadata["project"] for unit in first.units] == ["A", "B"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
