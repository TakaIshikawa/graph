from __future__ import annotations

import json

from graph.adapters.archivebox_index_json import ArchiveBoxIndexJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_archivebox_index_json_imports_entries(tmp_path):
    export = tmp_path / "index.json"
    export.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "timestamp": "2025-01-02T03:04:05Z",
                        "tags": ["research", "web"],
                        "status": "succeeded",
                        "history": {"title": [{"status": "succeeded"}]},
                        "archive_path": "archive/1735787045",
                        "index_path": "archive/1735787045/index.html",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ArchiveBoxIndexJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.ARCHIVEBOX_INDEX_JSON
    assert unit.title == "Example Article"
    assert unit.metadata["url"] == "https://example.com/article"
    assert unit.metadata["timestamp"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["tags"] == ["research", "web"]
    assert unit.metadata["status"] == "succeeded"
    assert "archive/1735787045/index.html" in unit.metadata["archive_paths"]
    assert "history" in unit.metadata["extractor_outputs"]


def test_archivebox_index_json_falls_back_to_url_and_registry(tmp_path):
    export = tmp_path / "index.json"
    export.write_text(
        json.dumps({"snapshots": {"abc": {"url": "https://example.com/untitled", "timestamp": "1735689600"}}}),
        encoding="utf-8",
    )

    result = ArchiveBoxIndexJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "https://example.com/untitled"
    assert result.units[0].metadata["timestamp"] == "2025-01-01T00:00:00+00:00"
    assert get_adapter("archivebox_index_json", path=str(export)).name == "archivebox_index_json"
