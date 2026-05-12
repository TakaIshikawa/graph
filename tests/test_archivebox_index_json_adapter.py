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


def test_archivebox_index_json_emits_extractor_artifacts_and_edges(tmp_path):
    export = tmp_path / "index.json"
    export.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "timestamp": "2025-01-02T03:04:05Z",
                        "history": {
                            "readability": {"path": "archive/123/readability/content.html"},
                            "pdf": {"path": "archive/123/output.pdf"},
                        },
                        "screenshot": "archive/123/screenshot.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = ArchiveBoxIndexJsonAdapter(path=str(export))
    assert adapter.entity_types == ["archive", "artifact", "url_reference", "domain"]

    result = adapter.ingest(entity_types=["archive", "artifact"])
    archives = [unit for unit in result.units if unit.source_entity_type == "archive"]
    artifacts = [unit for unit in result.units if unit.source_entity_type == "artifact"]

    assert len(archives) == 1
    assert {unit.metadata["extractor"] for unit in artifacts} == {"pdf", "readability", "screenshot", "title"}
    assert all(unit.metadata["parent_archive_source_id"] == archives[0].source_id for unit in artifacts)
    assert all(unit.metadata["source_file"] == "index.json" for unit in artifacts)
    assert all(unit.metadata["original_url"] == "https://example.com/article" for unit in artifacts)
    assert len(result.edges) == 4
    assert {edge.from_unit_id for edge in result.edges} == {archives[0].source_id}
    assert {edge.to_unit_id for edge in result.edges} == {unit.source_id for unit in artifacts}


def test_archivebox_index_json_artifact_edges_respect_entity_filtering(tmp_path):
    export = tmp_path / "index.json"
    export.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "timestamp": "2025-01-02T03:04:05Z",
                        "pdf": "archive/123/output.pdf",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    archive_only = ArchiveBoxIndexJsonAdapter(path=str(export)).ingest(entity_types=["archive"])
    artifact_only = ArchiveBoxIndexJsonAdapter(path=str(export)).ingest(entity_types=["artifact"])

    assert [unit.source_entity_type for unit in archive_only.units] == ["archive"]
    assert archive_only.edges == []
    assert [unit.source_entity_type for unit in artifact_only.units] == ["artifact", "artifact"]
    assert {unit.metadata["extractor"] for unit in artifact_only.units} == {"pdf", "title"}
    assert artifact_only.edges == []


def test_archivebox_index_json_emits_outbound_url_references(tmp_path):
    export = tmp_path / "index.json"
    export.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "timestamp": "2025-01-02T03:04:05Z",
                        "outlinks": [
                            {"url": "https://target.example/a", "title": "Target A"},
                            {"href": "https://target.example/b", "text": "Target B"},
                            {"url": "https://target.example/a", "title": "Duplicate"},
                        ],
                        "metadata": {"outlinks": ["https://target.example/c"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ArchiveBoxIndexJsonAdapter(path=str(export)).ingest(entity_types=["archive", "url_reference"])

    archive = next(unit for unit in result.units if unit.source_entity_type == "archive")
    references = sorted(
        [unit for unit in result.units if unit.source_entity_type == "url_reference"],
        key=lambda unit: unit.metadata["url"],
    )
    assert [unit.metadata["url"] for unit in references] == [
        "https://target.example/a",
        "https://target.example/b",
        "https://target.example/c",
    ]
    assert references[0].metadata["title"] == "Target A"
    assert references[1].metadata["text"] == "Target B"
    assert all(unit.metadata["source_file"] == "index.json" for unit in references)
    assert len(result.edges) == 3
    assert {edge.from_unit_id for edge in result.edges} == {archive.source_id}
    assert {edge.to_unit_id for edge in result.edges} == {unit.source_id for unit in references}


def test_archivebox_index_json_emits_deduplicated_domain_units_and_edges(tmp_path):
    export = tmp_path / "index.json"
    export.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "url": "https://www.example.com/article",
                        "title": "Example A",
                        "timestamp": "2025-01-02T03:04:05Z",
                    },
                    {
                        "url": "https://example.com/other",
                        "title": "Example B",
                        "timestamp": "2025-01-03T03:04:05Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ArchiveBoxIndexJsonAdapter(path=str(export)).ingest(entity_types=["archive", "domain"])

    archives = [unit for unit in result.units if unit.source_entity_type == "archive"]
    domains = [unit for unit in result.units if unit.source_entity_type == "domain"]
    assert len(archives) == 2
    assert len(domains) == 1
    assert domains[0].source_id.startswith("archivebox_index_json:domain:")
    assert domains[0].metadata["domain"] == "example.com"
    assert domains[0].title == "example.com"
    assert len(result.edges) == 2
    assert {edge.to_unit_id for edge in result.edges} == {domains[0].source_id}


def test_archivebox_index_json_domain_filtering(tmp_path):
    export = tmp_path / "index.json"
    export.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "url": "https://www.example.com/article",
                        "title": "Example",
                        "timestamp": "2025-01-02T03:04:05Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    domain_only = ArchiveBoxIndexJsonAdapter(path=str(export)).ingest(entity_types=["domain"])
    archive_only = ArchiveBoxIndexJsonAdapter(path=str(export)).ingest(entity_types=["archive"])

    assert [unit.source_entity_type for unit in domain_only.units] == ["domain"]
    assert domain_only.edges == []
    assert [unit.source_entity_type for unit in archive_only.units] == ["archive"]
    assert archive_only.edges == []
