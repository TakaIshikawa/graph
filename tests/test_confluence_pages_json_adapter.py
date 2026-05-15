from __future__ import annotations

from graph.adapters.confluence_pages_json import ConfluencePagesJsonAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_confluence_pages_json_ingests_api_results_with_storage_body(tmp_path):
    export = tmp_path / "pages.json"
    export.write_text(
        """
        {
          "results": [
            {
              "id": "123",
              "title": "Import plan",
              "body": {"storage": {"value": "<p>Storage body</p>"}},
              "_links": {"base": "https://wiki.example", "webui": "/spaces/ENG/pages/123"},
              "space": {"key": "ENG", "name": "Engineering"},
              "history": {"createdDate": "2026-05-01T10:00:00Z", "createdBy": {"displayName": "Ada"}},
              "version": {"number": 7, "when": "2026-05-02T11:00:00Z", "message": "Update"},
              "labels": {"results": [{"name": "import"}, {"name": "csv"}]}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    result = ConfluencePagesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.CONFLUENCE_PAGES_JSON
    assert unit.source_id == "confluence_pages_json:123"
    assert unit.metadata["title"] == "Import plan"
    assert unit.metadata["body"] == "<p>Storage body</p>"
    assert unit.metadata["source_url"] == "https://wiki.example/spaces/ENG/pages/123"
    assert unit.metadata["space_key"] == "ENG"
    assert unit.metadata["space_name"] == "Engineering"
    assert unit.metadata["creator"] == "Ada"
    assert unit.metadata["version_number"] == 7
    assert unit.metadata["labels"] == ["import", "csv"]
    assert unit.metadata["created_at"] == "2026-05-01T10:00:00+00:00"
    assert unit.metadata["updated_at"] == "2026-05-02T11:00:00+00:00"


def test_confluence_pages_json_handles_array_and_view_body(tmp_path):
    export = tmp_path / "pages.json"
    export.write_text(
        """
        [
          {
            "id": "456",
            "title": "Runbook",
            "body": {"view": {"value": "<p>View body</p>"}},
            "url": "https://wiki.example/runbook",
            "labels": ["ops"]
          }
        ]
        """,
        encoding="utf-8",
    )

    result = ConfluencePagesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["body"] == "<p>View body</p>"
    assert result.units[0].metadata["labels"] == ["ops"]


def test_confluence_pages_json_is_registered():
    assert "confluence_pages_json" in list_adapters()
    assert isinstance(get_adapter("confluence-pages-json"), ConfluencePagesJsonAdapter)
