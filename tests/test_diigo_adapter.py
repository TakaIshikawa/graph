from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.diigo import DiigoAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def unix_time(year: int, month: int, day: int) -> str:
    return str(int(datetime(year, month, day, tzinfo=timezone.utc).timestamp()))


def test_diigo_csv_ingests_annotated_bookmarks(tmp_path):
    export = tmp_path / "diigo.csv"
    export.write_text(
        "\n".join(
            [
                "url,title,tags,description,annotations,highlights,privacy,created_at,id",
                f"https://example.com/article,Article Title,research;important,Great article,My annotation here,Highlighted text,private,{unix_time(2025, 1, 10)},abc123",
            ]
        ),
        encoding="utf-8",
    )

    result = DiigoAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.DIIGO
    assert unit.source_id == "diigo:abc123"
    assert unit.source_entity_type == "bookmark"
    assert unit.title == "Article Title"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Article Title" in unit.content
    assert "URL: https://example.com/article" in unit.content
    assert "Description: Great article" in unit.content
    assert "Annotations: My annotation here" in unit.content
    assert "Highlights: Highlighted text" in unit.content
    assert unit.metadata == {
        "url": "https://example.com/article",
        "description": "Great article",
        "annotations": "My annotation here",
        "highlights": "Highlighted text",
        "privacy": "private",
        "tags": ["research", "important"],
        "created_at": unix_time(2025, 1, 10),
        "diigo_id": "abc123",
    }
    assert unit.tags == ["research", "important"]
    assert unit.created_at == datetime(2025, 1, 10, tzinfo=timezone.utc)


def test_diigo_csv_handles_missing_fields(tmp_path):
    export = tmp_path / "diigo.csv"
    export.write_text(
        "\n".join(
            [
                "url,title",
                "https://example.com/minimal,Minimal Bookmark",
            ]
        ),
        encoding="utf-8",
    )

    result = DiigoAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "url:https://example.com/minimal"
    assert unit.title == "Minimal Bookmark"
    assert unit.metadata["annotations"] == ""
    assert unit.metadata["highlights"] == ""
    assert unit.tags == []


def test_diigo_csv_uses_url_fallback_for_source_id(tmp_path):
    export = tmp_path / "diigo.csv"
    export.write_text(
        "\n".join(
            [
                "url,title",
                "https://example.com/no-id,No ID Bookmark",
            ]
        ),
        encoding="utf-8",
    )

    result = DiigoAdapter(path=str(export)).ingest()

    assert result.units[0].source_id == "url:https://example.com/no-id"


def test_diigo_csv_filters_by_sync_state(tmp_path):
    export = tmp_path / "diigo.csv"
    export.write_text(
        "\n".join(
            [
                "id,url,title,created_at",
                f"old,https://example.com/old,Old,{unix_time(2025, 1, 1)}",
                f"equal,https://example.com/equal,Equal,{unix_time(2025, 1, 2)}",
                f"new,https://example.com/new,New,{unix_time(2025, 1, 3)}",
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="diigo",
        source_entity_type="bookmark",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = DiigoAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["diigo:new"]


def test_diigo_respects_entity_types(tmp_path):
    export = tmp_path / "diigo.csv"
    export.write_text(
        "\n".join(
            [
                "url,title",
                "https://example.com,Test",
            ]
        ),
        encoding="utf-8",
    )

    result = DiigoAdapter(path=str(export)).ingest(entity_types=["saved_item"])

    assert result.units == []


def test_diigo_handles_malformed_csv(tmp_path):
    export = tmp_path / "diigo.csv"
    export.write_text("not,valid,csv\ndata", encoding="utf-8")

    # Should not raise an exception, just return empty result
    result = DiigoAdapter(path=str(export)).ingest()
    # May have some units if the CSV can be partially parsed
    assert isinstance(result.units, list)


def test_diigo_handles_missing_file():
    result = DiigoAdapter(path="/nonexistent/file.csv").ingest()
    assert result.units == []
    assert result.edges == []


def test_diigo_adapter_is_registered():
    assert "diigo" in list_adapters()
    adapter = get_adapter("diigo", path="/tmp/diigo.csv")
    assert adapter.name == "diigo"
