from __future__ import annotations

from graph.adapters.pocket_reading_list_csv import PocketReadingListCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_pocket_reading_list_csv_ingests_unread_archived_and_favorite_items(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "title,url,time_added,time_read,tags,status,excerpt,favorite,lang\n"
        "Unread,https://example.com/a,2026-05-01T10:00:00Z,,\"Read Later, Python\",unread,Useful article,0,en\n"
        "Archived,https://example.com/b,2026-05-02T10:00:00Z,2026-05-03T10:00:00Z,Archive,archive,Read article,true,ja\n",
        encoding="utf-8",
    )

    result = PocketReadingListCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Unread", "Archived"]
    unread = result.units[0]
    assert unread.source_project == SourceProject.POCKET_READING_LIST_CSV
    assert unread.source_entity_type == "saved_item"
    assert unread.metadata["url"] == "https://example.com/a"
    assert unread.metadata["source_url"] == "https://example.com/a"
    assert unread.metadata["status"] == "unread"
    assert unread.metadata["archived"] is False
    assert unread.metadata["favorite"] is False
    assert unread.metadata["tags"] == ["read later", "python"]
    assert unread.metadata["excerpt"] == "Useful article"
    assert unread.metadata["language"] == "en"
    assert unread.metadata["domain"] == "example.com"
    assert unread.metadata["added_at"] == "2026-05-01T10:00:00+00:00"
    assert unread.tags == ["read later", "python"]
    assert "Excerpt: Useful article" in unread.content
    archived = result.units[1]
    assert archived.metadata["archived"] is True
    assert archived.metadata["favorite"] is True
    assert archived.metadata["read_at"] == "2026-05-03T10:00:00+00:00"
    assert "Favorite: true" in archived.content


def test_pocket_reading_list_csv_handles_minimal_rows_and_missing_optional_columns(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text(
        "title,url\n"
        ",\n"
        "Title only,\n"
        ",https://example.org/item\n",
        encoding="utf-8",
    )

    result = PocketReadingListCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Title only", "https://example.org/item"]
    assert result.units[0].metadata["status"] == "unread"
    assert result.units[1].metadata["domain"] == "example.org"


def test_pocket_reading_list_csv_ids_are_deterministic(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "title,url,time_added\n"
        "Unread,https://example.com/a,2026-05-01\n",
        encoding="utf-8",
    )

    first = PocketReadingListCsvAdapter(path=str(export)).ingest().units[0]
    second = PocketReadingListCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id


def test_pocket_reading_list_csv_is_registered():
    assert "pocket_reading_list_csv" in list_adapters()
    assert isinstance(get_adapter("pocket-reading-list-csv"), PocketReadingListCsvAdapter)
