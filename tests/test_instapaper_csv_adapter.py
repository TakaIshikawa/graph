from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.instapaper_csv import InstapaperCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_instapaper_csv_ingests_article_with_selection_and_progress(tmp_path):
    export = tmp_path / "instapaper.csv"
    export.write_text(
        "Title,URL,Folder,State,Selection,Description,Date Saved,Progress,Progress Position,Progress Total\n"
        'Readable Systems,https://example.com/read,Archive,Read,"Keep the interface boring.",'
        "A useful essay,2025-01-02T03:04:05Z,75%,120,160\n",
        encoding="utf-8",
    )

    result = InstapaperCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "instapaper_csv"
    assert unit.source_id.startswith("instapaper_csv:")
    assert unit.source_entity_type == "article"
    assert unit.title == "Readable Systems"
    assert unit.content_type == ContentType.ARTIFACT
    assert "URL: https://example.com/read" in unit.content
    assert "Folder: Archive" in unit.content
    assert "State: Read" in unit.content
    assert "Description: A useful essay" in unit.content
    assert "Selection: Keep the interface boring." in unit.content
    assert "Progress: 75% / 120 / 160" in unit.content
    assert unit.metadata["url"] == "https://example.com/read"
    assert unit.metadata["folder"] == "Archive"
    assert unit.metadata["state"] == "Read"
    assert unit.metadata["folder_tag"] == "archive"
    assert unit.metadata["state_tag"] == "read"
    assert unit.metadata["selection"] == "Keep the interface boring."
    assert unit.metadata["highlight"] == "Keep the interface boring."
    assert unit.metadata["description"] == "A useful essay"
    assert unit.metadata["date_saved"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["progress"] == "75%"
    assert unit.metadata["progress_position"] == "120"
    assert unit.metadata["progress_total"] == "160"
    assert unit.tags == ["instapaper", "archive", "read"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_instapaper_csv_handles_minimal_article_and_stable_ids(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text("Article Title,Link,Saved At\nOnly URL,https://example.org/a,2025-02-03\n", encoding="utf-8")

    first = InstapaperCsvAdapter(path=str(export)).ingest().units[0]
    second = InstapaperCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.title == "Only URL"
    assert first.metadata["date_saved"] == "2025-02-03T00:00:00+00:00"
    assert first.tags == ["instapaper"]


def test_instapaper_csv_since_and_entity_filters(tmp_path):
    export = tmp_path / "instapaper.csv"
    export.write_text(
        "Title,URL,Date Saved\n"
        "Old,https://example.com/old,2025-01-01T00:00:00Z\n"
        "New,https://example.com/new,2025-01-03T00:00:00Z\n",
        encoding="utf-8",
    )

    filtered = InstapaperCsvAdapter(path=str(export)).ingest(
        since=SyncState(source_project="instapaper_csv", source_entity_type="article", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    )
    wrong_entity = InstapaperCsvAdapter(path=str(export)).ingest(entity_types=["bookmark"])

    assert [unit.title for unit in filtered.units] == ["New"]
    assert wrong_entity.units == []
