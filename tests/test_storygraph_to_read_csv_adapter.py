from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.storygraph_to_read_csv import StoryGraphToReadCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_storygraph_to_read_csv_ingests_reading_queue_metadata(tmp_path):
    export = tmp_path / "to-read.csv"
    _write_csv(
        export,
        [
            {
                "Title": "A Psalm for the Wild-Built",
                "Authors": "Becky Chambers",
                "ISBN": "9781250236210",
                "Format": "ebook",
                "Pages": "160",
                "Date Added": "2025-03-04",
                "Tags": "sci-fi, cozy",
                "Owned": "Yes",
                "URL": "https://app.thestorygraph.com/books/example",
            }
        ],
    )

    result = StoryGraphToReadCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "storygraph_to_read_csv"
    assert unit.source_id.startswith("storygraph_to_read_csv:")
    assert unit.source_entity_type == "reading_queue_item"
    assert unit.title == "A Psalm for the Wild-Built by Becky Chambers"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Title: A Psalm for the Wild-Built" in unit.content
    assert "Authors: Becky Chambers" in unit.content
    assert "ISBN: 9781250236210" in unit.content
    assert "Format: ebook" in unit.content
    assert "Pages: 160" in unit.content
    assert "Tags: sci-fi, cozy" in unit.content
    assert "Owned: True" in unit.content
    assert "Date added: 2025-03-04" in unit.content
    assert "URL: https://app.thestorygraph.com/books/example" in unit.content
    assert unit.metadata["title"] == "A Psalm for the Wild-Built"
    assert unit.metadata["authors"] == ["Becky Chambers"]
    assert unit.metadata["isbn"] == "9781250236210"
    assert unit.metadata["format"] == "ebook"
    assert unit.metadata["pages"] == 160
    assert unit.metadata["tags"] == ["sci-fi", "cozy"]
    assert unit.metadata["owned"] is True
    assert unit.metadata["date_added"] == "2025-03-04T00:00:00+00:00"
    assert unit.metadata["url"] == "https://app.thestorygraph.com/books/example"
    assert unit.metadata["source_file"] == "to-read.csv"
    assert unit.tags == ["storygraph", "to-read", "sci-fi", "cozy"]
    assert unit.created_at == datetime(2025, 3, 4, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 3, 4, tzinfo=timezone.utc)


def test_storygraph_to_read_csv_directory_filters_and_distinct_isbns(tmp_path):
    _write_csv(
        tmp_path / "old.csv",
        [
            {"Title": "Same Title", "Authors": "Ada", "ISBN": "111", "Date Added": "2025-01-01"},
            {},
        ],
    )
    _write_csv(
        tmp_path / "new.csv",
        [
            {"Title": "Same Title", "Authors": "Ada", "ISBN": "222", "Date Added": "2025-01-03"},
            {"Title": "Same Title", "Authors": "Ada", "ISBN": "333", "Date Added": "2025-01-04"},
            {"Title": "   ", "Authors": "", "ISBN": "", "URL": ""},
        ],
    )
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = StoryGraphToReadCsvAdapter(path=str(tmp_path))
    sync = SyncState(
        source_project="storygraph_to_read_csv",
        source_entity_type="reading_queue_item",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.metadata["isbn"] for unit in first.units] == ["333", "222"]
    assert len({unit.source_id for unit in first.units}) == 2
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["book"]).units == []
