from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.goodreads_quotes_csv import GoodreadsQuotesCsvAdapter
from graph.types.models import SyncState


def test_goodreads_quotes_csv_ingests_quotes_with_book_author_and_tags(tmp_path):
    export = tmp_path / "quotes.csv"
    export.write_text(
        "Quote ID,Quote,Author,Book,Tags,Date Added,Page,URL\n"
        "q1,Stay gold,Ponyboy,The Outsiders,\"classic, ya\",2026-05-01,12,https://example.com/q1\n"
        ",No quote,,,,2026-05-02,,\n",
        encoding="utf-8",
    )

    result = GoodreadsQuotesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_id == "goodreads_quotes_csv:q1"
    assert unit.source_entity_type == "quote"
    assert unit.metadata["author"] == "Ponyboy"
    assert unit.metadata["book"] == "The Outsiders"
    assert unit.metadata["tags"] == ["classic", "ya"]
    assert unit.metadata["page"] == 12
    assert unit.created_at == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert "Author: Ponyboy" in unit.content


def test_goodreads_quotes_csv_uses_digest_fallback_and_filters(tmp_path):
    export = tmp_path / "quotes.csv"
    export.write_text(
        "Quote,Author,Book,Date Added\n"
        "Old quote,Author A,Book A,2026-05-01\n"
        "New quote,Author B,Book B,2026-05-03\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="goodreads_quotes_csv", source_entity_type="quote", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    first = GoodreadsQuotesCsvAdapter(path=str(export)).ingest()
    second = GoodreadsQuotesCsvAdapter(path=str(export)).ingest()
    filtered = GoodreadsQuotesCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert [unit.metadata["quote"] for unit in filtered.units] == ["New quote"]
    assert GoodreadsQuotesCsvAdapter(path=str(export)).ingest(entity_types=["book"]).units == []
