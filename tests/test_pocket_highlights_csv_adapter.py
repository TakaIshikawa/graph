from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.pocket_highlights_csv import PocketHighlightsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_pocket_highlights_csv_header_variants_skips_blanks_filters_and_registry(tmp_path):
    path = tmp_path / "highlights.csv"
    path.write_text("Article Title,Article URL,Highlight Text,Note,Tags,Created At\nOld,https://e/old,Old quote,,a|b,2026-04-01\nNew,https://e/new,New quote,Note,c,2026-05-03\nBlank,https://e/blank,,,x,2026-05-04\n", encoding="utf-8")
    since = SyncState(source_project="pocket_highlights_csv", source_entity_type="highlight", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = PocketHighlightsCsvAdapter(path=str(path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New"]
    unit = result.units[0]
    assert unit.source_id == PocketHighlightsCsvAdapter(path=str(path)).ingest().units[1].source_id
    assert unit.metadata["note"] == "Note"
    assert unit.metadata["tags"] == ["c"]
    assert PocketHighlightsCsvAdapter(path=str(path)).ingest(entity_types=["article"]).units == []
    assert get_adapter("pocket_highlights_csv", path=str(path)).name == "pocket_highlights_csv"
