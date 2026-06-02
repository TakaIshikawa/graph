from datetime import datetime, timezone

from graph.adapters.google_scholar_library_csv import GoogleScholarLibraryCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_google_scholar_library_csv_ingests_citation_metadata(tmp_path):
    path = tmp_path / "scholar.csv"
    path.write_text("Title,Authors,Publication,Year,Citations,URL,Cluster ID,Labels,Added At\nGraph Paper,\"Ada;Grace\",Journal,2024,12,https://scholar.test,cluster1,\"ml,graphs\",2026-05-02T00:00:00Z\n", encoding="utf-8")

    unit = GoogleScholarLibraryCsvAdapter(str(path)).ingest().units[0]

    assert unit.source_id == "google_scholar_library_csv:cluster1"
    assert unit.metadata["authors"] == ["Ada", "Grace"]
    assert unit.metadata["citations"] == 12
    assert unit.metadata["year"] == 2024
    assert {"scholar", "library", "ml", "graphs"}.issubset(set(unit.tags))


def test_google_scholar_library_csv_since_entity_filter_and_registry(tmp_path):
    path = tmp_path / "scholar.csv"
    path.write_text("title,cluster_id,added_at\nOld,old,2026-04-01T00:00:00Z\nNew,new,2026-05-02T00:00:00Z\n", encoding="utf-8")
    since = SyncState(source_project="google_scholar_library_csv", source_entity_type="scholarly_item", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = GoogleScholarLibraryCsvAdapter(str(path)).ingest(since=since, entity_types=["scholarly_item"])

    assert [unit.source_id for unit in result.units] == ["google_scholar_library_csv:new"]
    assert GoogleScholarLibraryCsvAdapter(str(path)).ingest(entity_types=["book"]).units == []
    assert get_adapter("google_scholar_library_csv", path=str(path)).name == "google_scholar_library_csv"
